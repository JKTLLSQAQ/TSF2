import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion


class ChannelAttentionModule(nn.Module):
    """通道注意力机制 - 放在模型最前端"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        # 全局平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x.transpose(1, 2)).squeeze(-1))  # [B, C]
        max_out = self.fc(self.max_pool(x.transpose(1, 2)).squeeze(-1))  # [B, C]

        # 注意力权重
        attention_weights = self.sigmoid(avg_out + max_out).unsqueeze(1)  # [B, 1, C]

        # 应用注意力权重
        return x * attention_weights


class FourierBlock(nn.Module):
    """频域特征提取模块（基于论文中的FNN设计）"""

    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # 频域映射层
        self.freq_mapping = nn.Linear(d_model, d_model)

        # 多头自注意力机制用于频域特征学习
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)

        # 频域特征变换
        self.freq_transform = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )

        # 逆FFT后的处理
        self.ifft_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        # 映射到高维空间
        x_mapped = self.freq_mapping(x)

        # FFT变换到频域
        x_freq = torch.fft.rfft(x_mapped, dim=1)  # [B, T//2+1, C]
        x_freq_real = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # [B, T//2+1, 2C]

        # 调整维度以匹配d_model
        if x_freq_real.shape[-1] != self.d_model:
            # 创建一个可学习的线性层来调整维度
            if not hasattr(self, 'freq_dim_adapter'):
                self.freq_dim_adapter = nn.Linear(x_freq_real.shape[-1], self.d_model).to(x.device)
            x_freq_real = self.freq_dim_adapter(x_freq_real)

        # 自注意力机制学习频域特征
        freq_attn_out, _ = self.multihead_attn(x_freq_real, x_freq_real, x_freq_real)

        # 频域特征变换
        freq_features = self.freq_transform(freq_attn_out)

        # IFFT回到时域
        # 处理维度以进行IFFT
        if freq_features.shape[-1] >= self.d_model // 2:
            # 分离实部和虚部
            half_dim = freq_features.shape[-1] // 2
            freq_real = freq_features[..., :half_dim]
            freq_imag = freq_features[..., half_dim:half_dim * 2] if freq_features.shape[
                                                                         -1] > half_dim else torch.zeros_like(freq_real)
            freq_complex = torch.complex(freq_real, freq_imag)
        else:
            # 如果维度不足，用零填充虚部
            freq_complex = torch.complex(freq_features, torch.zeros_like(freq_features))

        # 确保频域序列长度正确
        if freq_complex.shape[1] != (T // 2 + 1):
            # 调整频域序列长度
            target_freq_len = T // 2 + 1
            current_freq_len = freq_complex.shape[1]
            if current_freq_len > target_freq_len:
                freq_complex = freq_complex[:, :target_freq_len, :]
            else:
                # 用零填充
                pad_len = target_freq_len - current_freq_len
                pad_tensor = torch.zeros(freq_complex.shape[0], pad_len, freq_complex.shape[2],
                                         dtype=freq_complex.dtype, device=freq_complex.device)
                freq_complex = torch.cat([freq_complex, pad_tensor], dim=1)

        x_ifft = torch.fft.irfft(freq_complex, n=T, dim=1)  # [B, T, C]

        # 最终投影，确保输出维度与输入一致
        if x_ifft.shape[-1] != C:
            if not hasattr(self, 'output_adapter'):
                self.output_adapter = nn.Linear(x_ifft.shape[-1], C).to(x.device)
            x_ifft = self.output_adapter(x_ifft)

        return self.ifft_proj(x_ifft)


class LightweightDiffusion(nn.Module):
    """轻量级扩散模块"""

    def __init__(self, time_steps=20, device='cuda', scheduler='linear'):
        super().__init__()
        self.diffusion = Diffusion(time_steps=time_steps, device=device, scheduler=scheduler)

    def forward(self, x, apply_noise=True):
        if apply_noise and self.training:
            return self.diffusion(x)
        else:
            return x, None, None


class AdaptiveKANMixer(nn.Module):
    """自适应KAN混合器"""

    def __init__(self, d_model, component_type='trend'):
        super().__init__()
        # 根据分量类型选择KAN阶数
        order_map = {'trend': 3, 'seasonal': 5, 'residual': 4}
        order = order_map.get(component_type, 4)

        self.kan_layer = ChebyKANLinear(d_model, d_model, order)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + x_kan + x_conv)


class TimeDomainBranch(nn.Module):
    """时域分支（基于原有模型）"""

    def __init__(self, configs):
        super().__init__()

        # KAN混合器
        self.trend_mixer = AdaptiveKANMixer(configs.d_model, 'trend')
        self.seasonal_mixer = AdaptiveKANMixer(configs.d_model, 'seasonal')
        self.residual_mixer = AdaptiveKANMixer(configs.d_model, 'residual')

        # 轻量级扩散（仅用于seasonal）
        self.diffusion = LightweightDiffusion(time_steps=20, device=configs.device)

        # 特征融合
        self.feature_fusion = nn.Linear(configs.d_model * 3, configs.d_model)

    def forward(self, seasonal_emb, trend_emb, residual_emb):
        # 分量处理
        trend_out = self.trend_mixer(trend_emb)

        # seasonal加入扩散噪声
        if self.training:
            seasonal_noise, _, _ = self.diffusion(seasonal_emb, apply_noise=True)
            seasonal_out = self.seasonal_mixer(seasonal_noise)
        else:
            seasonal_out = self.seasonal_mixer(seasonal_emb)

        residual_out = self.residual_mixer(residual_emb)

        # 特征融合
        combined = torch.cat([trend_out, seasonal_out, residual_out], dim=-1)
        return self.feature_fusion(combined)


class FrequencyDomainBranch(nn.Module):
    """频域分支（基于论文设计）"""

    def __init__(self, configs):
        super().__init__()

        # 频域处理器
        self.fourier_trend = FourierBlock(configs.d_model, configs.seq_len)
        self.fourier_seasonal = FourierBlock(configs.d_model, configs.seq_len)
        self.fourier_residual = FourierBlock(configs.d_model, configs.seq_len)

        # 频域特征融合
        self.freq_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 3, configs.d_model * 2),
            nn.GELU(),
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.Dropout(0.1)
        )

    def forward(self, seasonal_emb, trend_emb, residual_emb):
        # 频域处理
        trend_freq = self.fourier_trend(trend_emb)
        seasonal_freq = self.fourier_seasonal(seasonal_emb)
        residual_freq = self.fourier_residual(residual_emb)

        # 频域特征融合
        combined_freq = torch.cat([trend_freq, seasonal_freq, residual_freq], dim=-1)
        return self.freq_fusion(combined_freq)


class Model(nn.Module):
    """双分支时频域融合模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 1. 通道注意力机制（放在最前端）
        self.channel_attention = ChannelAttentionModule(configs.enc_in, reduction=8)

        # 2. 分解模块
        self.decomposition = series_decomp(configs.moving_avg)

        # 3. 嵌入层
        if configs.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 4. 时域分支
        self.time_branch = TimeDomainBranch(configs)

        # 5. 频域分支
        self.freq_branch = FrequencyDomainBranch(configs)

        # 6. 特征融合层
        self.fusion_conv = nn.Conv1d(configs.d_model * 2, configs.d_model, kernel_size=1)
        self.fusion_attention = nn.MultiheadAttention(configs.d_model, num_heads=8, batch_first=True)

        # 7. 时序预测层
        self.temporal_predictor = nn.Linear(configs.seq_len, configs.pred_len)

        # 8. 输出投影层
        if configs.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 9. 归一化
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for i in range(configs.down_sampling_layers + 1)
        ])

        # 10. 可学习融合权重
        self.branch_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))  # time, freq, fused

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 1. 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # 2. 通道注意力（最前端）
        x_attended = self.channel_attention(x_enc)

        # 3. 分解
        seasonal, trend = self.decomposition(x_attended)
        residual = x_attended - seasonal - trend

        # 4. 处理通道独立性
        if self.configs.channel_independence == 1:
            seasonal = seasonal.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            trend = trend.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            residual = residual.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc.repeat(N, 1, 1)

        # 5. 嵌入
        seasonal_emb = self.enc_embedding(seasonal, x_mark_enc)
        trend_emb = self.enc_embedding(trend, x_mark_enc)
        residual_emb = self.enc_embedding(residual, x_mark_enc)

        # 6. 双分支处理
        # 时域分支
        time_features = self.time_branch(seasonal_emb, trend_emb, residual_emb)

        # 频域分支
        freq_features = self.freq_branch(seasonal_emb, trend_emb, residual_emb)

        # 7. 特征融合
        # 拼接两个分支的特征
        combined_features = torch.cat([time_features, freq_features], dim=-1)  # [B, T, 2*d_model]

        # 1D卷积融合
        fused_features = self.fusion_conv(combined_features.transpose(1, 2)).transpose(1, 2)  # [B, T, d_model]

        # 注意力融合
        fused_features, _ = self.fusion_attention(fused_features, fused_features, fused_features)

        # 加权融合
        weights = F.softmax(self.branch_weights, dim=0)
        final_features = weights[0] * time_features + weights[1] * freq_features + weights[2] * fused_features

        # 8. 时序预测
        pred_features = self.temporal_predictor(final_features.transpose(1, 2)).transpose(1, 2)

        # 9. 输出投影
        dec_out = self.projection_layer(pred_features)

        # 10. 输出重塑
        if self.configs.channel_independence == 1:
            dec_out = dec_out.reshape(B, N, self.pred_len, -1)
            if dec_out.shape[-1] == 1:
                dec_out = dec_out.squeeze(-1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

        # 确保输出维度正确
        if dec_out.shape[-1] > self.configs.c_out:
            dec_out = dec_out[..., :self.configs.c_out]

        # 11. 反归一化
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out