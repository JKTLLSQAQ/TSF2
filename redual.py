import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from utils.RevIN import RevIN


class WaveletDecomposition(nn.Module):
    """多级小波分解模块 - 基于WPMixer设计"""

    def __init__(self, wavelet_name='db4', level=3, channel=1, device='cuda', use_amp=False):
        super().__init__()
        self.wavelet_name = wavelet_name
        self.level = level
        self.channel = channel
        self.device_type = device
        self.use_amp = use_amp

        # 初始化小波变换
        if device == 'cuda':
            self.dwt = DWT1DForward(wave=wavelet_name, J=level, use_amp=use_amp).cuda()
            self.idwt = DWT1DInverse(wave=wavelet_name, use_amp=use_amp).cuda()
        else:
            self.dwt = DWT1DForward(wave=wavelet_name, J=level, use_amp=use_amp)
            self.idwt = DWT1DInverse(wave=wavelet_name, use_amp=use_amp)

        # RevIN归一化
        self.revin_components = nn.ModuleList([
            RevIN(channel, eps=1e-5, affine=True, subtract_last=False)
            for _ in range(level + 1)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T, C] 输入时序数据
        Returns:
            components: 分解后的各小波系数 [approximation, detail1, detail2, ...]
        """
        B, T, C = x.shape

        # 转换维度: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # 小波分解
        yl, yh = self.dwt(x)  # yl: 近似系数, yh: 细节系数列表

        # 组织分解结果
        components = []

        # 近似系数归一化 [B, C, T_approx] -> [B, T_approx, C]
        yl = yl.transpose(1, 2)
        yl_norm = self.revin_components[0](yl, 'norm')
        components.append(yl_norm.transpose(1, 2))  # 转回 [B, C, T_approx]

        # 各级细节系数归一化
        for i, yh_i in enumerate(yh):
            yh_i = yh_i.transpose(1, 2)  # [B, C, T_detail] -> [B, T_detail, C]
            yh_i_norm = self.revin_components[i + 1](yh_i, 'norm')
            components.append(yh_i_norm.transpose(1, 2))  # 转回 [B, C, T_detail]

        return components

    def inverse_transform(self, components):
        """
        Args:
            components: 各小波系数 [approximation, detail1, detail2, ...]
        Returns:
            x: [B, T, C] 重构的时序数据
        """
        # 反归一化
        yl = components[0].transpose(1, 2)  # [B, C, T] -> [B, T, C]
        yl = self.revin_components[0](yl, 'denorm')
        yl = yl.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        yh = []
        for i, comp in enumerate(components[1:]):
            comp = comp.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            comp = self.revin_components[i + 1](comp, 'denorm')
            yh.append(comp.transpose(1, 2))  # [B, T, C] -> [B, C, T]

        # 小波重构
        x = self.idwt((yl, yh))  # [B, C, T]

        return x.transpose(1, 2)  # [B, C, T] -> [B, T, C]


class ChannelAttentionModule(nn.Module):
    """通道注意力机制"""

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
        B, T, C = x.shape

        avg_out = self.fc(self.avg_pool(x.transpose(1, 2)).squeeze(-1))
        max_out = self.fc(self.max_pool(x.transpose(1, 2)).squeeze(-1))

        attention_weights = self.sigmoid(avg_out + max_out).unsqueeze(1)
        return x * attention_weights


class WaveletComponentProcessor(nn.Module):
    """小波系数专用处理器"""

    def __init__(self, d_model, component_type='approximation'):
        super().__init__()
        self.component_type = component_type

        # 根据分量类型选择不同的处理策略
        if component_type == 'approximation':
            # 近似系数：低频信息，使用较低阶的KAN
            self.kan_layer = ChebyKANLinear(d_model, d_model, order=3)
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        else:
            # 细节系数：高频信息，使用较高阶的KAN
            self.kan_layer = ChebyKANLinear(d_model, d_model, order=5)
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: [B, T, C] 小波系数嵌入
        """
        B, T, C = x.shape

        # KAN处理
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)

        # 卷积处理
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)

        # 残差连接和归一化
        return self.norm(x + self.dropout(x_kan + x_conv))


class EnhancedFrequencyBranch(nn.Module):
    """增强的频域分支 - 基于多级小波分解"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.level = getattr(args, 'wavelet_level', 3)

        # 为各小波系数创建专用处理器
        self.component_processors = nn.ModuleList([
            WaveletComponentProcessor(args.d_model, 'approximation')  # 近似系数
        ])

        # 各级细节系数处理器
        for i in range(self.level):
            self.component_processors.append(
                WaveletComponentProcessor(args.d_model, f'detail_{i + 1}')
            )

        # 多头注意力用于分量间交互
        self.cross_component_attention = nn.MultiheadAttention(
            args.d_model, num_heads=4, dropout=0.1, batch_first=True
        )

        # 特征融合
        self.freq_fusion = nn.Sequential(
            nn.Linear(args.d_model * (self.level + 1), args.d_model * 2),
            nn.GELU(),
            nn.Linear(args.d_model * 2, args.d_model),
            nn.Dropout(0.1)
        )

        # 频域预测头
        self.temporal_predictor = nn.Linear(args.seq_len, args.pred_len)

        # 输出投影
        if args.channel_independence == 1:
            self.output_projection = nn.Linear(args.d_model, 1)
        else:
            self.output_projection = nn.Linear(args.d_model, args.c_out)

    def forward(self, wavelet_components_emb):
        """
        Args:
            wavelet_components_emb: List of [B, T, d_model] 各小波系数的嵌入
        """
        processed_components = []

        # 各分量独立处理
        for i, (component_emb, processor) in enumerate(zip(wavelet_components_emb, self.component_processors)):
            processed_comp = processor(component_emb)
            processed_components.append(processed_comp)

        # 分量间交互注意力
        if len(processed_components) > 1:
            # 拼接所有分量 [B, T*(level+1), d_model]
            all_components = torch.cat(processed_components, dim=1)

            # 自注意力交互
            attended_components, _ = self.cross_component_attention(
                all_components, all_components, all_components
            )

            # 重新分割
            T = processed_components[0].shape[1]
            split_components = torch.split(attended_components, T, dim=1)
        else:
            split_components = processed_components

        # 在特征维度融合
        if len(split_components) > 1:
            # 对齐序列长度（取最短）
            min_len = min(comp.shape[1] for comp in split_components)
            aligned_components = [comp[:, :min_len, :] for comp in split_components]

            # 特征维度拼接
            combined_freq = torch.cat(aligned_components, dim=-1)
            freq_features = self.freq_fusion(combined_freq)
        else:
            freq_features = split_components[0]

        # 频域预测
        freq_pred = self.temporal_predictor(freq_features.transpose(1, 2)).transpose(1, 2)
        freq_output = self.output_projection(freq_pred)

        return freq_output, freq_features


class BalancedTimeBranch(nn.Module):
    """平衡的时域分支 - 简化设计"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # 统一KAN混合器（简化版）
        self.unified_mixer = nn.Sequential(
            ChebyKANLinear(args.d_model * 3, args.d_model * 2, order=3),
            nn.GELU(),
            ChebyKANLinear(args.d_model * 2, args.d_model, order=3),
            nn.Dropout(0.1)
        )

        # 简化的正则化模块
        self.light_regularization = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.Dropout(0.1),
            nn.LayerNorm(args.d_model)
        )

        # 时域预测头
        self.temporal_predictor = nn.Linear(args.seq_len, args.pred_len)

        # 输出投影
        if args.channel_independence == 1:
            self.output_projection = nn.Linear(args.d_model, 1)
        else:
            self.output_projection = nn.Linear(args.d_model, args.c_out)

    def forward(self, seasonal_emb, trend_emb, residual_emb):
        """使用平均长度对齐不同小波系数"""
        # 获取最小长度
        min_len = min(seasonal_emb.shape[1], trend_emb.shape[1], residual_emb.shape[1])

        # 截取到相同长度
        seasonal_aligned = seasonal_emb[:, :min_len, :]
        trend_aligned = trend_emb[:, :min_len, :]
        residual_aligned = residual_emb[:, :min_len, :]

        # 特征融合
        combined = torch.cat([seasonal_aligned, trend_aligned, residual_aligned], dim=-1)
        time_features = self.unified_mixer(combined)

        # 轻量级正则化
        time_features = self.light_regularization(time_features)

        # 时域预测
        time_pred = self.temporal_predictor(time_features.transpose(1, 2)).transpose(1, 2)
        time_output = self.output_projection(time_pred)

        return time_output, time_features


class ProgressiveInteractionAttention(nn.Module):
    """渐进式交互注意力融合模块 - 基于TwinsFormer设计"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # 多层交互
        self.interaction_layers = nn.ModuleList([
            self._build_interaction_layer(args) for _ in range(2)
        ])

        # 最终注意力权重计算
        if args.channel_independence == 1:
            input_dim = 1
        else:
            input_dim = args.c_out

        self.final_attention = nn.Sequential(
            nn.Linear(input_dim * 2 + args.d_model * 2, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 2),
            nn.Softmax(dim=-1)
        )

        # 置信度估计
        self.confidence_net = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Linear(args.d_model // 2, 1),
            nn.Sigmoid()
        )

        # 门控机制
        self.gate_mechanism = nn.Sequential(
            nn.Conv1d(args.d_model, args.d_model, 1),
            nn.Sigmoid()
        )

    def _build_interaction_layer(self, args):
        return nn.MultiheadAttention(
            args.d_model, num_heads=4, dropout=0.1, batch_first=True
        )

    def forward(self, time_pred, freq_pred, time_features, freq_features):
        B, pred_len = time_pred.shape[:2]

        # 渐进式交互
        for interaction_layer in self.interaction_layers:
            # 交互增强
            enhanced_time, _ = interaction_layer(time_features, freq_features, freq_features)
            enhanced_freq, _ = interaction_layer(freq_features, time_features, time_features)

            # 使用减法机制（参考TwinsFormer）
            time_features = time_features - enhanced_freq * 0.1
            freq_features = freq_features - enhanced_time * 0.1

        # 计算全局特征
        time_global = time_features.mean(dim=1)
        freq_global = freq_features.mean(dim=1)

        # 预测差异
        pred_diff = torch.abs(time_pred - freq_pred).mean(dim=1)

        # 注意力权重
        attention_input = torch.cat([
            time_global, freq_global,
            pred_diff, (time_pred + freq_pred).mean(dim=1)
        ], dim=-1)

        attention_weights = self.final_attention(attention_input)

        # 置信度
        time_conf = self.confidence_net(time_global)
        freq_conf = self.confidence_net(freq_global)

        # 门控融合
        combined_features = time_features + freq_features
        gate = self.gate_mechanism(combined_features.transpose(1, 2)).transpose(1, 2)

        # 最终融合
        conf_weights = torch.cat([time_conf, freq_conf], dim=-1)
        final_weights = attention_weights * conf_weights
        final_weights = F.softmax(final_weights, dim=-1)

        final_pred = (final_weights[:, 0:1].unsqueeze(1) * time_pred +
                      final_weights[:, 1:2].unsqueeze(1) * freq_pred)

        return final_pred, {
            'attention_weights': attention_weights,
            'confidence_weights': conf_weights,
            'final_weights': final_weights,
            'time_confidence': time_conf,
            'freq_confidence': freq_conf
        }


class Model(nn.Module):
    """改进的预测级融合双分支模型 - 集成多级小波分解"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        # 小波分解参数（使用getattr提供默认值）
        self.wavelet_name = getattr(args, 'wavelet_name', 'db4')
        self.wavelet_level = getattr(args, 'wavelet_level', 3)

        # 1. 通道注意力机制
        self.channel_attention = ChannelAttentionModule(args.enc_in, reduction=8)

        # 2. 多级小波分解模块
        self.wavelet_decomposition = WaveletDecomposition(
            wavelet_name=self.wavelet_name,
            level=self.wavelet_level,
            channel=args.enc_in,
            device=getattr(args, 'device', 'cuda'),
            use_amp=getattr(args, 'use_amp', False)
        )

        # 3. 嵌入层
        if args.channel_independence == 1:
            self.component_embeddings = nn.ModuleList([
                DataEmbedding_wo_pos(1, args.d_model, args.embed, args.freq, args.dropout)
                for _ in range(self.wavelet_level + 1)
            ])
        else:
            self.component_embeddings = nn.ModuleList([
                DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
                for _ in range(self.wavelet_level + 1)
            ])

        # 4. 平衡的时域分支
        self.time_branch = BalancedTimeBranch(args)

        # 5. 增强的频域分支
        self.freq_branch = EnhancedFrequencyBranch(args)

        # 6. 渐进式交互注意力融合
        self.progressive_attention = ProgressiveInteractionAttention(args)

        # 7. 归一化
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(args.enc_in, affine=True, non_norm=True if args.use_norm == 0 else False)
            for i in range(args.down_sampling_layers + 1)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 1. 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # 2. 通道注意力
        x_attended = self.channel_attention(x_enc)

        # 3. 多级小波分解
        wavelet_components = self.wavelet_decomposition(x_attended)

        # 4. 处理通道独立性并嵌入各小波系数
        component_embeddings = []

        for i, component in enumerate(wavelet_components):
            if self.configs.channel_independence == 1:
                # [B, C, T] -> [B*N, T, 1]
                component = component.permute(0, 2, 1).contiguous().reshape(B * N, component.shape[-1], 1)
                if x_mark_enc is not None:
                    x_mark_repeated = x_mark_enc.repeat(N, 1, 1)
                else:
                    x_mark_repeated = None
            else:
                # [B, C, T] -> [B, T, C]
                component = component.transpose(1, 2)
                x_mark_repeated = x_mark_enc

            # 嵌入
            component_emb = self.component_embeddings[i](component, x_mark_repeated)
            component_embeddings.append(component_emb)

        # 5. 为分支准备输入（使用前3个主要分量）
        # 近似系数作为trend，第一个细节系数作为seasonal，第二个细节系数作为residual
        if len(component_embeddings) >= 3:
            trend_emb = component_embeddings[0]  # 近似系数（低频趋势）
            seasonal_emb = component_embeddings[1]  # 第一级细节系数
            residual_emb = component_embeddings[2]  # 第二级细节系数
        else:
            # 如果分解级数较少，适当处理
            trend_emb = component_embeddings[0]
            seasonal_emb = component_embeddings[1] if len(component_embeddings) > 1 else component_embeddings[0]
            residual_emb = component_embeddings[-1]

        # 6. 双分支预测
        time_pred, time_features = self.time_branch(seasonal_emb, trend_emb, residual_emb)
        freq_pred, freq_features = self.freq_branch(component_embeddings)

        # 7. 渐进式交互注意力融合
        final_pred, attention_info = self.progressive_attention(
            time_pred, freq_pred, time_features, freq_features
        )

        # 8. 输出重塑
        if self.args.channel_independence == 1:
            final_pred = final_pred.reshape(B, N, self.pred_len, -1)
            if final_pred.shape[-1] == 1:
                final_pred = final_pred.squeeze(-1)
            final_pred = final_pred.permute(0, 2, 1).contiguous()

        # 确保输出维度正确
        if final_pred.shape[-1] > self.args.c_out:
            final_pred = final_pred[..., :self.args.c_out]

        # 9. 反归一化
        final_pred = self.normalize_layers[0](final_pred, 'denorm')

        # 存储注意力信息
        self.last_attention_info = attention_info

        return final_pred

    def get_branch_contributions(self):
        """获取各分支的贡献度分析"""
        if hasattr(self, 'last_attention_info'):
            return self.last_attention_info
        else:
            return None