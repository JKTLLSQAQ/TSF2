import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion


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


class ComponentProcessor(nn.Module):
    """分量处理器"""

    def __init__(self, configs, component_type):
        super().__init__()
        self.component_type = component_type

        if component_type == 'trend':
            self.processor = nn.Sequential(
                AdaptiveKANMixer(configs.d_model, 'trend'),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout)
            )
        elif component_type == 'seasonal':
            self.diffusion = LightweightDiffusion(time_steps=20, device=configs.device)
            self.processor = AdaptiveKANMixer(configs.d_model, 'seasonal')
        else:  # residual
            self.processor = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
                nn.Dropout(configs.dropout)
            )

    def forward(self, x):
        if self.component_type == 'seasonal' and self.training:
            x_noise, noise, t = self.diffusion(x, apply_noise=True)
            return self.processor(x_noise)
        else:
            return self.processor(x)


class Model(nn.Module):
    """融合时序模型主体"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 分解模块
        self.decomposition = series_decomp(configs.moving_avg)

        # 嵌入层
        if configs.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 分量处理器
        self.trend_processor = ComponentProcessor(configs, 'trend')
        self.seasonal_processor = ComponentProcessor(configs, 'seasonal')
        self.residual_processor = ComponentProcessor(configs, 'residual')

        # 归一化
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for i in range(configs.down_sampling_layers + 1)
        ])

        # 预测层
        self.trend_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.seasonal_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.residual_predictor = nn.Linear(configs.seq_len, configs.pred_len)

        # 输出投影
        if configs.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 可学习融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # 分解
        seasonal, trend = self.decomposition(x_enc)
        residual = x_enc - seasonal - trend

        # 通道独立性处理
        if self.configs.channel_independence == 1:
            seasonal = seasonal.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            trend = trend.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            residual = residual.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 嵌入
        if self.configs.channel_independence == 1 and x_mark_enc is not None:
            x_mark_enc_expanded = x_mark_enc.repeat(N, 1, 1)  # [B*N, T, mark_dim]
        else:
            x_mark_enc_expanded = x_mark_enc

        seasonal_emb = self.enc_embedding(seasonal, x_mark_enc_expanded)
        trend_emb = self.enc_embedding(trend, x_mark_enc_expanded)
        residual_emb = self.enc_embedding(residual, x_mark_enc_expanded)

        # 分量处理
        seasonal_out = self.seasonal_processor(seasonal_emb)
        trend_out = self.trend_processor(trend_emb)
        residual_out = self.residual_processor(residual_emb)

        # 时序预测
        seasonal_pred = self.seasonal_predictor(seasonal_out.permute(0, 2, 1)).permute(0, 2, 1)
        trend_pred = self.trend_predictor(trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_predictor(residual_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 投影
        seasonal_pred = self.projection_layer(seasonal_pred)
        trend_pred = self.projection_layer(trend_pred)
        residual_pred = self.projection_layer(residual_pred)

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        dec_out = (weights[0] * trend_pred +
                   weights[1] * seasonal_pred +
                   weights[2] * residual_pred)

        # 输出重塑
        if self.configs.channel_independence == 1:
            dec_out = dec_out.reshape(B, N, self.pred_len, -1)
            if dec_out.shape[-1] == 1:
                dec_out = dec_out.squeeze(-1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

        if dec_out.shape[-1] > self.configs.c_out:
            dec_out = dec_out[..., :self.configs.c_out]

        # 反归一化
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out