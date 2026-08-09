import torch
import torch.nn as nn
import numpy as np
from .tools.hrvae_modules import HRVAEBackbone

def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    """变长序列mask生成（与HRVQVAE保持一致）"""
    if device is None:
        device = "cpu"
    if isinstance(length, list):
        length = torch.tensor(length)
    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


class HRVAE(nn.Module):
    """
    基于连续潜空间的层级VAE，接口与HRVQVAE完全兼容。

    主要差异:
    - encode返回(None, mu) 代替 (code_idx, all_codes)
    - forward返回(x_out, kl_loss, perplexity=0) 代替 (x_out, commit_loss, perplexity)
    - 支持流式编解码(stream_encode/stream_decode)
    """

    def __init__(
            self,
            # args=None,  # 保持与HRVQVAE一致的配置接口
            input_width=263,
            z_dim=16,
            dim=160,
            dec_dim=512,
            num_res_blocks=1,
            dropout=0.0,
            dim_mult=[1, 1, 1],
            temperal_downsample=[True, True],
            ):
        super().__init__()
        # self.cfg = args if args is not None else {}
        self.input_width = input_width
        self.z_dim = z_dim

        # 核心VAE模型（修改自HRVAEBackbone以支持1D时序）
        self.model = HRVAEBackbone(
            input_dim=input_width,
            dim=dim,
            dec_dim=dec_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
        )

        # 计算时序下采样因子（用于长度对齐）
        downsample_factor = 1
        for flag in temperal_downsample:
            if flag:
                downsample_factor *= 2
        self.downsample_factor = downsample_factor

    # ------------------- 数据预处理（与HRVQVAE完全一致） -------------------
    def preprocess(self, x):
        """(bs, T, C) -> (bs, C, T)"""
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        """(bs, C, T) -> (bs, T, C)"""
        x = x.permute(0, 2, 1)
        return x

    # ------------------- 编码/解码接口（兼容HRVQVAE） -------------------
    def encode(self, x, m_lens=None):
        """
        参数:
            x: (bs, T, input_width) 输入特征
            m_lens: [可选] 各序列实际长度列表

        返回:
            code_idx: None (连续VAE无离散索引)
            all_codes: (bs, T_latent, z_dim) 潜变量均值mu
                      T_latent = T // downsample_factor
        """
        x_in = self.preprocess(x)  # (bs, C, T)

        # 编码 (bs, z_dim*2, T_latent) -> 分割为mu, log_var
        mu, log_var = self.model.encode(x_in, scale=[0, 1], return_dist=True)
        # z = self.model.reparameterize(mu, log_var)
        # 转换为HRVQVAE格式: (bs, T_latent, z_dim)
        mu = self.postprocess(mu)
        return mu

    def decode(self, x, m_lengths=None):
        """
        参数:
            x: (bs, T_latent, z_dim) 潜变量（即encode返回的all_codes/mu）
            m_lengths: [可选] 原始序列长度（用于padding mask）

        返回:
            x_out: (bs, T, input_width) 重构输出
        """
        # 维度转换: (bs, T_latent, z_dim) -> (bs, z_dim, T_latent)
        x_in = self.preprocess(x)

        # 处理长度mask（在latent空间）
        if m_lengths is not None:
            m_lengths_latent = m_lengths // self.downsample_factor
            mask = length_to_mask(m_lengths_latent, x_in.shape[2], x_in.device)
            x_in = x_in.permute(0, 2, 1)  # (bs, T_latent, z_dim)
            x_in[~mask] = 0
            x_in = x_in.permute(0, 2, 1)  # 转回 (bs, z_dim, T_latent)

        # 解码
        x_decoder = self.model.decode(x_in, scale=[0, 1])
        x_out = self.postprocess(x_decoder)
        return x_out

    def forward(self, x, m_lengths=None):
        """
        参数:
            x: (bs, T, input_width)
            m_lengths: (bs,) 实际长度

        返回:
            x_out: (bs, T', input_width) 重构输出，T'为对齐后的长度
            commit_loss: 标量（VAE中为KL散度，对应离散VAE的commitment loss）
            perplexity: 0.0（连续VAE无困惑度概念）
        """
        # 归一化与维度转换
        # x_norm = (x - self.mean) / self.std
        x_in = self.preprocess(x)

        # 编码
        mu, log_var = self.model.encode(x_in, scale=[0, 1], return_dist=True)

        # 重参数化采样
        z = self.model.reparameterize(mu, log_var)

        # 解码
        x_decoder = self.model.decode(z, scale=[0, 1])
        x_out = self.postprocess(x_decoder)

        # 长度对齐（处理下采样的边界效应）
        T_orig, T_out = x.shape[1], x_out.shape[1]
        if T_out != T_orig:
            min_len = min(T_orig, T_out)
            x_out = x_out[:, :min_len, :]
            # 同步截断原始数据以便后续可能的reconstruction loss计算
            # x_norm = x_norm[:, :min_len, :]

        # 计算KL Loss（对应离散VAE的commit_loss位置）
        # 应用mask处理变长序列
        if m_lengths is not None:
            T_latent = mu.shape[2]
            m_lengths_latent = m_lengths.clamp(min=1) // self.downsample_factor
            # 构建latent空间mask: (bs, 1, T_latent)
            mask = torch.zeros(x.shape[0], T_latent, device=x.device)
            for i in range(x.shape[0]):
                mask[i, :m_lengths_latent[i]] = 1.0
            mask = mask.unsqueeze(1)
            # 计算masked KL: -0.5 * (1 + log_var - mu^2 - exp(log_var))
            kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_masked = kl_per_element * mask
            commit_loss = torch.sum(kl_masked) / torch.sum(mask)
        else:
            commit_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        if isinstance(commit_loss, torch.Tensor) and commit_loss.dim() > 0:
            commit_loss = commit_loss.mean()
        perplexity = torch.tensor(0.0, device=x.device)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x, m_lengths=None):
        """
        兼容HRVQVAE.forward_decoder接口
        x: (bs, T_latent, z_dim) 来自encode的all_codes
        """
        return self.decode(x, m_lengths)
    # ------------------- 流式接口（继承自HRVAE能力） -------------------
    @torch.no_grad()
    def stream_encode(self, x, first_chunk=True):
        """流式编码（用于实时推理）"""
        x = (x - self.mean) / self.std
        x_in = self.preprocess(x)
        mu = self.model.stream_encode(x_in, first_chunk=first_chunk, scale=[0, 1])
        mu = self.postprocess(mu)
        return mu
    @torch.no_grad()
    def stream_decode(self, mu, first_chunk=True):
        """流式解码"""
        mu_in = self.preprocess(mu)
        x_decoder = self.model.stream_decode(mu_in, first_chunk=first_chunk, scale=[0, 1])
        x_out = self.postprocess(x_decoder)
        x_out = x_out * self.std + self.mean
        return x_out
    def clear_cache(self):
        """清除流式推理缓存"""
        self.model.clear_cache()
