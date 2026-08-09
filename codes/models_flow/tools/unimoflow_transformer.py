# UniMoFlow transformer backbone
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional, List, Union, Tuple
from einops import rearrange
from .attention import attention as sdpa_attention


# -----------------------------------------------------------------------------
# 基础组件（复用并简化）
# -----------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim, position):
    """标准正弦位置编码"""
    assert dim % 2 == 0
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device).float().div(half))
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params_1d(max_seq_len, dim, theta=10000, device=None):
    """生成 RoPE 频率"""
    assert dim % 2 == 0
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().to(device) / dim)
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_1d(x, freqs, start_pos=0):
    """应用 RoPE"""
    b, l, n, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, l, n, -1, 2))
    freqs_curr = freqs[start_pos:start_pos + l].view(1, l, 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_curr.to(x_complex.device)).flatten(3)
    return x_out.type_as(x)


class AdaRMSNorm(nn.Module):
    """自适应 RMS Normalization（支持外部 modulation）"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x, modulation_scale=None, modulation_shift=None):
        with torch.amp.autocast("cuda", enabled=False):
            x_float = x.float()
            normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            normed = normed * self.scale.float()
            if modulation_scale is not None:
                normed = normed * (1 + modulation_scale.float())
            if modulation_shift is not None:
                normed = normed + modulation_shift.float()
            else:
                normed = normed + self.shift.float()
            return normed.type_as(x)


class UniMoFlowTransformerBlock(nn.Module):
    """
    UniMoFlow 风格的 Transformer Block（P0: adaLN-Zero 改进版）
    - 使用标准 Pre-LN + Post-LN 结构
    - 【P0】adaLN-Zero 六参数调制替代旧版固定 Gated Residual
    - 纯 Self Attention，所有模态在统一空间交互
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qk_norm: bool = True,
            eps: float = 1e-6,
            use_adaln: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_adaln = use_adaln

        # 第一层：Self Attention
        self.norm1 = AdaRMSNorm(dim, eps)
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.o = nn.Linear(dim, dim)

        # 第二层：FFN
        self.norm2 = AdaRMSNorm(dim, eps)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, dim),
        )

        if use_adaln:
            # 【P0】adaLN-Zero: 6 params = shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim),
            )
            # 零初始化，确保训练初期每个 block 近似恒等映射
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)
        else:
            self.modulation = None
            # Fallback：旧版 gated residual（无 adaLN 时使用）
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

        # 初始化
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.xavier_uniform_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(
            self,
            x: torch.Tensor,
            c: Optional[torch.Tensor] = None,
            seq_lens: Optional[torch.Tensor] = None,
            freqs: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads

        # ----- 计算 adaLN-Zero 调制参数 -----
        if self.use_adaln and c is not None:
            params = self.modulation(c)  # [B, 6*D]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
            shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)
        else:
            shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = None

        # ----- Self Attention with adaLN-Zero -----
        if self.use_adaln:
            h = self.norm1(x, modulation_scale=scale_msa, modulation_shift=shift_msa)
        else:
            h = self.norm1(x)

        # QKV投影
        qkv = self.qkv(h).reshape(B, L, 3, H, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 应用 RoPE
        if freqs is not None:
            q = rope_apply_1d(q, freqs[:L])
            k = rope_apply_1d(k, freqs[:L])

        # Attention计算
        out = sdpa_attention(
            q, k, v,
            q_lens=seq_lens,
            k_lens=None,
            attn_mask=attention_mask,
            causal=False,
        )
        out = rearrange(out, 'b l h d -> b l (h d)')
        out = out.to(self.o.weight.dtype)
        out = self.o(out)

        # 【P0】adaLN-Zero 残差门控
        if self.use_adaln and gate_msa is not None:
            x = x + gate_msa * out
        elif not self.use_adaln:
            gate = torch.tanh(self.gate_attn)
            x = x + out * gate
        else:
            x = x + out

        # ----- FFN with adaLN-Zero -----
        if self.use_adaln:
            h = self.norm2(x, modulation_scale=scale_mlp, modulation_shift=shift_mlp)
        else:
            h = self.norm2(x)

        ffn_out = self.ffn(h)

        # 【P0】adaLN-Zero 残差门控
        if self.use_adaln and gate_mlp is not None:
            x = x + gate_mlp * ffn_out
        elif not self.use_adaln:
            gate = torch.tanh(self.gate_ffn)
            x = x + ffn_out * gate
        else:
            x = x + ffn_out
        return x


class UniMoFlowTransformer(ModelMixin, ConfigMixin):
    """
    UniMoFlow 风格的纯上下文学习编辑模型（P0 adaLN-Zero + P3 动态深度）
    """

    _no_split_modules = ["UniMoFlowTransformerBlock"]

    @register_to_config
    def __init__(
            self,
            in_dim: int = 128,
            out_dim: int = 128,
            dim: int = 1024,
            text_dim: int = 4096,
            num_layers: int = 12,
            num_heads: int = 16,
            mlp_ratio: float = 2.0,
            ffn_dim: Optional[int] = None,
            max_seq_len: int = 2048,
            qk_norm: bool = True,
            eps: float = 1e-6,
            num_registers: int = 0,
            use_sep_token: bool = True,
            source_target_separation: bool = True,
            sep_position: str = "middle",
            use_role_tags: bool = False,
            use_text_tags: bool = False,
            dropout: float = 0.0,
            use_dynamic_depth: bool = False,  # 【P3】
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.num_registers = num_registers
        self.use_sep_token = use_sep_token
        self.source_target_separation = source_target_separation
        self.sep_position = sep_position
        self.use_role_tags = use_role_tags
        self.use_text_tags = use_text_tags
        self.use_dynamic_depth = use_dynamic_depth

        # ---------------------------------------------------------------------
        # 1. 输入编码器
        # ---------------------------------------------------------------------
        freq_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_freq_dim = freq_dim

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.motion_proj = nn.Linear(in_dim, dim)

        # ---------------------------------------------------------------------
        # 2. 特殊 Token
        # ---------------------------------------------------------------------
        if use_role_tags:
            self.target_tag_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.source_tag_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.target_tag_proj[-1].weight)
            nn.init.zeros_(self.target_tag_proj[-1].bias)
            nn.init.zeros_(self.source_tag_proj[-1].weight)
            nn.init.zeros_(self.source_tag_proj[-1].bias)
        else:
            self.target_tag_proj = None
            self.source_tag_proj = None

        if use_text_tags:
            self.gen_text_tag_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.edit_text_tag_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.gen_text_tag_proj[-1].weight)
            nn.init.zeros_(self.gen_text_tag_proj[-1].bias)
            nn.init.zeros_(self.edit_text_tag_proj[-1].weight)
            nn.init.zeros_(self.edit_text_tag_proj[-1].bias)
        else:
            self.gen_text_tag_proj = None
            self.edit_text_tag_proj = None

        self.edit_task_embed = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.gen_task_embed = nn.Parameter(torch.randn(1, dim) * 0.02)

        if use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        else:
            self.sep_token = None

        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, dim) * 0.02)
        else:
            self.register_tokens = None

        if ffn_dim is not None:
            mlp_ratio = ffn_dim / dim
        self.mlp_ratio = mlp_ratio

        # ---------------------------------------------------------------------
        # 3. 核心 Transformer（P0：统一使用 adaLN-Zero，移除旧版 gated_residual）
        # ---------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            UniMoFlowTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        # ---------------------------------------------------------------------
        # 4. 输出头
        # ---------------------------------------------------------------------
        self.edit_norm_final = AdaRMSNorm(dim, eps)
        self.gen_norm_final = AdaRMSNorm(dim, eps)
        self.edit_head = nn.Linear(dim, out_dim)
        self.gen_head = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.edit_head.weight)
        nn.init.zeros_(self.edit_head.bias)
        nn.init.zeros_(self.gen_head.weight)
        nn.init.zeros_(self.gen_head.bias)

        # RoPE 缓存
        head_dim = dim // num_heads
        self.register_buffer("freqs", rope_params_1d(max_seq_len, head_dim), persistent=False)

    def _build_sequence(
            self,
            t_emb: torch.Tensor,
            text_tokens: torch.Tensor,
            target_tokens: torch.Tensor,
            source_tokens: torch.Tensor,
            text_lens: torch.Tensor,
            seq_lens_target: torch.Tensor,
            seq_lens_source: torch.Tensor,
            attention_mode: str = "edit",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], slice]:
        """
        构建非对称注意力序列（支持编辑/生成两种模式）

        Args:
            attention_mode: "edit" - 标准编辑模式，target看到所有token
                           "gen"  - 生成模式，target不看source，source只看自身
        """
        B = t_emb.shape[0]
        device = t_emb.device

        t_token = t_emb.unsqueeze(1)
        L_time = 1

        max_text_len = text_tokens.shape[1]
        actual_max_text = text_lens.max().item()
        text_tokens = text_tokens[:, :actual_max_text]
        L_text = actual_max_text

        L_target = target_tokens.shape[1]
        L_source = source_tokens.shape[1]

        tokens_list = []
        curr_pos = 0

        pos_regs, pos_time, pos_text, pos_target, pos_sep, pos_source = 0, 0, 0, 0, 0, 0
        len_regs = 0

        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1)
            tokens_list.append(registers)
            pos_regs = curr_pos
            len_regs = self.num_registers
            curr_pos += len_regs

        tokens_list.append(t_token)
        pos_time = curr_pos
        curr_pos += L_time

        tokens_list.append(text_tokens)
        pos_text = curr_pos
        curr_pos += L_text

        tokens_list.append(target_tokens)
        pos_target = curr_pos
        curr_pos += L_target

        include_source = attention_mode != "gen"

        if include_source and self.sep_token is not None:
            sep = self.sep_token.expand(B, -1, -1)
            tokens_list.append(sep)
            pos_sep = curr_pos
            curr_pos += 1
        else:
            pos_sep = curr_pos

        if include_source:
            tokens_list.append(source_tokens)
            pos_source = curr_pos
            curr_pos += L_source
        else:
            pos_source = curr_pos
            L_source = 0

        x = torch.cat(tokens_list, dim=1)
        L_total = x.shape[1]

        target_slice = slice(pos_target, pos_target + L_target)

        mask = torch.full((B, L_total, L_total), float('-inf'), device=device)

        # Text Query -> see all
        text_q_start, text_q_end = pos_text, pos_text + L_text
        mask[:, text_q_start:text_q_end, :] = 0.0

        if attention_mode == "gen":
            mask[:, text_q_start:text_q_end, :] = float('-inf')
            mask[:, text_q_start:text_q_end, pos_time:pos_time + L_time] = 0.0
            mask[:, text_q_start:text_q_end, pos_text:pos_text + L_text] = 0.0

            target_q_start, target_q_end = pos_target, pos_target + L_target
            mask[:, target_q_start:target_q_end, pos_time:pos_time + L_time] = 0.0
            mask[:, target_q_start:target_q_end, pos_text:pos_text + L_text] = 0.0
            mask[:, target_q_start:target_q_end, pos_target:pos_target + L_target] = 0.0
        else:
            # Source Query -> 仅 Source + Time（标准编辑模式）
            source_q_start, source_q_end = pos_source, pos_source + L_source
            mask[:, source_q_start:source_q_end, pos_source:pos_source + L_source] = 0.0
            mask[:, source_q_start:source_q_end, pos_time:pos_time + L_time] = 0.0

            # Target Query -> 看到所有（标准编辑模式）
            target_q_start, target_q_end = pos_target, pos_target + L_target
            mask[:, target_q_start:target_q_end, :] = 0.0

        # Time Query -> 仅自身
        mask[:, pos_time:pos_time + L_time, pos_time:pos_time + L_time] = 0.0

        # Registers -> 看到所有
        if self.register_tokens is not None:
            mask[:, pos_regs:pos_regs + len_regs, :] = 0.0

        # Padding Mask
        for b in range(B):
            if text_lens[b] < L_text:
                mask[b, :, pos_text + text_lens[b]:pos_text + L_text] = float('-inf')
            if seq_lens_target[b] < L_target:
                mask[b, :, pos_target + seq_lens_target[b]:pos_target + L_target] = float('-inf')
            if include_source and seq_lens_source[b] < L_source:
                mask[b, :, pos_source + seq_lens_source[b]:pos_source + L_source] = float('-inf')

        attention_mask = mask.unsqueeze(1)
        seq_lens_full = torch.full((B,), L_total, device=device, dtype=torch.long)

        return x, seq_lens_full, attention_mask, target_slice

    def forward(
            self,
            x_target_noisy: torch.Tensor,
            x_source_clean: torch.Tensor,
            t: torch.Tensor,
            context: Union[List[torch.Tensor], torch.Tensor],
            seq_lens: Optional[torch.Tensor] = None,
            active_layers: Optional[int] = None,
            attention_mode: str = "edit",
    ):
        """
        UniMoFlow 风格前向传播（支持编辑/生成混合模式）

        Args:
            active_layers: 【P3】若指定，仅计算前 active_layers 个 block（推理加速）
            attention_mode: "edit" - 标准编辑模式，target看到所有token
                           "gen"  - 生成模式，target不看source，source只看自身
        """
        device = x_target_noisy.device
        B = x_target_noisy.shape[0]
        target_dtype = self.motion_proj.weight.dtype

        # 1. 编码各模态
        t_freq = sinusoidal_embedding_1d(self.time_freq_dim, t).to(device)
        t_emb = self.time_mlp(t_freq)

        if isinstance(context, list):
            text_lens = torch.tensor([c.shape[0] for c in context], device=device)
            max_len = text_lens.max().item()
            text_batch = torch.zeros(B, max_len, context[0].shape[-1], device=device, dtype=target_dtype)
            for i, c in enumerate(context):
                c = c.to(target_dtype) if c.dtype != target_dtype else c
                actual_len = min(c.shape[0], max_len)
                text_batch[i, :actual_len] = c[:actual_len]
                text_lens[i] = actual_len
        else:
            text_batch = context.to(target_dtype) if context.dtype != target_dtype else context
            text_lens = torch.full((B,), text_batch.shape[1], device=device, dtype=torch.long)

        text_tokens = self.text_proj(text_batch)

        if self.use_text_tags:
            if attention_mode == "gen":
                text_tokens = text_tokens + self.gen_text_tag_proj(t_emb).unsqueeze(1)
            else:
                text_tokens = text_tokens + self.edit_text_tag_proj(t_emb).unsqueeze(1)

        x_target = x_target_noisy.to(target_dtype)
        x_source = x_source_clean.to(target_dtype)

        target_tokens = self.motion_proj(x_target)
        source_tokens = self.motion_proj(x_source)

        if self.use_role_tags:
            target_tokens = target_tokens + self.target_tag_proj(t_emb).unsqueeze(1)
            source_tokens = source_tokens + self.source_tag_proj(t_emb).unsqueeze(1)

        T_target = x_target.shape[1]
        T_source = x_source.shape[1]
        if seq_lens is None:
            seq_lens_target = torch.full((B,), T_target, device=device, dtype=torch.long)
        else:
            seq_lens_target = seq_lens
        seq_lens_source = torch.full((B,), T_source, device=device, dtype=torch.long)

        # 2. 构建 UniMoFlow 序列
        x, seq_lens_full, attention_mask, target_slice = self._build_sequence(
            t_emb=t_emb,
            text_tokens=text_tokens,
            target_tokens=target_tokens,
            source_tokens=source_tokens,
            text_lens=text_lens,
            seq_lens_target=seq_lens_target,
            seq_lens_source=seq_lens_source,
            attention_mode=attention_mode,
        )

        # 3. 通过 Denoiser Transformer D
        max_seq_len = x.shape[1]
        if max_seq_len > self.freqs.shape[0]:
            head_dim = self.dim // (self.dim // 64)
            freqs = rope_params_1d(max_seq_len + 128, head_dim, device=device)
        else:
            freqs = self.freqs.to(device)

        # 【P3】动态深度：根据 active_layers 限制计算层数
        total_layers = len(self.blocks)
        if active_layers is None:
            active_layers = total_layers
        else:
            active_layers = min(active_layers, total_layers)

        task_embed = self.edit_task_embed if attention_mode == "edit" else self.gen_task_embed
        c_combined = t_emb + task_embed

        for i in range(active_layers):
            x = self.blocks[i](
                x=x,
                c=c_combined,
                seq_lens=seq_lens_full,
                freqs=freqs,
                attention_mask=attention_mask,
            )

        # 4. 提取 Target 部分并输出
        x_target_out = x[:, target_slice, :]
        if attention_mode == "gen":
            x_target_out = self.gen_norm_final(x_target_out)
            output = self.gen_head(x_target_out)
        else:
            x_target_out = self.edit_norm_final(x_target_out)
            output = self.edit_head(x_target_out)
        return output
