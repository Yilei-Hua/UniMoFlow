# Omni-MoEdit motion DiT backbone
import math
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from models_flow.tools.attention import flash_attention as attention


# -----------------------------------------------------------------------------
# 1D Positional Embeddings
# -----------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim, position):
    # standard sinusoidal embedding
    assert dim % 2 == 0
    half = dim // 2
    # Ensure calculation happens in high precision
    position = position.float()
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device).float().div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

@torch.amp.autocast("cuda", enabled=False)
def rope_params_1d(max_seq_len, dim, theta=10000):
    """Generates RoPE frequencies for 1D sequences."""
    assert dim % 2 == 0
    # Shape: [dim/2]
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float() / dim)
    # Shape: [max_seq_len]
    t = torch.arange(max_seq_len).float()
    # Shape: [max_seq_len, dim/2]
    freqs = torch.outer(t, freqs)
    # Polar form for rotation
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_1d(x, freqs):
    """
    Applies RoPE to x.
    Args:
        x: [B, L, Num_Heads, Head_Dim]
        freqs: [Max_L, Head_Dim/2] (Complex)
    """
    # x: [B, L, H, D]
    b, l, n, d = x.shape

    # Reshape x for complex multiplication: [B, L, H, D/2, 2] -> complex [B, L, H, D/2]
    # Explicitly cast to float for precision during rotation
    x_complex = torch.view_as_complex(x.float().reshape(b, l, n, -1, 2))

    # Slice freqs to current length L: [L, D/2] -> [1, L, 1, D/2]
    freqs_curr = freqs[:l].view(1, l, 1, -1)

    # Rotate
    x_out = torch.view_as_real(x_complex * freqs_curr.to(x_complex.device)).flatten(3)

    # Cast back to input dtype (fp16/bf16)
    return x_out.type_as(x)


# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------

class OmniMoEditRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # [MixPrecision Fix]
        # Calculate Norm in FP32, Apply Weight in FP32, then cast back.
        # This prevents underflow in FP16 during the squaring operation.
        with torch.amp.autocast("cuda", enabled=False):
            x_float = x.float()
            normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            out = normed * self.weight.float()
        return out.type_as(x)


class OmniMoEditLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        # [MixPrecision Fix] Force FP32 for LayerNorm stats
        with torch.amp.autocast("cuda", enabled=False):
            out = super().forward(x.float())
        return out.type_as(x)


class OmniMoEditSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, causal=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.causal = causal

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # QK Norm is crucial for training stability in large models
        self.norm_q = OmniMoEditRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = OmniMoEditRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, freqs):
        """
        Args:
            x: [B, L, C] -> reshaped inside to [B, L, H, D]
            seq_lens: [B] valid lengths
            freqs: [Max_L, D/2]
        """
        b, l, c = x.shape
        n, d = self.num_heads, self.head_dim

        # QKV Projections (Autocast handles this, usually FP16/BF16)
        q = self.norm_q(self.q(x)).view(b, l, n, d)
        k = self.norm_k(self.k(x)).view(b, l, n, d)
        v = self.v(x).view(b, l, n, d)

        # Apply RoPE (1D) - Internally casts to FP32 then back
        q = rope_apply_1d(q, freqs)
        k = rope_apply_1d(k, freqs)

        # Attention (Flash Attention usually supports FP16/BF16 inputs)
        x = attention(
            q=q, k=k, v=v,
            q_lens=seq_lens, k_lens=seq_lens,
            window_size=self.window_size,
            causal=self.causal,
        )

        # Output projection
        x = x.reshape(b, l, c)
        x = x.to(self.o.weight.dtype)
        x = self.o(x)
        return x


class OmniMoEditCrossAttention(OmniMoEditSelfAttention):
    def forward(self, x, context, context_lens, seq_lens):
        out_sizes = x.size()
        b, n, d = context.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # Context is likely FP16/BF16 coming from T5/CLIP
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v, q_lens=seq_lens, k_lens=context_lens)

        # output
        x = x.flatten(2).view(*out_sizes)
        x = x.to(self.o.weight.dtype)
        x = self.o(x)
        return x


class OmniMoEditAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            causal=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.norm1 = OmniMoEditLayerNorm(dim, eps)
        self.self_attn = OmniMoEditSelfAttention(
            dim, num_heads, window_size, qk_norm, eps, causal
        )
        self.norm3 = (
            OmniMoEditLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        self.cross_attn = OmniMoEditCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = OmniMoEditLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
            self,
            x,
            e,
            seq_lens,
            freqs,
            context,
            context_lens,
    ):
        # [MixPrecision Fix] Modulation Parameter Generation
        # Ensure modulation parameters are calculated in FP32 to avoid degradation
        with torch.amp.autocast("cuda", enabled=False):
            # Upcast inputs to FP32 for the modulation calculation
            e_float = e.float()
            mod_float = self.modulation.float()
            # (1, 6, D) + (B, L, 6, D)
            e_chunks = (mod_float.unsqueeze(0) + e_float).chunk(6, dim=2)
            # Helper to retrieve chunk and keep it in FP32
            def get_chunk(idx):
                return e_chunks[idx].squeeze(2)

        # --- Self-Attention Block ---
        # 1. Modulation (Scale/Shift)
        # Perform (Norm(x) * (1+scale) + shift) in FP32, then cast to x.dtype for Attention
        x_norm = self.norm1(x)
        with torch.amp.autocast("cuda", enabled=False):
            x_norm_mod = x_norm.float() * (1 + get_chunk(1)) + get_chunk(0)

        # 2. Attention (FP16/BF16)
        y = self.self_attn(x_norm_mod.type_as(x), seq_lens, freqs)

        # 3. Residual & Gate
        # Perform (x + y * gate) in FP32 for accumulation precision, then cast back
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float() + y.float() * get_chunk(2)
            # Cast back to low precision to save VRAM for next block inputs
            x = x.type_as(y)

        # --- Cross-Attention & FFN Block ---
        # 1. Cross Attn
        # (Norm is handled internally by blocks or explicit Norm3)
        # Note: If cross_attn_norm is False, norm3 is Identity.
        # If True, OmniMoEditLayerNorm forces FP32 calc.
        y_cross = self.cross_attn(self.norm3(x), context, context_lens, seq_lens=seq_lens)
        # Residual Cross (Simple add, or gated? Original code had simple add for cross)
        x = x + y_cross

        # 2. FFN Modulation
        x_norm = self.norm2(x)
        with torch.amp.autocast("cuda", enabled=False):
            x_norm_mod = x_norm.float() * (1 + get_chunk(4)) + get_chunk(3)
        # 3. FFN (FP16/BF16)
        y_ffn = self.ffn(x_norm_mod.type_as(x))
        # 4. Residual & Gate
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float() + y_ffn.float() * get_chunk(5)
            x = x.type_as(y_ffn)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.norm = OmniMoEditLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        # Final head usually benefits from FP32 execution for stability
        # especially for velocity/noise prediction
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            # Modulation
            x_norm = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            # Final Linear
            x = self.head(x_norm)
        return x


class OmniMoEditDiTBackbone(ModelMixin, ConfigMixin):
    r"""
    Omni-MoEdit diffusion backbone for 1D Sequence Data (B, T, D).
    """
    _no_split_modules = ["OmniMoEditAttentionBlock"]

    @register_to_config
    def __init__(
            self,
            model_type="t2v",
            patch_size=(1, 1, 1),
            text_len=512,
            in_dim=128,
            dim=1024,
            ffn_dim=4096,
            freq_dim=256,
            text_dim=4096,
            out_dim=128,
            num_heads=16,
            num_layers=16,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            causal=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.freq_dim = freq_dim
        self.causal = causal
        self.text_dim = text_dim

        self.patch_embedding = nn.Linear(in_dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            OmniMoEditAttentionBlock(
                dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, causal
            )
            for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, eps)

        assert (dim % num_heads) == 0
        head_dim = dim // num_heads
        self.register_buffer("freqs", rope_params_1d(4096, head_dim), persistent=False)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.unsqueeze(0))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    def forward(
            self,
            x,  # [B, T, D]
            t,  # [B]
            context,  # List[Tensor] or [B, L_text, D_text]
            seq_lens=None,  # [B]
    ):
        device = x.device
        B, T, _ = x.shape

        # 1. Input Projection
        # Autocast: Linear will run in FP16/BF16
        x = self.patch_embedding(x)

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=device, dtype=torch.long)

        # 2. Time Embeddings
        # t is usually float, but ensure it matches device
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t).to(device)
        t_emb = self.time_embedding(t_emb)  # [B, Dim]

        # Project time to modulation params
        t_mod = self.time_projection(t_emb)
        t_mod = t_mod.view(B, 6, self.dim).unsqueeze(1).expand(-1, T, -1, -1)

        # 3. Context Embeddings
        if isinstance(context, list):
            context_lens = torch.tensor([c.shape[0] for c in context], device=device)
            max_len = context_lens.max().item()
            # Initialize with correct dtype
            padded_context = torch.zeros(B, max_len, self.text_dim, device=device, dtype=context[0].dtype)
            for i, c in enumerate(context):
                l = c.shape[0]
                padded_context[i, :l] = c
            context = padded_context
        else:
            context_lens = torch.full((B,), context.shape[1], device=device)

        context = self.text_embedding(context)

        # 4. RoPE Frequencies
        if T > self.freqs.shape[0]:
            head_dim = self.dim // self.num_heads
            self.freqs = rope_params_1d(T + 1024, head_dim).to(device)
        current_freqs = self.freqs.to(device)

        # 5. Block Loop
        for block in self.blocks:
            x = block(
                x,
                t_mod,
                seq_lens,
                current_freqs,
                context,
                context_lens,
            )

        # 6. Final Head
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, T, -1)
        x = self.head(x, t_emb_expanded)

        return x
