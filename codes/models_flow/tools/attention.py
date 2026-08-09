# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import math
import torch.nn.functional as F
try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    "flash_attention",
    "attention",
]
try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

__all__ =["flash_attention", "attention", "flash_attention_with_mask"]

def _get_cu_seqlens(lens):
    """
    根据长度生成累积长度 (Cumulative Sequence Lengths)
    lens: [B] -> cu_seqlens: [B+1] (0, l1, l1+l2, ...)
    """
    b = lens.shape[0]
    cu_seqlens = torch.zeros(b + 1, dtype=torch.int32, device=lens.device)
    cu_seqlens[1:] = torch.cumsum(lens, dim=0)
    return cu_seqlens


def flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
):
    """
    高效优化的 Flash Attention 接口。
    使用 Boolean Masking 替代 Python Loop 进行 Packing，消除 CPU 瓶颈。

    Args:
        q, k, v: [B, L, H, D] (Batch-First)
        q_lens, k_lens: [B] (int32) 有效长度
    """
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError("flash_attn library is not installed.")

    # 1. 准备参数
    # Flash Attn 需要 fp16/bf16
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    B, Lq, Hq, D = q.shape
    Bk, Lk, Hk, Dk = k.shape
    assert B == Bk
    assert D == Dk

    # 2. 高效 Packing (关键优化：移除 Python 循环)
    # -----------------------------------------------------------------
    # 生成 Query Mask
    if q_lens is None:
        # 如果没有 lens，假设全长，直接 flatten
        q_packed = q.reshape(-1, Hq, D)
        cu_seqlens_q = torch.arange(0, (B + 1) * Lq, step=Lq, dtype=torch.int32, device=q.device)
        max_seqlen_q = Lq
        indices_q = None  # 用于 Unpack 的索引
    else:
        # 构造布尔 Mask: [B, Lq] -> True 表示有效位置
        mask_q = torch.arange(Lq, device=q.device)[None, :] < q_lens[:, None]
        # 利用 Mask 直接选出有效 Token，无需循环切片
        q_packed = q[mask_q]  # [Total_Valid_Q, H, D]
        cu_seqlens_q = _get_cu_seqlens(q_lens)
        max_seqlen_q = q_lens.max().item()
        indices_q = mask_q  # 保存 Mask 用于后续 Unpack

    # 生成 Key/Value Mask
    if k_lens is None:
        k_packed = k.reshape(-1, Hk, D)
        v_packed = v.reshape(-1, Hk, D)
        cu_seqlens_k = torch.arange(0, (B + 1) * Lk, step=Lk, dtype=torch.int32, device=k.device)
        max_seqlen_k = Lk
    else:
        mask_k = torch.arange(Lk, device=k.device)[None, :] < k_lens[:, None]
        k_packed = k[mask_k]
        v_packed = v[mask_k]
        cu_seqlens_k = _get_cu_seqlens(k_lens)
        max_seqlen_k = k_lens.max().item()

    # 3. 调用 Flash Attention Kernel
    # -----------------------------------------------------------------
    out_packed = flash_attn_varlen_func(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )

    # 4. 高效 Unpacking (关键优化：移除 Zero Cat 循环)
    # -----------------------------------------------------------------
    if indices_q is None:
        # 没有 lens 的情况，直接 reshape 回去
        out = out_packed.view(B, Lq, Hq, D)
    else:
        # 创建全零 Tensor
        out = torch.zeros(B, Lq, Hq, D, dtype=dtype, device=q.device)
        # 利用 Mask 索引，将计算结果一次性填回对应位置
        # 这里的 indices_q 就是之前的 mask_q (bool tensor)
        out[indices_q] = out_packed
    return out

# def flash_attention(
#     q,
#     k,
#     v,
#     q_lens=None,
#     k_lens=None,
#     dropout_p=0.0,
#     softmax_scale=None,
#     q_scale=None,
#     causal=False,
#     window_size=(-1, -1),
#     deterministic=False,
#     dtype=torch.bfloat16,
#     version=None,
# ):
#     """
#     q:              [B, Lq, Nq, C1].
#     k:              [B, Lk, Nk, C1].
#     v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
#     q_lens:         [B].
#     k_lens:         [B].
#     dropout_p:      float. Dropout probability.
#     softmax_scale:  float. The scaling of QK^T before applying softmax.
#     causal:         bool. Whether to apply causal attention mask.
#     window_size:    (left right). If not (-1, -1), apply sliding window local attention.
#     deterministic:  bool. If True, slightly slower and uses more memory.
#     dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
#     """
#     half_dtypes = (torch.float16, torch.bfloat16)
#     assert dtype in half_dtypes
#     assert q.device.type == "cuda" and q.size(-1) <= 256
#
#     # params
#     b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
#
#     def half(x):
#         return x if x.dtype in half_dtypes else x.to(dtype)
#
#     # preprocess query
#     if q_lens is None:
#         q = half(q.flatten(0, 1))
#         q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
#             device=q.device, non_blocking=True
#         )
#     else:
#         q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
#
#     # preprocess key, value
#     if k_lens is None:
#         k = half(k.flatten(0, 1))
#         v = half(v.flatten(0, 1))
#         k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
#             device=k.device, non_blocking=True
#         )
#     else:
#         k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
#         v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))
#
#     q = q.to(v.dtype)
#     k = k.to(v.dtype)
#
#     if q_scale is not None:
#         q = q * q_scale
#
#     if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
#         warnings.warn(
#             "Flash attention 3 is not available, use flash attention 2 instead."
#         )
#
#     if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
#         x_packed = flash_attn_interface.flash_attn_varlen_func(
#             q=q, k=k, v=v,
#             cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device,
#                                                                                                     non_blocking=True),
#             cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device,
#                                                                                                     non_blocking=True),
#             seqused_q=None, seqused_k=None,
#             max_seqlen_q=lq, max_seqlen_k=lk,
#             softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
#         )[0]
#     else:
#         assert FLASH_ATTN_2_AVAILABLE
#         x_packed = flash_attn.flash_attn_varlen_func(
#             q=q, k=k, v=v,
#             cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device,
#                                                                                                     non_blocking=True),
#             cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device,
#                                                                                                     non_blocking=True),
#             max_seqlen_q=lq, max_seqlen_k=lk,
#             dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
#             window_size=window_size, deterministic=deterministic,
#         )
#
#     # 2. 手动 Unpack (填充回 Padding 形状)
#     # 如果没有提供 q_lens，说明原本就没 Pack，可以直接 reshape
#     if q_lens is None:
#         x = x_packed.view(b, lq, *x_packed.shape[1:])
#     else:
#         # 创建全零 Tensor: [Batch, Max_Len, Heads, Head_Dim]
#         x = torch.zeros((b, lq, *x_packed.shape[1:]), dtype=x_packed.dtype, device=x_packed.device)
#
#         curr_offset = 0
#         for i, length in enumerate(q_lens):
#             # 将有效数据填入
#             if length > 0:
#                 x[i, :length] = x_packed[curr_offset: curr_offset + length]
#             curr_offset += length
#
#     # output
#     return x.type(out_dtype)

def attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        fa_version=None,
        attn_mask=None,
):
    # 1. 维度转换 [B, L, H, D] -> [B, H, L, D]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    b, h, lq, d = q.shape
    lk = k.shape[2]
    device = q.device
    input_dtype = q.dtype
    # --- 关键修复：确保 lens 在正确的设备上 ---
    if q_lens is not None:
        q_lens = q_lens.to(device)
    if k_lens is not None:
        k_lens = k_lens.to(device)

    # 2. 处理 Scaling
    # FlashAttn: Softmax(Q * q_scale * K^T * softmax_scale)
    # SDPA:      Softmax(Q * K^T / sqrt(d))
    if q_scale is not None:
        q = q * q_scale

    if softmax_scale is not None:
        # 抵消 SDPA 内部的 1/sqrt(d)，并应用自定义 scale
        scale_factor = softmax_scale * math.sqrt(d)
        q = q * scale_factor

    # 3. 构建 Attention Mask
    # 初始化 mask。如果有任何约束条件，我们需要创建一个 dense mask
    mask = None

    need_mask = (
            q_lens is not None or
            k_lens is not None or
            causal or
            window_size != (-1, -1) or
            attn_mask is not None
    )

    if need_mask:
        # 使用 float mask (默认为 0), 形状 (B, 1, Lq, Lk) 用于广播到 H
        mask = torch.zeros((b, 1, lq, lk), device=device, dtype=q.dtype)

        # 3.1 处理 Padding Mask (Key 的有效长度)
        if k_lens is not None:
            # seq_ids: (1, 1, 1, Lk)
            seq_ids = torch.arange(lk, device=device)[None, None, None, :]
            # k_lens: (B, 1, 1, 1)
            k_lens_expanded = k_lens[:, None, None, None]
            # Mask 掉 id >= len 的部分
            mask = mask.masked_fill(seq_ids >= k_lens_expanded, float("-inf"))

        # 3.2 处理 Causal Mask
        if causal:
            q_ids = torch.arange(lq, device=device)[:, None]
            k_ids = torch.arange(lk, device=device)[None, :]
            mask = mask.masked_fill(q_ids < k_ids, float("-inf"))

        # 3.3 处理 Window Attention
        w_left, w_right = window_size
        if w_left != -1 and w_right != -1:
            q_ids = torch.arange(lq, device=device)[:, None]
            k_ids = torch.arange(lk, device=device)[None, :]
            dist = q_ids - k_ids
            mask = mask.masked_fill(dist < -w_right, float("-inf"))
            mask = mask.masked_fill(dist > w_left, float("-inf"))

        # 3.4 合并外部传入的 attn_mask
        if attn_mask is not None:
            # 确保 attn_mask 在正确设备
            attn_mask = attn_mask.to(device)

            # BUG FIX: 确保 attn_mask 与 q 的 dtype 一致
            attn_mask = attn_mask.to(q.dtype)

            # 检查 attn_mask 形状并进行广播处理
            # 假设 attn_mask 可能是 (B, 1, Lq, Lk) 或者 (B, Lq, Lk)
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)

            if attn_mask.dtype == torch.bool:
                mask = mask.masked_fill(attn_mask, float("-inf"))
            else:
                # 假设是加性 mask (例如 ALiBi 或 预先计算好的 float mask)
                mask = mask + attn_mask

    # 4. 执行 Attention
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=False  # 我们已手动处理 causal
    )

    # 5. 转换回 [B, L, H, D]
    out = out.transpose(1, 2).contiguous()

    # 6. 处理 Output Padding (Query 的有效长度)
    # 将超过 q_lens 的输出置零，保持与 FlashAttn varlen 行为一致
    if q_lens is not None:
        seq_ids = torch.arange(lq, device=device)[None, :, None, None]
        q_lens_expanded = q_lens[:, None, None, None]
        out = out.masked_fill(seq_ids >= q_lens_expanded, 0.0)

    if out.dtype != input_dtype:
        out = out.to(input_dtype)

    return out


# =============================================================================
# 【新增】专为 Edit Model 设计的带 Attention Mask 优化 Flash Attention
# =============================================================================
# def flash_attention_with_mask(
#         q,
#         k,
#         v,
#         q_lens=None,
#         k_lens=None,
#         attn_mask=None,
#         dropout_p=0.0,
#         softmax_scale=None,
#         causal=False,
#         window_size=(-1, -1),
#         deterministic=False,
#         dtype=torch.bfloat16,
#         num_registers=4,
# ):
#     """
#     专为 Edit Model 设计的带 Attention Mask 的 Flash Attention
#     支持 [Register + Source + Target] 结构的特殊 attention 掩码
#     """
#     B, L, H, D = q.shape
#     device = q.device
#     R = num_registers
#
#     # 判断是否是 Edit Block 的特殊结构 (Register+Source+Target)
#     is_edit_block = False
#     if L > R * 2:
#         if isinstance(q_lens, tuple) or isinstance(k_lens, tuple):
#             is_edit_block = True
#         elif attn_mask is not None and attn_mask.shape[-1] == L:
#             is_edit_block = True
#
#     # Boolean mask 提取器
#     def get_bool_mask(lens, total_len):
#         """将长度转换为 boolean mask [B, total_len]"""
#         if lens is None:
#             return torch.ones((B, total_len), dtype=torch.bool, device=device)
#
#         if isinstance(lens, tuple):
#             # (src_lens, tgt_lens) 格式，用于 Edit Block
#             src_lens, tgt_lens = lens
#             mask = torch.zeros((B, total_len), dtype=torch.bool, device=device)
#             mask[:, :R] = True  # Register
#
#             S = (total_len - R) // 2
#             src_ids = torch.arange(S, device=device)[None, :]
#             mask[:, R:R + S] = src_ids < src_lens[:, None]
#
#             tgt_ids = torch.arange(S, device=device)[None, :]
#             start_tgt = R + S
#             mask[:, start_tgt:start_tgt + S] = tgt_ids < tgt_lens[:, None]
#             return mask
#         else:
#             # 标准长度格式 [B]
#             if lens.dtype == torch.long or lens.dtype == torch.int:
#                 mask = torch.arange(total_len, device=device)[None, :] < lens[:, None]
#                 return mask
#             else:
#                 # 已经是 boolean mask
#                 return lens
#
#     # 辅助函数：将 boolean mask 转换为长度 [B]
#     def mask_to_lens(mask):
#         if mask is None:
#             return None
#         if mask.dtype == torch.bool:
#             return mask.sum(dim=1)  # [B, L] -> [B]
#         return mask  # 已经是长度格式
#
#     # 1. 退化为纯 SDPA 处理 (当无 Flash Attn 时)
#     if not FLASH_ATTN_AVAILABLE:
#         full_mask = None
#         if attn_mask is not None:
#             full_mask = attn_mask.clone()
#             if full_mask.ndim == 3:
#                 full_mask = full_mask.unsqueeze(1)
#             elif full_mask.ndim == 2:
#                 full_mask = full_mask.unsqueeze(0).unsqueeze(0)
#
#         if is_edit_block:
#             q_mask = get_bool_mask(q_lens, L)
#             k_mask = get_bool_mask(k_lens, L)
#             return attention(
#                 q, k, v, q_lens=q_mask, k_lens=k_mask, dropout_p=dropout_p,
#                 softmax_scale=softmax_scale, causal=causal, window_size=window_size,
#                 deterministic=deterministic, dtype=dtype, attn_mask=full_mask
#             )
#         else:
#             return attention(
#                 q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
#                 softmax_scale=softmax_scale, causal=causal, window_size=window_size,
#                 deterministic=deterministic, dtype=dtype, attn_mask=full_mask
#             )
#
#     # 2. 标准 Flash Attention (非 Edit Block)
#     if not is_edit_block:
#         if isinstance(q_lens, tuple):
#             raise ValueError("q_lens cannot be a tuple for standard flash_attention")
#         return flash_attention(
#             q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
#             dropout_p=dropout_p, softmax_scale=softmax_scale,
#             causal=causal, window_size=window_size,
#             deterministic=deterministic, dtype=dtype,
#         )
#
#     # 3. Edit Block 特化处理: [Register(R) | Source(S) | Target(T)]
#     q = q.to(dtype)
#     k = k.to(dtype)
#     v = v.to(dtype)
#
#     S = (L - R) // 2  # Source 长度
#     T = L - R - S  # Target 长度
#
#     # 生成 boolean masks [B, L]
#     q_mask = get_bool_mask(q_lens, L)
#     k_mask = get_bool_mask(k_lens, L)
#
#     # 切分 Q [B, L, H, D] -> [B, R/S/T, H, D]
#     q_reg = q[:, :R]
#     q_src = q[:, R:R + S]
#     q_tgt = q[:, R + S:R + S + T]
#
#     # 判断 Source 是否需要见 Target
#     src_sees_tgt = True
#     if attn_mask is not None:
#         src_first = R
#         tgt_first = R + S
#         if attn_mask.ndim == 4:
#             src_sees_tgt = attn_mask[0, 0, src_first, tgt_first] > -1e9
#         elif attn_mask.ndim == 3:
#             src_sees_tgt = attn_mask[0, src_first, tgt_first] > -1e9
#         else:
#             src_sees_tgt = attn_mask[src_first, tgt_first] > -1e9
#
#     # 计算各部分的实际长度 [B]
#     q_lens_reg = mask_to_lens(q_mask[:, :R]) if q_mask is not None else None
#     q_lens_src = mask_to_lens(q_mask[:, R:R + S]) if q_mask is not None else None
#     q_lens_tgt = mask_to_lens(q_mask[:, R + S:]) if q_mask is not None else None
#     k_lens_full = mask_to_lens(k_mask) if k_mask is not None else None
#
#     # Register: attend to all (global aggregation)
#     out_reg = flash_attention(
#         q=q_reg, k=k, v=v,
#         q_lens=q_lens_reg,
#         k_lens=k_lens_full,
#         dropout_p=dropout_p, softmax_scale=softmax_scale, causal=False, dtype=dtype,
#     )
#
#     # Source: may be protected (not see Target)
#     if not src_sees_tgt:
#         # Source only sees Register and Source
#         k_for_src = k[:, :R + S]
#         v_for_src = v[:, :R + S]
#         k_lens_for_src = mask_to_lens(k_mask[:, :R + S]) if k_mask is not None else None
#     else:
#         k_for_src = k
#         v_for_src = v
#         k_lens_for_src = k_lens_full
#
#     out_src = flash_attention(
#         q=q_src, k=k_for_src, v=v_for_src,
#         q_lens=q_lens_src,
#         k_lens=k_lens_for_src,
#         dropout_p=dropout_p, softmax_scale=softmax_scale, causal=False, dtype=dtype,
#     )
#
#     # Target: attend to all (Register + Source + Target)
#     out_tgt = flash_attention(
#         q=q_tgt, k=k, v=v,
#         q_lens=q_lens_tgt,
#         k_lens=k_lens_full,
#         dropout_p=dropout_p, softmax_scale=softmax_scale, causal=False, dtype=dtype,
#     )
#
#     # Concatenate back
#     out = torch.cat([out_reg, out_src, out_tgt], dim=1)
#     return out
