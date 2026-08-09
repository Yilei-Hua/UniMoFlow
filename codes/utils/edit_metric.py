# utils/edit_metrics.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


def calculate_target_alignment(generated_emb, target_text_emb, metric='cosine'):
    """
    计算生成样本与目标文本的对齐度

    Args:
        generated_emb: [B, D] 生成动作的embedding
        target_text_emb: [B, D] 目标文本的embedding
        metric: 'cosine' 或 'euclidean'
    """
    if metric == 'cosine':
        similarity = F.cosine_similarity(generated_emb, target_text_emb, dim=1)
        return similarity.mean().item(), similarity.std().item()
    else:
        distance = torch.norm(generated_emb - target_text_emb, dim=1)
        # 转换为相似度（越小越相似）
        similarity = torch.exp(-distance)
        return similarity.mean().item(), similarity.std().item()


def calculate_source_preservation(generated_emb, source_emb, metric='cosine'):
    """
    计算生成样本与源样本的结构保留度

    Args:
        generated_emb: [B, D] 生成动作的embedding
        source_emb: [B, D] 源动作的embedding
    """
    if metric == 'cosine':
        similarity = F.cosine_similarity(generated_emb, source_emb, dim=1)
        return similarity.mean().item(), similarity.std().item()
    else:
        distance = torch.norm(generated_emb - source_emb, dim=1)
        similarity = torch.exp(-distance)
        return similarity.mean().item(), similarity.std().item()


def calculate_cycle_consistency(cycle_emb, original_source_emb, metric='cosine'):
    """
    计算循环一致性：正向编辑后逆向编辑，与原始源的相似度

    Args:
        cycle_emb: [B, D] 循环后的动作embedding
        original_source_emb: [B, D] 原始源动作embedding
    """
    if metric == 'cosine':
        similarity = F.cosine_similarity(cycle_emb, original_source_emb, dim=1)
        return similarity.mean().item(), similarity.std().item()
    else:
        distance = torch.norm(cycle_emb - original_source_emb, dim=1)
        similarity = torch.exp(-distance)
        return similarity.mean().item(), similarity.std().item()


def calculate_edit_fidelity(generated_emb, source_emb, target_emb, alpha=0.5):
    """
    计算编辑保真度：在保留源结构的同时对齐目标

    fidelity = alpha * align_target + (1-alpha) * preserve_source

    Args:
        generated_emb: [B, D] 生成动作的embedding
        source_emb: [B, D] 源动作embedding
        target_emb: [B, D] 目标文本embedding
        alpha: 目标对齐的权重
    """
    align_target, _ = calculate_target_alignment(generated_emb, target_emb)
    preserve_source, _ = calculate_source_preservation(generated_emb, source_emb)

    fidelity = alpha * align_target + (1 - alpha) * preserve_source
    return fidelity


def calculate_relative_improvement(base_metric, edit_metric):
    """
    计算相对改进率（用于比较编辑模型与基础生成模型）

    improvement = (edit_metric - base_metric) / |base_metric|
    """
    return (edit_metric - base_metric) / (abs(base_metric) + 1e-8)


def extract_significant_segments(delta_norm: np.ndarray,
                                  threshold_percentile: float = 75.0,
                                  min_seg_len: int = 8) -> List[Tuple[int, int]]:
    """
    从逐帧差异中提取显著变化区域（连续段）

    Args:
        delta_norm: [T] 逐帧差异范数
        threshold_percentile: 判定显著变化的百分位阈值
        min_seg_len: 最小有效段长度

    Returns:
        List[(start, end)]: 连续变化区域的列表
    """
    threshold = np.percentile(delta_norm, threshold_percentile)
    sig_mask = delta_norm > threshold

    if sig_mask.sum() == 0:
        return []

    sig_indices = np.where(sig_mask)[0]
    segments = []
    start = sig_indices[0]
    prev = sig_indices[0]

    for idx in sig_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev + 1))
            start = idx
            prev = idx
    segments.append((start, prev + 1))

    # 过滤过短的段
    segments = [s for s in segments if s[1] - s[0] >= min_seg_len]

    return segments


def compute_all_edit_metrics(generated_emb, source_emb, target_text_emb,
                             cycle_emb=None, base_generated_emb=None):
    """
    计算所有编辑相关指标

    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {}

    # 基础指标
    metrics['target_align_mean'], metrics['target_align_std'] = \
        calculate_target_alignment(generated_emb, target_text_emb)

    metrics['source_preserve_mean'], metrics['source_preserve_std'] = \
        calculate_source_preservation(generated_emb, source_emb)

    # 编辑保真度
    metrics['edit_fidelity'] = calculate_edit_fidelity(
        generated_emb, source_emb, target_text_emb
    )

    # 循环一致性（如果提供）
    if cycle_emb is not None:
        metrics['cycle_consistency_mean'], metrics['cycle_consistency_std'] = \
            calculate_cycle_consistency(cycle_emb, source_emb)

    # 相对改进（如果提供基础模型生成结果）
    if base_generated_emb is not None:
        base_align, _ = calculate_target_alignment(base_generated_emb, target_text_emb)
        metrics['relative_improvement'] = calculate_relative_improvement(
            base_align, metrics['target_align_mean']
        )

    return metrics
