import sys as _release_sys
from pathlib import Path as _ReleasePath

_CODES_ROOT = _ReleasePath(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_CODES_ROOT))

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

# 项目模块导入
from omni_moedit.models.text_to_motion_dit import OmniMoEditDiT
from models_flow.hrvae import HRVAE
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from config.load_config import load_config


# -----------------------------------------------------------------------------
# 1. Dataset 类 - 保持不变
# -----------------------------------------------------------------------------
class FlowEditDataset(Dataset):
    """
    FlowEdit 数据集类 - 全变体版
    每个原始样本生成所有变化方向，并记录所属的split和完整的编辑命令信息
    """

    def __init__(self, json_path: str, cfg: Dict[str, Any], mean: np.ndarray, std: np.ndarray):
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.data_root = cfg.data.root_dir

        # Latent 目录（已切分，直接加载）
        self.latent_dirs = {
            'train': pjoin(self.data_root, cfg.data.latent_dir, 'train'),
            'val': pjoin(self.data_root, cfg.data.latent_dir, 'val'),
            'test': pjoin(self.data_root, cfg.data.latent_dir, 'test'),
        }

        # 解析 JSON 并展开所有变体
        self.items = self._parse_json_with_all_variations(json_path)
        print(f"[Dataset] Loaded {len(self.items)} total variations from {json_path}")

    def _determine_split(self, cid: str) -> str:
        """根据样本key判断属于哪个split"""
        for split_name in ['train', 'val', 'test']:
            path = pjoin(self.latent_dirs[split_name], f"{cid}.npy")
            if os.path.exists(path):
                return split_name
        return 'train'

    def _get_source_path(self, cid: str, split: str) -> str:
        """获取源数据的原始路径"""
        return pjoin(self.latent_dirs[split], f"{cid}.npy")

    def _parse_json_with_all_variations(self, json_path: str) -> List[Dict]:
        """解析 JSON 并为每个源样本展开所有变化变体"""
        items = []

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for entry in raw_data:
            original_key = entry['original_key']
            src_caption = entry['original_caption']
            variations = []
            is_new_format = False

            if 'edits' in entry:
                variations = entry['edits']
                is_new_format = True
            elif 'variations' in entry:
                variations = entry['variations']
                is_new_format = False
            else:
                continue

            if not variations:
                continue

            split = self._determine_split(original_key)
            source_path = self._get_source_path(original_key, split)

            for var_idx, var in enumerate(variations):
                if is_new_format:
                    target_caption = var.get('target_caption', '')
                    reverse_command = var.get(
                        'reverse_command', var.get('reverse_edit_command', '')
                    )
                else:
                    target_caption = var.get('new_caption', '')
                    reverse_command = var.get(
                        'reverse_edit_command', var.get('reverse_command', '')
                    )

                items.append({
                    'original_key': original_key,
                    'split': split,
                    'source_path': source_path,
                    'source_caption': src_caption,
                    'target_caption': target_caption,
                    'edit_command': var.get('edit_command', ''),
                    'reverse_edit_command': reverse_command,
                    'variation_idx': var_idx,
                    'total_variations': len(variations),
                })

        return items

    def __len__(self):
        return len(self.items)

    def _load_latent(self, cid: str) -> Tuple[np.ndarray, bool, int]:
        """加载已切分的 latent，返回 (data, success, original_length)"""
        for split_name, dir_path in self.latent_dirs.items():
            path = pjoin(dir_path, f"{cid}.npy")
            if os.path.exists(path):
                latent = np.load(path)
                original_length = len(latent)
                return latent, True, original_length
        return None, False, 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        cid = item['original_key']

        latent_data, is_latent, original_length = self._load_latent(cid)

        if not is_latent:
            latent_dim = self.cfg.diffusion.input_dim
            latent_data = np.zeros((10, latent_dim))
            is_latent = True
            original_length = 10

        return {
            'latent': torch.from_numpy(latent_data).float(),
            'is_latent': is_latent,
            'original_length': original_length,
            'original_key': cid,
            'split': item['split'],
            'source_path': item['source_path'],
            'source_caption': item['source_caption'],
            'target_caption': item['target_caption'],
            'edit_command': item['edit_command'],
            'reverse_edit_command': item['reverse_edit_command'],
            'variation_idx': item['variation_idx'],
            'total_variations': item['total_variations'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """自定义 Collate 函数"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    lengths = [len(b['latent']) for b in batch]
    max_len = max(lengths)
    feat_dim = batch[0]['latent'].shape[-1]
    bsz = len(batch)

    padded_latents = torch.zeros(bsz, max_len, feat_dim)
    for i, b in enumerate(batch):
        l = len(b['latent'])
        padded_latents[i, :l] = b['latent']

    return {
        'latent': padded_latents,
        'latent_length': torch.tensor(lengths),
        'original_length': torch.tensor([b['original_length'] for b in batch]),
        'original_key': [b['original_key'] for b in batch],
        'split': [b['split'] for b in batch],
        'source_path': [b['source_path'] for b in batch],
        'source_caption': [b['source_caption'] for b in batch],
        'target_caption': [b['target_caption'] for b in batch],
        'edit_command': [b['edit_command'] for b in batch],
        'reverse_edit_command': [b['reverse_edit_command'] for b in batch],
        'variation_idx': [b['variation_idx'] for b in batch],
        'total_variations': [b['total_variations'] for b in batch],
    }


# -----------------------------------------------------------------------------
# 2. 核心评估逻辑：Matching Score主导 + R-precision上限处理
# -----------------------------------------------------------------------------
def compute_comprehensive_metrics(
        eval_wrapper: EvaluatorWrapper,
        source_latents: torch.Tensor,
        edited_latents: torch.Tensor,
        base_latents: torch.Tensor,
        latent_lengths: torch.Tensor,
        target_captions: List[str],
        source_captions: List[str],
        vae: HRVAE,
        cfg: Dict[str, Any],
        device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict]:
    """
    综合评估指标计算 - Matching Score主导筛选
    核心逻辑：
    1. 硬阈值：R-precision >= 0.7, Matching Score >= 0.6
    2. Matching Score必须比源数据有显著提升（主要判断依据）
    3. R-precision处理：
       - 如果源R-precision < 1：要求编辑后R-precision > 源R-precision
       - 如果源R-precision == 1：只要编辑后R-precision == 1即可（已达上限），主要看Matching提升
    """

    decoded_lengths = latent_lengths * cfg.data.downsample_ratio
    B, T_max, D = source_latents.shape

    # --- 0. 解码所有 latent ---
    with torch.no_grad():
        source_decoded = vae.decode(source_latents)
        edited_decoded = vae.decode(edited_latents)
        base_decoded = vae.decode(base_latents)

    # 提取 motion 特征
    feat_dim = 148  # 根据您的数据调整
    source_motion = source_decoded[..., :feat_dim]
    edited_motion = edited_decoded[..., :feat_dim]
    base_motion = base_decoded[..., :feat_dim]

    # --- 1. 计算 Text Embedding ---
    text_emb, _ = eval_wrapper.encode_text(target_captions)
    text_emb_norm = F.normalize(text_emb, p=2, dim=-1)

    # --- 2. 计算 Motion Embedding ---
    _, source_motion_emb, _ = eval_wrapper.encode_motion(source_motion, decoded_lengths)
    _, edited_motion_emb, _ = eval_wrapper.encode_motion(edited_motion, decoded_lengths)
    _, base_motion_emb, _ = eval_wrapper.encode_motion(base_motion, decoded_lengths)

    source_motion_emb_norm = F.normalize(source_motion_emb, p=2, dim=-1)
    edited_motion_emb_norm = F.normalize(edited_motion_emb, p=2, dim=-1)
    base_motion_emb_norm = F.normalize(base_motion_emb, p=2, dim=-1)

    # --- 3. 计算 Matching Score (余弦相似度) ---
    matching_score_source = (source_motion_emb_norm * text_emb_norm).sum(dim=-1)
    matching_score_edit = (edited_motion_emb_norm * text_emb_norm).sum(dim=-1)
    matching_score_base = (base_motion_emb_norm * text_emb_norm).sum(dim=-1)

    # --- 4. 计算 R Precision (Top-K 检索准确率) ---
    def compute_r_precision(sim_matrix, k_list=[1, 2, 3]):
        """计算 R Precision：看正确配对是否在Top-K中"""
        B = sim_matrix.shape[0]
        top_k_indices = torch.argsort(sim_matrix, dim=1, descending=True)
        r_precisions = {}
        for k in k_list:
            correct_positions = torch.arange(B, device=device).unsqueeze(1)
            is_in_topk = (top_k_indices[:, :k] == correct_positions).any(dim=1)
            r_precisions[f'r_precision_top{k}'] = is_in_topk.float()
        return r_precisions

    sim_matrix_source = torch.mm(source_motion_emb_norm, text_emb_norm.t())
    sim_matrix_edit = torch.mm(edited_motion_emb_norm, text_emb_norm.t())
    sim_matrix_base = torch.mm(base_motion_emb_norm, text_emb_norm.t())

    r_prec_source = compute_r_precision(sim_matrix_source)
    r_prec_edit = compute_r_precision(sim_matrix_edit)
    r_prec_base = compute_r_precision(sim_matrix_base)

    r_precision_source = r_prec_source['r_precision_top3']
    r_precision_edit = r_prec_edit['r_precision_top3']
    r_precision_base = r_prec_base['r_precision_top3']
    r_precision_top1_source = r_prec_source['r_precision_top1']
    r_precision_top1_edit = r_prec_edit['r_precision_top1']

    # --- 5. 源内容保留度 ---
    r_preserve_edit_list = []
    r_preserve_base_list = []
    for i in range(B):
        actual_len = int(latent_lengths[i].item())
        src = source_latents[i, :actual_len].flatten()
        edit = edited_latents[i, :actual_len].flatten()
        base = base_latents[i, :actual_len].flatten()
        preserve_edit = F.cosine_similarity(src.unsqueeze(0), edit.unsqueeze(0), dim=-1)
        preserve_base = F.cosine_similarity(src.unsqueeze(0), base.unsqueeze(0), dim=-1)
        r_preserve_edit_list.append(preserve_edit)
        r_preserve_base_list.append(preserve_base)
    r_preserve_edit = torch.cat(r_preserve_edit_list)
    r_preserve_base = torch.cat(r_preserve_base_list)

    # --- 6. 改进幅度计算 ---
    matching_improvement = matching_score_edit - matching_score_source
    r_precision_improvement = r_precision_edit - r_precision_source
    r_precision_top1_improvement = r_precision_top1_edit - r_precision_top1_source

    # --- 7. 硬阈值筛选条件 ---
    min_r_precision_threshold = cfg.filtering.get('min_r_precision_threshold', 0.7)
    min_matching_threshold = cfg.filtering.get('min_matching_threshold', 0.6)

    # 硬阈值：编辑结果必须满足最低质量要求
    cond_hard_r_precision = r_precision_edit >= min_r_precision_threshold
    cond_hard_matching = matching_score_edit >= min_matching_threshold

    # --- 8. 🌟 核心改进：Matching Score主导 + R-precision上限感知 ---
    min_matching_margin = cfg.filtering.get('min_matching_margin', 0.02)

    # 主要条件：Matching Score必须比源数据高一定阈值（核心判断依据）
    cond_matching_improved = matching_score_edit >= (matching_score_source + min_matching_margin)

    # R-precision改进条件（上限感知处理）：
    # 情况1: 源R-precision < 1 → 要求编辑R-precision > 源R-precision
    # 情况2: 源R-precision == 1 → 要求编辑R-precision == 1（已达上限，无法提升）
    source_r_is_perfect = (r_precision_source >= 0.999)  # 浮点数比较，视为1
    edit_r_is_perfect = (r_precision_edit >= 0.999)

    # R-precision改进判断：
    # - 如果源不是1：要求编辑R > 源R
    # - 如果源是1：要求编辑R也是1（保持上限）
    cond_r_precision_improved = torch.zeros(B, dtype=torch.bool, device=device)

    # 源R < 1的样本：要求R-precision严格提升
    mask_source_not_perfect = ~source_r_is_perfect
    cond_r_precision_improved[mask_source_not_perfect] = (
            r_precision_edit[mask_source_not_perfect] > r_precision_source[mask_source_not_perfect]
    )

    # 源R == 1的样本：要求编辑R也是1（保持上限即可）
    mask_source_perfect = source_r_is_perfect
    cond_r_precision_improved[mask_source_perfect] = edit_r_is_perfect[mask_source_perfect]

    # 如果源R不是1，但编辑后R == 1（提升到上限），也接受
    cond_r_reached_perfect = (~source_r_is_perfect) & edit_r_is_perfect
    cond_r_precision_improved = cond_r_precision_improved | cond_r_reached_perfect

    # --- 9. 保留度约束 ---
    min_preserve = cfg.filtering.get('min_preserve', 0.5)
    max_preserve = cfg.filtering.get('max_preserve', 0.95)
    cond_preserve_min = r_preserve_edit >= min_preserve
    cond_preserve_max = r_preserve_edit <= max_preserve
    cond_preserve = cond_preserve_min & cond_preserve_max

    # --- 10. 综合筛选条件 🌟 ---
    # 核心逻辑：硬阈值 + Matching Score主导提升 + R-precision上限感知
    quality_mask = (
            cond_hard_r_precision &
            cond_hard_matching &
            cond_matching_improved &  # Matching Score必须提升（主要）
            cond_r_precision_improved &  # R-precision改进或保持上限
            cond_preserve
    )

    # 可选：Top-1 R Precision 也要求提升（使用相同的逻辑）
    if cfg.filtering.get('require_top1_improvement', False):
        source_top1_is_perfect = (r_precision_top1_source >= 0.999)
        edit_top1_is_perfect = (r_precision_top1_edit >= 0.999)

        cond_top1_improved = torch.zeros(B, dtype=torch.bool, device=device)
        mask_top1_source_not_perfect = ~source_top1_is_perfect
        cond_top1_improved[mask_top1_source_not_perfect] = (
                r_precision_top1_edit[mask_top1_source_not_perfect] > r_precision_top1_source[
            mask_top1_source_not_perfect]
        )
        mask_top1_source_perfect = source_top1_is_perfect
        cond_top1_improved[mask_top1_source_perfect] = edit_top1_is_perfect[mask_top1_source_perfect]

        cond_top1_reached_perfect = (~source_top1_is_perfect) & edit_top1_is_perfect
        cond_top1_improved = cond_top1_improved | cond_top1_reached_perfect

        quality_mask = quality_mask & cond_top1_improved
    else:
        cond_top1_improved = torch.ones(B, dtype=torch.bool, device=device)

    metrics = {
        'matching_score_edit': matching_score_edit,
        'matching_score_base': matching_score_base,
        'matching_score_source': matching_score_source,
        'matching_improvement': matching_improvement,
        'r_precision_edit': r_precision_edit,
        'r_precision_source': r_precision_source,
        'r_precision_base': r_precision_base,
        'r_precision_improvement': r_precision_improvement,
        'r_precision_top1_edit': r_precision_top1_edit,
        'r_precision_top1_source': r_precision_top1_source,
        'r_precision_top1_improvement': r_precision_top1_improvement,
        'r_preserve_edit': r_preserve_edit,
        'r_preserve_base': r_preserve_base,
    }

    details = {
        # 硬阈值条件
        'cond_hard_r_precision': cond_hard_r_precision,
        'cond_hard_matching': cond_hard_matching,
        # Matching Score主导条件
        'cond_matching_improved': cond_matching_improved,
        # R-precision上限感知条件
        'cond_r_precision_improved': cond_r_precision_improved,
        'source_r_is_perfect': source_r_is_perfect,
        'edit_r_is_perfect': edit_r_is_perfect,
        'cond_r_reached_perfect': cond_r_reached_perfect,
        # 保留度条件
        'cond_preserve': cond_preserve,
        'cond_preserve_min': cond_preserve_min,
        'cond_preserve_max': cond_preserve_max,
        # Top1条件
        'cond_top1_improved': cond_top1_improved if cfg.filtering.get('require_top1_improvement',
                                                                      False) else torch.ones(B, dtype=torch.bool,
                                                                                             device=device),
    }

    return metrics, quality_mask, details


# -----------------------------------------------------------------------------
# 3. 筛选与保存模式（只保存编辑后的样本，记录源路径）
# -----------------------------------------------------------------------------
def run_filtering_mode(
        cfg: Dict[str, Any],
        dataloader: DataLoader,
        vae: HRVAE,
        diff_model: OmniMoEditDiT,
        eval_wrapper: EvaluatorWrapper,
        device: torch.device
):
    """
    筛选模式：遍历所有变体，评估筛选，只保存编辑后的样本
    """
    print(f"\n{'=' * 70}")
    print(f"[FILTERING MODE] Matching Score主导 + R-Precision上限感知")
    print(f"核心逻辑：")
    print(
        f"  1. 硬阈值: R-Precision ≥ {cfg.filtering.get('min_r_precision_threshold', 0.7)}, Matching ≥ {cfg.filtering.get('min_matching_threshold', 0.6)}")
    print(f"  2. 主要判断: Matching Score > Source + {cfg.filtering.get('min_matching_margin', 0.02)}")
    print(f"  3. R-Precision处理:")
    print(f"     - 源R < 1: 要求编辑R > 源R")
    print(f"     - 源R = 1: 要求编辑R = 1（已达上限，Matching主导）")
    print(f"FlowEdit: steps={cfg.flowedit.num_steps}, cfg_scale={cfg.flowedit.cfg_scale_tgt}")
    print(f"{'=' * 70}\n")

    # 创建输出目录结构
    output_dirs = {}
    for split in ['train', 'val', 'test']:
        output_dirs[split] = pjoin(cfg.io.output_dir, 'edited', split)
        os.makedirs(output_dirs[split], exist_ok=True)

    if cfg.filtering.get('save_all_generated', False):
        all_dirs = {}
        for split in ['train', 'val', 'test']:
            all_dirs[split] = pjoin(cfg.io.output_dir, 'all_edited', split)
            os.makedirs(all_dirs[split], exist_ok=True)

    accepted_records = []
    rejected_records = []
    all_records = []

    # 更新统计字典
    stats = {
        'total': 0,
        'accepted': 0,
        'rejected': {
            'hard_r_precision': 0,  # 不满足R-precision硬阈值
            'hard_matching': 0,  # 不满足Matching Score硬阈值
            'matching_not_improved': 0,  # Matching Score未提升
            'rprec_not_improved': 0,  # R-Precision未改进且未达上限
            'rprec_dropped': 0,  # R-Precision从1下降
            'over_edit': 0,
            'under_edit': 0,
            'length_mismatch': 0,
        }
    }

    pbar = tqdm(dataloader, desc="Processing variations")

    for batch_idx, batch in enumerate(pbar):
        if not batch:
            continue

        try:
            B = len(batch['original_key'])
            stats['total'] += B

            # 准备输入
            source_latents = batch['latent'].to(device)
            latent_lengths = batch['latent_length'].to(device)
            original_lengths = batch['original_length'].to(device)
            splits = batch['split']
            source_paths = batch['source_path']

            max_length = source_latents.shape[1]

            with torch.no_grad():
                # FlowEdit 生成
                x_in_edit = {
                    "feature": source_latents,
                    "text": batch['source_caption'],
                    "feature_length": latent_lengths,
                }
                edit_out = diff_model.flow_edit(
                    x=x_in_edit,
                    target_text=batch['target_caption'],
                    num_steps=cfg.flowedit.num_steps,
                    cfg_scale_tgt=cfg.flowedit.cfg_scale_tgt
                )
                edited_latents = edit_out['generated']

                # Base 生成 (用于对比分析，不参与筛选)
                x_in_base = {
                    "text": batch['target_caption'],
                    "feature_length": latent_lengths,
                }
                base_out = diff_model.generate(
                    x=x_in_base,
                    num_denoise_steps=cfg.flowedit.num_steps
                )
                base_latents = base_out['generated']

            # 确保长度一致
            def ensure_length(latent, target_len):
                B, T, D = latent.shape
                if T == target_len:
                    return latent
                elif T > target_len:
                    return latent[:, :target_len, :]
                else:
                    pad_len = target_len - T
                    last_frame = latent[:, -1:, :].expand(B, pad_len, D)
                    return torch.cat([latent, last_frame], dim=1)

            if base_latents.shape[1] != max_length:
                base_latents = ensure_length(base_latents, max_length)
            if edited_latents.shape[1] != max_length:
                edited_latents = ensure_length(edited_latents, max_length)

            assert source_latents.shape == edited_latents.shape
            assert source_latents.shape == base_latents.shape

            # 综合评估
            metrics, quality_mask, details = compute_comprehensive_metrics(
                eval_wrapper=eval_wrapper,
                source_latents=source_latents,
                edited_latents=edited_latents,
                base_latents=base_latents,
                latent_lengths=latent_lengths,
                target_captions=batch['target_caption'],
                source_captions=batch['source_caption'],
                vae=vae,
                cfg=cfg,
                device=device
            )

            # 处理每个样本
            for i in range(B):
                key = batch['original_key'][i]
                var_idx = batch['variation_idx'][i]
                total_var = batch['total_variations'][i]
                split = splits[i]
                src_path = source_paths[i]
                actual_len = int(latent_lengths[i].item())

                base_name = f"{key.replace('#', '_')}_var{var_idx}"

                record = {
                    'original_key': key,
                    'split': split,
                    'variation_idx': var_idx,
                    'total_variations': total_var,
                    'source_path': src_path,
                    'edited_path': None,
                    'original_length': int(original_lengths[i].item()),
                    'edited_length': actual_len,
                    'source_caption': batch['source_caption'][i],
                    'target_caption': batch['target_caption'][i],
                    'edit_command': batch['edit_command'][i],
                    'reverse_edit_command': batch['reverse_edit_command'][i],
                    'metrics': {
                        'matching_score_edit': float(metrics['matching_score_edit'][i].cpu()),
                        'matching_score_base': float(metrics['matching_score_base'][i].cpu()),
                        'matching_score_source': float(metrics['matching_score_source'][i].cpu()),
                        'matching_improvement': float(metrics['matching_improvement'][i].cpu()),
                        'r_precision_edit': float(metrics['r_precision_edit'][i].cpu()),
                        'r_precision_source': float(metrics['r_precision_source'][i].cpu()),
                        'r_precision_improvement': float(metrics['r_precision_improvement'][i].cpu()),
                        'r_precision_top1_edit': float(metrics['r_precision_top1_edit'][i].cpu()),
                        'r_precision_top1_source': float(metrics['r_precision_top1_source'][i].cpu()),
                        'r_preserve_edit': float(metrics['r_preserve_edit'][i].cpu()),
                        'r_preserve_base': float(metrics['r_preserve_base'][i].cpu()),
                    },
                    'status': 'accepted' if quality_mask[i].item() else 'rejected',
                    'reject_reason': None,
                }

                # 可选：保存所有生成的
                if cfg.filtering.get('save_all_generated', False):
                    edit_path = pjoin(all_dirs[split], f"{base_name}.npy")
                    np.save(edit_path, edited_latents[i, :actual_len].cpu().numpy())
                    record['all_edited_path'] = edit_path

                # 判断是否通过筛选
                if quality_mask[i].item():
                    stats['accepted'] += 1

                    edit_path = pjoin(output_dirs[split], f"{base_name}.npy")
                    np.save(edit_path, edited_latents[i, :actual_len].cpu().numpy())

                    record['edited_path'] = edit_path
                    accepted_records.append(record)
                else:
                    # 详细的拒绝原因分析
                    reasons = []

                    # 优先级1: 硬阈值检查
                    if not details['cond_hard_r_precision'][i].item():
                        reasons.append('hard_r_precision')
                        stats['rejected']['hard_r_precision'] += 1

                    if not details['cond_hard_matching'][i].item():
                        reasons.append('hard_matching')
                        stats['rejected']['hard_matching'] += 1

                    # 优先级2: Matching Score主导检查
                    if not details['cond_matching_improved'][i].item():
                        reasons.append('matching_not_improved')
                        stats['rejected']['matching_not_improved'] += 1

                    # 优先级3: R-precision上限感知检查
                    if not details['cond_r_precision_improved'][i].item():
                        # 细分R-precision失败类型
                        if details['source_r_is_perfect'][i].item() and not details['edit_r_is_perfect'][i].item():
                            # 源是1但编辑后不是1：R-precision下降了
                            reasons.append('rprec_dropped')
                            stats['rejected']['rprec_dropped'] += 1
                        else:
                            # 其他R-precision未改进情况
                            reasons.append('rprec_not_improved')
                            stats['rejected']['rprec_not_improved'] += 1

                    # 优先级4: 保留度检查
                    if not details['cond_preserve_min'][i].item():
                        reasons.append('over_edit')
                        stats['rejected']['over_edit'] += 1
                    elif not details['cond_preserve_max'][i].item():
                        reasons.append('under_edit')
                        stats['rejected']['under_edit'] += 1

                    # 记录主要拒绝原因（优先级排序）
                    priority_order = [
                        'hard_r_precision',
                        'hard_matching',
                        'matching_not_improved',
                        'rprec_dropped',
                        'rprec_not_improved',
                        'over_edit',
                        'under_edit'
                    ]

                    primary_reason = 'unknown'
                    for reason in priority_order:
                        if reason in reasons:
                            primary_reason = reason
                            break

                    record['reject_reason'] = primary_reason
                    record['reject_details'] = reasons

                    # 记录详细的对比指标（用于分析）
                    record['source_matching_score'] = float(metrics['matching_score_source'][i].cpu())
                    record['edit_matching_score'] = float(metrics['matching_score_edit'][i].cpu())
                    record['matching_improvement'] = float(metrics['matching_improvement'][i].cpu())
                    record['source_r_precision'] = float(metrics['r_precision_source'][i].cpu())
                    record['edit_r_precision'] = float(metrics['r_precision_edit'][i].cpu())
                    record['r_precision_improvement'] = float(metrics['r_precision_improvement'][i].cpu())
                    record['source_r_is_perfect'] = bool(details['source_r_is_perfect'][i].item())
                    record['edit_r_is_perfect'] = bool(details['edit_r_is_perfect'][i].item())

                    rejected_records.append(record)

                all_records.append(record)

            if len(all_records) > 0:
                accept_rate = stats['accepted'] / len(all_records)
                pbar.set_postfix({
                    'Accept': f"{accept_rate:.1%}",
                    'Total': stats['total']
                })

        except Exception as e:
            print(f"\n[Error] Batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果
    print("\nSaving results...")

    accepted_json_path = pjoin(cfg.io.output_dir, 'accepted_pairs.json')
    with open(accepted_json_path, 'w', encoding='utf-8') as f:
        json.dump(accepted_records, f, indent=2, ensure_ascii=False)

    if cfg.filtering.get('save_rejected', True):
        rejected_json_path = pjoin(cfg.io.output_dir, 'rejected_records.json')
        with open(rejected_json_path, 'w', encoding='utf-8') as f:
            json.dump(rejected_records, f, indent=2, ensure_ascii=False)

    all_json_path = pjoin(cfg.io.output_dir, 'all_records.json')
    with open(all_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # 统计报告
    print(f"\n{'=' * 70}")
    print(f"Filtering Complete! (Matching Score主导 + R-Precision上限感知)")
    print(f"Total variations processed: {stats['total']}")
    print(f"✅ Accepted: {stats['accepted']} ({stats['accepted'] / max(1, stats['total']):.1%})")
    print(f"❌ Rejected: {stats['total'] - stats['accepted']} ({1 - stats['accepted'] / max(1, stats['total']):.1%})")

    print(f"\nRejection breakdown:")
    rejection_labels = {
        'hard_r_precision': f'R-Precision < {cfg.filtering.get("min_r_precision_threshold", 0.7)} (硬阈值)',
        'hard_matching': f'Matching Score < {cfg.filtering.get("min_matching_threshold", 0.6)} (硬阈值)',
        'matching_not_improved': f'Matching Score ≤ Source + {cfg.filtering.get("min_matching_margin", 0.02)} (未提升)',
        'rprec_not_improved': 'R-Precision未提升(源<1)',
        'rprec_dropped': 'R-Precision从1下降(源=1但编辑<1)',
        'over_edit': 'Over-editing (low preservation)',
        'under_edit': 'Under-editing (no change)',
        'length_mismatch': 'Length mismatch',
    }

    for reason, count in stats['rejected'].items():
        if count > 0:
            label = rejection_labels.get(reason, reason)
            print(f"  - {label}: {count}")

    print(f"\nSplit breakdown (Accepted):")
    for split in ['train', 'val', 'test']:
        split_accepted = [r for r in accepted_records if r['split'] == split]
        print(f"  [{split}]: {len(split_accepted)}")

    if accepted_records:
        matches = [r['metrics']['matching_score_edit'] for r in accepted_records]
        r_precs = [r['metrics']['r_precision_edit'] for r in accepted_records]
        preserves = [r['metrics']['r_preserve_edit'] for r in accepted_records]

        match_improvements = [
            r['metrics']['matching_score_edit'] - r['metrics']['matching_score_source']
            for r in accepted_records
        ]
        r_prec_improvements = [
            r['metrics']['r_precision_edit'] - r['metrics']['r_precision_source']
            for r in accepted_records
        ]

        # 统计源R-precision=1的情况
        perfect_source_cases = [r for r in accepted_records if r.get('source_r_is_perfect', False)]

        print(f"\nAccepted samples quality:")
        print(f"  Matching Score: {np.mean(matches):.3f} ± {np.std(matches):.3f} (min: {np.min(matches):.3f})")
        print(f"  R Precision (Top-3): {np.mean(r_precs):.3f} ± {np.std(r_precs):.3f}")
        print(f"  Source Preserve: {np.mean(preserves):.3f} ± {np.std(preserves):.3f}")
        print(f"\n📈 Improvement over source:")
        print(f"  Matching Score Δ: {np.mean(match_improvements):.3f} ± {np.std(match_improvements):.3f}")
        print(f"  R Precision Δ: {np.mean(r_prec_improvements):.3f} ± {np.std(r_prec_improvements):.3f}")
        if perfect_source_cases:
            perfect_match_improvements = [
                r['metrics']['matching_score_edit'] - r['metrics']['matching_score_source']
                for r in perfect_source_cases
            ]
            print(f"\n  🎯 源R=1的案例: {len(perfect_source_cases)}个")
            print(
                f"     Matching Score Δ: {np.mean(perfect_match_improvements):.3f} ± {np.std(perfect_match_improvements):.3f}")

    if rejected_records and cfg.filtering.get('save_rejected', True):
        print(f"\n❌ Rejected samples analysis:")

        # R-precision从1下降的案例
        dropped_cases = [r for r in rejected_records if r.get('reject_reason') == 'rprec_dropped'][:3]
        if dropped_cases:
            print(f"\n  ⚠️  R-Precision从1下降的案例 (源=1但编辑<1):")
            for case in dropped_cases:
                src_r = case.get('source_r_precision', 0)
                edit_r = case.get('edit_r_precision', 0)
                src_match = case.get('source_matching_score', 0)
                edit_match = case.get('edit_matching_score', 0)
                print(
                    f"    - {case['original_key']}: R-Prec {src_r:.0f}→{edit_r:.0f}, Match {src_match:.3f}→{edit_match:.3f}")

        # Matching未提升的案例
        matching_fails = [r for r in rejected_records if r.get('reject_reason') == 'matching_not_improved'][:3]
        if matching_fails:
            print(f"\n  ⚠️  Matching Score未提升的案例:")
            for case in matching_fails:
                src_match = case.get('source_matching_score', 0)
                edit_match = case.get('edit_matching_score', 0)
                src_r = case.get('source_r_precision', 0)
                edit_r = case.get('edit_r_precision', 0)
                print(
                    f"    - {case['original_key']}: Match {src_match:.3f}→{edit_match:.3f} (Δ{edit_match - src_match:+.3f}), R={src_r:.0f}→{edit_r:.0f}")

    print(f"\nOutput files:")
    print(f"  Accepted: {accepted_json_path}")
    if cfg.filtering.get('save_rejected', True):
        print(f"  Rejected: {rejected_json_path}")
    print(f"  All: {all_json_path}")
    print(f"{'=' * 70}")


# -----------------------------------------------------------------------------
# 4. 模型加载函数 - 保持不变
# -----------------------------------------------------------------------------
def load_vae_model(cfg: Dict[str, Any], device: torch.device) -> HRVAE:
    """从配置加载 HRVAE 模型"""
    vae_cfg = cfg.vae_cfg

    vae = HRVAE(
        input_width=vae_cfg.data.dim_pose,
        z_dim=vae_cfg.model.z_dim,
        dim=vae_cfg.model.dim,
        dec_dim=vae_cfg.model.dec_dim,
        num_res_blocks=vae_cfg.model.num_res_blocks,
        dropout=vae_cfg.model.dropout,
        dim_mult=vae_cfg.model.dim_mult,
        temperal_downsample=vae_cfg.model.temperal_downsample,
    )

    ckpt_path = cfg.vae_checkpoint

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vae.load_state_dict(ckpt[model_key])
    vae.to(device)
    vae.eval()

    print(f"[VAE] Loaded from {ckpt_path}")
    return vae


def load_diffusion_model(cfg: Dict[str, Any], device: torch.device) -> OmniMoEditDiT:
    """从配置加载 OmniMoEditDiT 模型"""
    d_cfg = cfg.diffusion

    diff_model = OmniMoEditDiT(
        checkpoint_path=d_cfg.get('text_encoder_path', d_cfg.get('checkpoint_path')),
        tokenizer_path=d_cfg.tokenizer_path,
        input_dim=d_cfg.input_dim,
        hidden_dim=d_cfg.hidden_dim,
        ffn_dim=d_cfg.ffn_dim,
        num_layers=d_cfg.num_layers,
        num_heads=d_cfg.num_heads,
        text_dim=d_cfg.text_dim,
        text_len=d_cfg.text_len,
        dropout_prob=0.0,
        noise_steps=d_cfg.noise_steps,
        drop_out=0.0,
        cfg_scale=d_cfg.cfg_scale,
        prediction_type=d_cfg.prediction_type,
        use_text_cond=d_cfg.use_text_cond,
        use_logit_normal=d_cfg.get('use_logit_normal', False),
        time_scale=d_cfg.get('time_scale', 10.0),
    )

    ckpt_path = d_cfg.get('model_checkpoint')
    if not ckpt_path:
        ckpt_path = pjoin(cfg.exp.checkpoint_dir, 'model', d_cfg.which_epoch)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    diff_model.load_state_dict(new_state_dict, strict=False)
    diff_model.to(device)
    diff_model.eval()

    print(f"[Diffusion] Loaded from {ckpt_path}")
    return diff_model


# -----------------------------------------------------------------------------
# 5. 主函数
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='FlowEdit Filter: Matching Score主导 + R-Precision上限感知'
    )
    parser.add_argument('--config', type=str, default='../configs/omni_moedit_filter.yaml',
                        help='Path to config file')
    parser.add_argument('--input_json', type=str, default=None,
                        help='Override input JSON path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--steps', type=int, default=None,
                        help='Override FlowEdit steps')
    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='Override CFG scale')

    # 硬阈值参数
    parser.add_argument('--min_r_precision', type=float, default=None,
                        help='Min R-precision threshold (default: 0.7)')
    parser.add_argument('--min_matching', type=float, default=None,
                        help='Min matching score threshold (default: 0.6)')

    # Matching Score主导参数
    parser.add_argument('--min_matching_margin', type=float, default=None,
                        help='Min margin for matching score improvement over source (default: 0.02)')

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行覆盖
    if args.input_json:
        cfg.io.input_json = args.input_json
    if args.output_dir:
        cfg.io.output_dir = args.output_dir
    if args.batch_size:
        cfg.io.batch_size = args.batch_size
    if args.steps:
        cfg.flowedit.num_steps = args.steps
    if args.cfg_scale:
        cfg.flowedit.cfg_scale_tgt = args.cfg_scale

    # 硬阈值参数覆盖
    if args.min_r_precision is not None:
        cfg.filtering.min_r_precision_threshold = args.min_r_precision
    if args.min_matching is not None:
        cfg.filtering.min_matching_threshold = args.min_matching

    # Matching Score主导参数覆盖
    if args.min_matching_margin is not None:
        cfg.filtering.min_matching_margin = args.min_matching_margin

    device = torch.device(cfg.exp.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 构建路径
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'diff', cfg.exp.diff_name)
    cfg.vae_cfg = load_config(cfg.vae_config)
    cfg.vae_cfg.exp.vae_ckpt = cfg.vae_ckpt

    # 设置默认值 (Matching Score主导 + R-precision上限感知)
    cfg.filtering.setdefault('min_r_precision_threshold', 0.7)  # 硬阈值: R-precision >= 0.7
    cfg.filtering.setdefault('min_matching_threshold', 0.6)  # 硬阈值: Matching >= 0.6
    cfg.filtering.setdefault('min_matching_margin', 0.02)  # Matching必须比源高0.02
    cfg.filtering.setdefault('min_preserve', 0.5)
    cfg.filtering.setdefault('max_preserve', 0.95)
    cfg.filtering.setdefault('require_top1_improvement', False)

    # 加载数据
    meta_dir = pjoin(cfg.data.root_dir, cfg.data.meta_dir)
    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

    dataset = FlowEditDataset(
        json_path=cfg.io.input_json,
        cfg=cfg,
        mean=mean,
        std=std
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.io.batch_size,
        shuffle=False,
        num_workers=cfg.io.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 加载模型
    print("\n[1/3] Loading VAE model...")
    vae = load_vae_model(cfg, device)

    print("\n[2/3] Loading Diffusion model...")
    diff_model = load_diffusion_model(cfg, device)

    print("\n[3/3] Loading Evaluator...")
    eval_cfg = load_config(cfg.evaluator.config_path)
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
    eval_wrapper.eval()

    print("\n[4/4] Running matching-score-first filtering mode...")
    run_filtering_mode(
        cfg=cfg,
        dataloader=dataloader,
        vae=vae,
        diff_model=diff_model,
        eval_wrapper=eval_wrapper,
        device=device
    )


if __name__ == "__main__":
    main()
