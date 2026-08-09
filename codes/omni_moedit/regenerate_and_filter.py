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

from omni_moedit.models.text_to_motion_dit import OmniMoEditDiT
from models_flow.hrvae import HRVAE
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from config.load_config import load_config


# -----------------------------------------------------------------------------
# 1. Dataset 类（兼容扁平列表与原始嵌套格式，支持 data_root 路径重写）
# -----------------------------------------------------------------------------
class ReEditDataset(Dataset):
    def __init__(self, json_path: str, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.data_root = cfg.data.root_dir
        self.latent_dirs = {
            'train': pjoin(self.data_root, cfg.data.latent_dir, 'train'),
            'val':   pjoin(self.data_root, cfg.data.latent_dir, 'val'),
            'test':  pjoin(self.data_root, cfg.data.latent_dir, 'test'),
        }
        self.items = self._parse_json(json_path)
        print(f"[Dataset] Loaded {len(self.items)} items from {json_path}")
        print(f"[Dataset] data_root resolved to: {self.data_root}")

    def _determine_split(self, cid: str) -> str:
        for split_name in ['train', 'val', 'test']:
            path = pjoin(self.latent_dirs[split_name], f"{cid}.npy")
            if os.path.exists(path):
                return split_name
        return 'train'

    def _get_source_path(self, cid: str, split: str) -> str:
        """基于当前 data_root 构建路径（嵌套格式使用）"""
        return pjoin(self.latent_dirs[split], f"{cid}.npy")

    def _resolve_path(self, stored_path: str) -> str:
        """
        将路径的前三级目录替换为当前 data_root。
        例：../data/SnapMoGen/latents/.../x.npy
            → ../data/SnapMoGen/latents/.../x.npy
        """
        if not stored_path or not isinstance(stored_path, str):
            return stored_path
        # 统一使用 posix separator 分割，再转回系统路径
        parts = stored_path.replace('\\', '/').split('/')
        parts = [p for p in parts if p]
        if len(parts) <= 3:
            return stored_path
        suffix = os.path.join(*parts[3:])
        return pjoin(self.data_root, suffix)

    def _find_latent(self, cid: str, preferred_path: str = None):
        """
        按优先级查找 latent 文件：
        1. preferred_path（若提供且存在）
        2. preferred_path 经 _resolve_path 修正后（若原路径不存在）
        3. 按 split 目录搜索
        返回: (latent_array, success, original_length, resolved_path, split)
        """
        # 尝试 1：原始路径
        if preferred_path and os.path.exists(preferred_path):
            latent = np.load(preferred_path)
            return latent, True, len(latent), preferred_path, None

        # 尝试 2：前三级替换为 data_root
        if preferred_path:
            resolved = self._resolve_path(preferred_path)
            if resolved != preferred_path and os.path.exists(resolved):
                latent = np.load(resolved)
                return latent, True, len(latent), resolved, None

        # 尝试 3：按 split 目录搜索
        for split_name, dir_path in self.latent_dirs.items():
            path = pjoin(dir_path, f"{cid}.npy")
            if os.path.exists(path):
                latent = np.load(path)
                return latent, True, len(latent), path, split_name

        # 兜底：返回零张量
        latent_dim = self.cfg.diffusion.input_dim
        return np.zeros((10, latent_dim)), True, 10, None, 'train'

    def _parse_json(self, json_path: str) -> List[Dict]:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        items = []
        if not isinstance(raw_data, list):
            raw_data = [raw_data]

        for entry in raw_data:
            # ---------- 扁平格式（如 accepted_pairs.json）----------
            if 'source_caption' in entry and 'target_caption' in entry:
                cid = entry.get('original_key', '')
                split = entry.get('split')
                stored_src_path = entry.get('source_path', '')

                # 若 JSON 中未提供 split，尝试推断
                if not split:
                    split = self._determine_split(cid)

                # 对 source_path 做前三级替换
                resolved_src_path = self._resolve_path(stored_src_path) if stored_src_path else None
                if not resolved_src_path or not os.path.exists(resolved_src_path):
                    resolved_src_path = self._get_source_path(cid, split)

                items.append({
                    'original_key': cid,
                    'split': split,
                    'source_path': resolved_src_path,
                    'source_caption': entry['source_caption'],
                    'target_caption': entry['target_caption'],
                    'edit_command': entry.get('edit_command', ''),
                    'reverse_edit_command': entry.get('reverse_edit_command', ''),
                    'variation_idx': entry.get('variation_idx', 0),
                    'total_variations': entry.get('total_variations', 1),
                })

            # ---------- 原始嵌套格式 ----------
            elif 'original_caption' in entry:
                cid = entry['original_key']
                src_cap = entry['original_caption']
                split = self._determine_split(cid)
                src_path = self._get_source_path(cid, split)

                variations = entry.get('edits', entry.get('variations', []))
                for var_idx, var in enumerate(variations):
                    if 'edits' in entry:
                        tgt_cap = var.get('target_caption', '')
                        rev_cmd = var.get('reverse_command', '')
                    else:
                        tgt_cap = var.get('new_caption', '')
                        rev_cmd = var.get('reverse_edit_command', '')

                    items.append({
                        'original_key': cid,
                        'split': split,
                        'source_path': src_path,
                        'source_caption': src_cap,
                        'target_caption': tgt_cap,
                        'edit_command': var.get('edit_command', ''),
                        'reverse_edit_command': rev_cmd,
                        'variation_idx': var_idx,
                        'total_variations': len(variations),
                    })
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        cid = item['original_key']
        stored_src_path = item.get('source_path', '')

        # 使用 _find_latent 按优先级查找
        latent, is_latent, original_length, found_path, found_split = self._find_latent(cid, stored_src_path)

        # 若通过搜索找到了新的 split/path，更新 item 信息
        if found_split and found_split != item['split']:
            item['split'] = found_split
        if found_path:
            item['source_path'] = found_path

        return {
            'latent': torch.from_numpy(latent).float(),
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
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    lengths = [len(b['latent']) for b in batch]
    max_len = max(lengths)
    feat_dim = batch[0]['latent'].shape[-1]
    bsz = len(batch)

    padded = torch.zeros(bsz, max_len, feat_dim)
    for i, b in enumerate(batch):
        l = len(b['latent'])
        padded[i, :l] = b['latent']

    return {
        'latent': padded,
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
# 2. 核心评估逻辑（语义特征级保留度 + 双候选选择）
# -----------------------------------------------------------------------------
def compute_metrics(
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
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    decoded_lengths = latent_lengths * cfg.data.downsample_ratio
    feat_dim = getattr(cfg.data, 'motion_dim', 148)

    with torch.no_grad():
        src_dec = vae.decode(source_latents)
        edit_dec = vae.decode(edited_latents)
        base_dec = vae.decode(base_latents)

    src_motion = src_dec[..., :feat_dim]
    edit_motion = edit_dec[..., :feat_dim]
    base_motion = base_dec[..., :feat_dim]

    # Text Embedding
    text_emb, _ = eval_wrapper.encode_text(target_captions)
    text_emb_norm = F.normalize(text_emb, p=2, dim=-1)

    # Motion Embedding
    _, src_emb, _ = eval_wrapper.encode_motion(src_motion, decoded_lengths)
    _, edit_emb, _ = eval_wrapper.encode_motion(edit_motion, decoded_lengths)
    _, base_emb, _ = eval_wrapper.encode_motion(base_motion, decoded_lengths)

    src_emb_norm = F.normalize(src_emb, p=2, dim=-1)
    edit_emb_norm = F.normalize(edit_emb, p=2, dim=-1)
    base_emb_norm = F.normalize(base_emb, p=2, dim=-1)

    # Matching Score（与目标文本）
    match_src   = (src_emb_norm   * text_emb_norm).sum(dim=-1)
    match_edit  = (edit_emb_norm  * text_emb_norm).sum(dim=-1)
    match_base  = (base_emb_norm  * text_emb_norm).sum(dim=-1)

    # 结构保留度（语义空间余弦相似度）
    preserve_edit = (src_emb_norm * edit_emb_norm).sum(dim=-1)
    preserve_base = (src_emb_norm * base_emb_norm).sum(dim=-1)

    # R-precision（Top-3）
    def r_precision(sim_matrix):
        B = sim_matrix.shape[0]
        top3 = torch.argsort(sim_matrix, dim=1, descending=True)[:, :3]
        correct = torch.arange(B, device=device).unsqueeze(1)
        return (top3 == correct).any(dim=1).float()

    sim_src  = torch.mm(src_emb_norm,  text_emb_norm.t())
    sim_edit = torch.mm(edit_emb_norm, text_emb_norm.t())
    sim_base = torch.mm(base_emb_norm, text_emb_norm.t())

    rprec_src  = r_precision(sim_src)
    rprec_edit = r_precision(sim_edit)
    rprec_base = r_precision(sim_base)

    edit_metrics = {
        'matching_score': match_edit,
        'r_precision': rprec_edit,
        'preserve': preserve_edit,
        'magnitude': 1.0 - preserve_edit,
    }
    base_metrics = {
        'matching_score': match_base,
        'r_precision': rprec_base,
        'preserve': preserve_base,
        'magnitude': 1.0 - preserve_base,
    }
    source_metrics = {
        'matching_score': match_src,
        'r_precision': rprec_src,
    }
    return edit_metrics, base_metrics, source_metrics


# -----------------------------------------------------------------------------
# 3. 主筛选与保存逻辑
# -----------------------------------------------------------------------------
def run_reedit_filtering(
    cfg: Dict[str, Any],
    dataloader: DataLoader,
    vae: HRVAE,
    diff_model: OmniMoEditDiT,
    eval_wrapper: EvaluatorWrapper,
    device: torch.device
):
    print(f"\n{'='*70}")
    print("[RE-EDIT MODE] 优化筛选：Matching>0.7 | 提升>0.1 | 语义保留<0.8 | 双候选择优")
    print(f"FlowEdit: steps={cfg.flowedit.num_steps}, cfg_scale={cfg.flowedit.cfg_scale_tgt}")
    print(f"{'='*70}\n")

    os.makedirs(cfg.io.output_dir, exist_ok=True)
    out_dirs = {}
    for split in ['train', 'val', 'test']:
        out_dirs[split] = pjoin(cfg.io.output_dir, split)
        os.makedirs(out_dirs[split], exist_ok=True)

    accepted_records = []
    rejected_records = []
    stats = {
        'total': 0, 'accepted': 0,
        'rejected': {
            'hard_matching': 0,
            'matching_improvement': 0,
            'edit_magnitude': 0,
        }
    }

    pbar = tqdm(dataloader, desc="Re-edit & Filter")
    for batch_idx, batch in enumerate(pbar):
        if not batch:
            continue
        try:
            B = len(batch['original_key'])
            stats['total'] += B

            source_latents = batch['latent'].to(device)
            latent_lengths = batch['latent_length'].to(device)
            max_len = source_latents.shape[1]

            with torch.no_grad():
                # FlowEdit
                edit_out = diff_model.flow_edit(
                    x={
                        "feature": source_latents,
                        "text": batch['source_caption'],
                        "feature_length": latent_lengths,
                    },
                    target_text=batch['target_caption'],
                    num_steps=cfg.flowedit.num_steps,
                    cfg_scale_tgt=cfg.flowedit.cfg_scale_tgt
                )
                edited_latents = edit_out['generated']

                # Base 直接生成
                base_out = diff_model.generate(
                    x={
                        "text": batch['target_caption'],
                        "feature_length": latent_lengths,
                    },
                    num_denoise_steps=cfg.flowedit.num_steps
                )
                base_latents = base_out['generated']

            # 长度对齐
            def align_len(x, target_len):
                if x.shape[1] == target_len:
                    return x
                if x.shape[1] > target_len:
                    return x[:, :target_len, :]
                pad = target_len - x.shape[1]
                return torch.cat([x, x[:, -1:, :].expand(B, pad, x.shape[-1])], dim=1)

            edited_latents = align_len(edited_latents, max_len)
            base_latents = align_len(base_latents, max_len)

            # 评估
            edit_m, base_m, src_m = compute_metrics(
                eval_wrapper, source_latents, edited_latents, base_latents,
                latent_lengths, batch['target_caption'], batch['source_caption'],
                vae, cfg, device
            )

            for i in range(B):
                key = batch['original_key'][i]
                var_idx = batch['variation_idx'][i]
                split = batch['split'][i]
                src_path = batch['source_path'][i]
                actual_len = int(latent_lengths[i].item())
                base_name = f"{key.replace('#', '_')}_var{var_idx}"

                # 双候选择优：选 matching_score 更高的
                if edit_m['matching_score'][i].item() >= base_m['matching_score'][i].item():
                    chosen_type = 'flowedit'
                    chosen_latent = edited_latents[i, :actual_len]
                    chosen_match = edit_m['matching_score'][i]
                    chosen_rprec = edit_m['r_precision'][i]
                    chosen_preserve = edit_m['preserve'][i]
                    chosen_magnitude = edit_m['magnitude'][i]
                else:
                    chosen_type = 'base'
                    chosen_latent = base_latents[i, :actual_len]
                    chosen_match = base_m['matching_score'][i]
                    chosen_rprec = base_m['r_precision'][i]
                    chosen_preserve = base_m['preserve'][i]
                    chosen_magnitude = base_m['magnitude'][i]

                src_match = src_m['matching_score'][i]

                # 筛选条件
                cond_hard = chosen_match > 0.7
                cond_improve = (chosen_match - src_match) > 0.1
                cond_magnitude = chosen_preserve < 0.8  # 编辑幅度 = 1-preserve > 0.2
                passed = cond_hard and cond_improve and cond_magnitude

                record = {
                    'original_key': key,
                    'split': split,
                    'variation_idx': var_idx,
                    'total_variations': batch['total_variations'][i],
                    'source_path': src_path,
                    'edited_path': None,
                    'chosen_type': chosen_type,
                    'original_length': int(batch['original_length'][i].item()),
                    'edited_length': actual_len,
                    'source_caption': batch['source_caption'][i],
                    'target_caption': batch['target_caption'][i],
                    'edit_command': batch['edit_command'][i],
                    'reverse_edit_command': batch['reverse_edit_command'][i],
                    'metrics': {
                        'matching_score_source': float(src_match.cpu()),
                        'matching_score_edit': float(edit_m['matching_score'][i].cpu()),
                        'matching_score_base': float(base_m['matching_score'][i].cpu()),
                        'matching_score_chosen': float(chosen_match.cpu()),
                        'r_precision_chosen': float(chosen_rprec.cpu()),
                        'preserve_chosen': float(chosen_preserve.cpu()),
                        'magnitude_chosen': float(chosen_magnitude.cpu()),
                        'matching_improvement': float((chosen_match - src_match).cpu()),
                    },
                    'status': 'accepted' if passed else 'rejected',
                    'reject_reason': None,
                }

                if passed:
                    stats['accepted'] += 1
                    save_path = pjoin(out_dirs[split], f"{base_name}.npy")
                    np.save(save_path, chosen_latent.cpu().numpy())
                    record['edited_path'] = save_path
                    accepted_records.append(record)
                else:
                    reasons = []
                    if not cond_hard:
                        reasons.append('hard_matching')
                        stats['rejected']['hard_matching'] += 1
                    if not cond_improve:
                        reasons.append('matching_improvement')
                        stats['rejected']['matching_improvement'] += 1
                    if not cond_magnitude:
                        reasons.append('edit_magnitude')
                        stats['rejected']['edit_magnitude'] += 1
                    record['reject_reason'] = reasons[0] if reasons else 'unknown'
                    record['reject_details'] = reasons
                    rejected_records.append(record)

            pbar.set_postfix({
                'Accept': f"{stats['accepted']}/{stats['total']}",
                'Rate': f"{stats['accepted']/max(1,stats['total']):.1%}"
            })

        except Exception as e:
            print(f"\n[Error] Batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果
    acc_path = pjoin(cfg.io.output_dir, 'accepted_reedit.json')
    rej_path = pjoin(cfg.io.output_dir, 'rejected_reedit.json')
    with open(acc_path, 'w', encoding='utf-8') as f:
        json.dump(accepted_records, f, indent=2, ensure_ascii=False)
    with open(rej_path, 'w', encoding='utf-8') as f:
        json.dump(rejected_records, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Re-edit Filtering Complete")
    print(f"Total: {stats['total']} | Accepted: {stats['accepted']} ({stats['accepted']/max(1,stats['total']):.1%})")
    print(f"Rejected breakdown:")
    for k, v in stats['rejected'].items():
        if v > 0:
            print(f"  - {k}: {v}")
    print(f"\nOutput: {cfg.io.output_dir}")
    print(f"  Accepted JSON: {acc_path}")
    print(f"  Rejected JSON: {rej_path}")
    print(f"{'='*70}")


# -----------------------------------------------------------------------------
# 4. 模型加载
# -----------------------------------------------------------------------------
def load_vae_model(cfg: Dict[str, Any], device: torch.device) -> HRVAE:
    vcfg = cfg.vae_cfg
    vae = HRVAE(
        input_width=vcfg.data.dim_pose,
        z_dim=vcfg.model.z_dim,
        dim=vcfg.model.dim,
        dec_dim=vcfg.model.dec_dim,
        num_res_blocks=vcfg.model.num_res_blocks,
        dropout=vcfg.model.dropout,
        dim_mult=vcfg.model.dim_mult,
        temperal_downsample=vcfg.model.temperal_downsample,
    )
    ckpt_path = cfg.vae_checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vae.load_state_dict(ckpt[key])
    vae.to(device).eval()
    print(f"[VAE] Loaded from {ckpt_path}")
    return vae


def load_diffusion_model(cfg: Dict[str, Any], device: torch.device) -> OmniMoEditDiT:
    d = cfg.diffusion
    model = OmniMoEditDiT(
        checkpoint_path=d.checkpoint_path,
        tokenizer_path=d.tokenizer_path,
        input_dim=d.input_dim,
        hidden_dim=d.hidden_dim,
        ffn_dim=d.ffn_dim,
        num_layers=d.num_layers,
        num_heads=d.num_heads,
        text_dim=d.text_dim,
        text_len=d.text_len,
        dropout_prob=0.0,
        noise_steps=d.noise_steps,
        drop_out=0.0,
        cfg_scale=d.cfg_scale,
        prediction_type=d.prediction_type,
        use_text_cond=d.use_text_cond,
        use_logit_normal=d.get('use_logit_normal', False),
        time_scale=d.get('time_scale', 10.0),
    )
    ckpt_path = pjoin(cfg.exp.checkpoint_dir, 'model', d.which_epoch)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"[Diffusion] Loaded from {ckpt_path}")
    return model


# -----------------------------------------------------------------------------
# 5. 主函数
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Re-edit Filtering with Optimized Criteria')
    parser.add_argument('--config', type=str, default='../configs/omni_moedit_regenerate.yaml')
    parser.add_argument('--input_json', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--cfg_scale', type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
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

    device = torch.device(cfg.exp.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'diff', cfg.exp.diff_name)
    cfg.vae_cfg = load_config(cfg.vae_config)
    cfg.vae_cfg.exp.vae_ckpt = cfg.vae_ckpt

    dataset = ReEditDataset(cfg.io.input_json, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.io.batch_size,
        shuffle=False,
        num_workers=cfg.io.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("\n[1/3] Loading VAE...")
    vae = load_vae_model(cfg, device)
    print("\n[2/3] Loading Diffusion...")
    diff_model = load_diffusion_model(cfg, device)
    print("\n[3/3] Loading Evaluator...")
    eval_cfg = load_config(cfg.evaluator.config_path)
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
    eval_wrapper.eval()

    print("\n[4/4] Running re-edit filtering...")
    run_reedit_filtering(cfg, dataloader, vae, diff_model, eval_wrapper, device)


if __name__ == "__main__":
    main()
