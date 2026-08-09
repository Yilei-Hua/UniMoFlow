import sys as _release_sys
from pathlib import Path as _ReleasePath

_CODES_ROOT = _ReleasePath(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_CODES_ROOT))

import os
import json
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

from models_flow.hrvae import HRVAE
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from config.load_config import load_config


# -----------------------------------------------------------------------------
# 1. Dataset：从多个 JSON 加载，实时读取 latent 并验证路径
# -----------------------------------------------------------------------------
class SecondaryFilterDataset(Dataset):
    def __init__(self, json_paths: List[str], cfg: Dict[str, Any], data_root: str = None):
        self.cfg = cfg
        self.data_root = data_root
        self.items: List[Dict[str, Any]] = []
        for jp in json_paths:
            self._load_json(jp)
        print(f"[Dataset] 共加载 {len(self.items)} 条有效记录，来自 {len(json_paths)} 个文件")

    def _resolve_path(self, stored_path: str) -> str:
        """若提供 data_root，将路径前三级替换为当前 root"""
        if not self.data_root or not stored_path or not isinstance(stored_path, str):
            return stored_path
        parts = stored_path.replace("\\", "/").split("/")
        parts = [p for p in parts if p]
        if len(parts) <= 3:
            return stored_path
        suffix = os.path.join(*parts[3:])
        resolved = pjoin(self.data_root, suffix)
        return resolved if os.path.exists(resolved) else stored_path

    def _find_file(self, path: str) -> Tuple[str, bool]:
        if os.path.exists(path):
            return path, True
        resolved = self._resolve_path(path)
        if resolved != path and os.path.exists(resolved):
            return resolved, True
        return path, False

    def _load_json(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        for entry in data:
            if not isinstance(entry, dict) or "original_key" not in entry or "edited_path" not in entry:
                continue

            src_path = entry.get("source_path", "")
            edit_path = entry.get("edited_path", "")

            src_actual, src_ok = self._find_file(src_path)
            edit_actual, edit_ok = self._find_file(edit_path)

            if not src_ok:
                print(f"[Warning] 跳过：source 不存在 {src_path}")
                continue
            if not edit_ok:
                print(f"[Warning] 跳过：edited 不存在 {edit_path}")
                continue

            try:
                src_latent = np.load(src_actual)
                edit_latent = np.load(edit_actual)
            except Exception as e:
                print(f"[Warning] 加载 npy 失败，跳过: {e}")
                continue

            self.items.append({
                "original_key": entry["original_key"],
                "split": entry.get("split", "train"),
                "source_path": src_actual,
                "edited_path": edit_actual,
                "source_caption": entry.get("source_caption", ""),
                "target_caption": entry.get("target_caption", ""),
                "edit_command": entry.get("edit_command", ""),
                "reverse_edit_command": entry.get("reverse_edit_command", ""),
                "variation_idx": entry.get("variation_idx", 0),
                "total_variations": entry.get("total_variations", 1),
                "source_latent": torch.from_numpy(src_latent).float(),
                "edited_latent": torch.from_numpy(edit_latent).float(),
                "source_length": len(src_latent),
                "edited_length": len(edit_latent),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    B = len(batch)
    src_lengths = [len(b["source_latent"]) for b in batch]
    edit_lengths = [len(b["edited_latent"]) for b in batch]
    max_src_len = max(src_lengths)
    max_edit_len = max(edit_lengths)
    feat_dim = batch[0]["source_latent"].shape[-1]

    src_padded = torch.zeros(B, max_src_len, feat_dim)
    edit_padded = torch.zeros(B, max_edit_len, feat_dim)

    for i, b in enumerate(batch):
        src_padded[i, :src_lengths[i]] = b["source_latent"]
        edit_padded[i, :edit_lengths[i]] = b["edited_latent"]

    return {
        "source_latent": src_padded,
        "edited_latent": edit_padded,
        "source_length": torch.tensor(src_lengths),
        "edited_length": torch.tensor(edit_lengths),
        "original_key": [b["original_key"] for b in batch],
        "split": [b["split"] for b in batch],
        "source_path": [b["source_path"] for b in batch],
        "edited_path": [b["edited_path"] for b in batch],
        "source_caption": [b["source_caption"] for b in batch],
        "target_caption": [b["target_caption"] for b in batch],
        "edit_command": [b["edit_command"] for b in batch],
        "reverse_edit_command": [b["reverse_edit_command"] for b in batch],
        "variation_idx": [b["variation_idx"] for b in batch],
        "total_variations": [b["total_variations"] for b in batch],
    }


# -----------------------------------------------------------------------------
# 2. 指标重新计算（仅计算二次筛选所需指标）
# -----------------------------------------------------------------------------
def compute_metrics_secondary(
    eval_wrapper: EvaluatorWrapper,
    source_latents: torch.Tensor,
    edited_latents: torch.Tensor,
    source_lengths: torch.Tensor,
    edited_lengths: torch.Tensor,
    target_captions: List[str],
    vae: HRVAE,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    重新计算：
      - match_src / match_edit：与 target caption 的 matching score
      - preserve_edit：source 与 edited 在语义空间的结构保留度（余弦相似度）
    """
    feat_dim = getattr(cfg.data, "motion_dim", 148)
    downsample_ratio = getattr(cfg.data, "downsample_ratio", 4)

    with torch.no_grad():
        src_dec = vae.decode(source_latents)
        edit_dec = vae.decode(edited_latents)

    src_motion = src_dec[..., :feat_dim]
    edit_motion = edit_dec[..., :feat_dim]

    src_dec_len = source_lengths * downsample_ratio
    edit_dec_len = edited_lengths * downsample_ratio

    # Text embedding
    text_emb, _ = eval_wrapper.encode_text(target_captions)
    text_emb_norm = F.normalize(text_emb, p=2, dim=-1)

    # Motion embedding
    _, src_emb, _ = eval_wrapper.encode_motion(src_motion, src_dec_len)
    _, edit_emb, _ = eval_wrapper.encode_motion(edit_motion, edit_dec_len)

    src_emb_norm = F.normalize(src_emb, p=2, dim=-1)
    edit_emb_norm = F.normalize(edit_emb, p=2, dim=-1)

    # Matching score with target text
    match_src = (src_emb_norm * text_emb_norm).sum(dim=-1)
    match_edit = (edit_emb_norm * text_emb_norm).sum(dim=-1)

    # Structure preservation (source vs edited cosine similarity)
    preserve_edit = (src_emb_norm * edit_emb_norm).sum(dim=-1)

    return {
        "match_src": match_src,
        "match_edit": match_edit,
        "preserve_edit": preserve_edit,
    }


# -----------------------------------------------------------------------------
# 3. 二次筛选主逻辑
# -----------------------------------------------------------------------------
def run_secondary_filter(
    cfg: Dict[str, Any],
    dataloader: DataLoader,
    vae: HRVAE,
    eval_wrapper: EvaluatorWrapper,
    device: torch.device,
    output_dir: str,
    min_align_improvement: float = 0.1,
    min_preserve: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    edited_out_dir = pjoin(output_dir, "edited")
    for split in ["train", "val", "test"]:
        os.makedirs(pjoin(edited_out_dir, split), exist_ok=True)

    accepted_records: List[Dict] = []
    rejected_records: List[Dict] = []
    stats = {
        "total": 0,
        "accepted": 0,
        "rejected": {"align_improvement": 0, "preserve": 0, "copy_failed": 0},
    }

    pbar = tqdm(dataloader, desc="Secondary Filter")
    for batch in pbar:
        if not batch:
            continue

        B = len(batch["original_key"])
        stats["total"] += B

        source_latents = batch["source_latent"].to(device)
        edited_latents = batch["edited_latent"].to(device)
        source_lengths = batch["source_length"].to(device)
        edited_lengths = batch["edited_length"].to(device)

        # 重新计算指标
        metrics = compute_metrics_secondary(
            eval_wrapper,
            source_latents,
            edited_latents,
            source_lengths,
            edited_lengths,
            batch["target_caption"],
            vae,
            cfg,
            device,
        )

        for i in range(B):
            key = batch["original_key"][i]
            var_idx = batch["variation_idx"][i]
            split = batch["split"][i]
            src_path = batch["source_path"][i]
            edit_path = batch["edited_path"][i]
            base_name = f"{key.replace('#', '_')}_var{var_idx}.npy"

            match_src = metrics["match_src"][i].item()
            match_edit = metrics["match_edit"][i].item()
            preserve_edit = metrics["preserve_edit"][i].item()
            align_improvement = match_edit - match_src

            # 筛选条件
            cond_align = align_improvement >= min_align_improvement
            cond_preserve = preserve_edit > min_preserve
            passed = cond_align and cond_preserve

            record = {
                "original_key": key,
                "split": split,
                "variation_idx": var_idx,
                "total_variations": batch["total_variations"][i],
                "source_path": src_path,
                "edited_path": None,
                "source_caption": batch["source_caption"][i],
                "target_caption": batch["target_caption"][i],
                "edit_command": batch["edit_command"][i],
                "reverse_edit_command": batch["reverse_edit_command"][i],
                "source_length": int(source_lengths[i].item()),
                "edited_length": int(edited_lengths[i].item()),
                "secondary_metrics": {
                    "match_src": round(match_src, 6),
                    "match_edit": round(match_edit, 6),
                    "align_improvement": round(align_improvement, 6),
                    "preserve_edit": round(preserve_edit, 6),
                },
                "status": "accepted" if passed else "rejected",
                "reject_reason": None,
            }

            if passed:
                new_path = pjoin(edited_out_dir, split, base_name)
                try:
                    shutil.copy2(edit_path, new_path)
                    record["edited_path"] = new_path
                    accepted_records.append(record)
                    stats["accepted"] += 1
                except Exception as e:
                    print(f"\n[Error] 复制失败 {edit_path} → {new_path}: {e}")
                    record["status"] = "rejected"
                    record["reject_reason"] = "copy_failed"
                    record["reject_details"] = ["copy_failed"]
                    rejected_records.append(record)
                    stats["rejected"]["copy_failed"] += 1
            else:
                reasons = []
                if not cond_align:
                    reasons.append("align_improvement")
                    stats["rejected"]["align_improvement"] += 1
                if not cond_preserve:
                    reasons.append("preserve")
                    stats["rejected"]["preserve"] += 1
                record["reject_reason"] = reasons[0] if reasons else "unknown"
                record["reject_details"] = reasons
                rejected_records.append(record)

            pbar.set_postfix({
                "Accept": f"{stats['accepted']}/{stats['total']}",
                "Rate": f"{stats['accepted'] / max(1, stats['total']):.1%}",
            })

    # 保存结果 JSON
    acc_path = pjoin(output_dir, "accepted_secondary.json")
    rej_path = pjoin(output_dir, "rejected_secondary.json")
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(accepted_records, f, indent=2, ensure_ascii=False)
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump(rejected_records, f, indent=2, ensure_ascii=False)

    # 打印统计
    print(f"\n{'='*70}")
    print("[Secondary Filter] 二次筛选完成")
    print(f"{'='*70}")
    print(f"  输入总数:     {stats['total']}")
    print(f"  通过数:       {stats['accepted']} ({stats['accepted']/max(1,stats['total']):.1%})")
    print(f"  拒绝数:       {stats['total'] - stats['accepted']}")
    print(f"\n  拒绝原因明细:")
    for k, v in stats["rejected"].items():
        if v > 0:
            print(f"    - {k}: {v}")
    print(f"\n  输出目录: {output_dir}")
    print(f"    通过样本: {edited_out_dir}")
    print(f"    通过记录: {acc_path}")
    print(f"    拒绝记录: {rej_path}")
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
    key = "vq_model" if "vq_model" in ckpt else "model"
    vae.load_state_dict(ckpt[key])
    vae.to(device).eval()
    print(f"[VAE] Loaded from {ckpt_path}")
    return vae


# -----------------------------------------------------------------------------
# 5. 主函数
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="二次筛选：重新计算指标并过滤编辑对")
    parser.add_argument("--config", type=str, default="../configs/omni_moedit_regenerate.yaml")
    parser.add_argument("--input_jsons", nargs="+", required=True, help="一个或多个已筛选 JSON 文件")
    parser.add_argument("--output_dir", type=str, required=True, help="二次筛选输出目录")
    parser.add_argument("--data_root", type=str, default=None, help="若路径前缀变更，提供新 data_root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_align_improvement", type=float, default=0.1, help="匹配度最小提升阈值")
    parser.add_argument("--min_preserve", type=float, default=0.5, help="结构保留度最小阈值")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    device = torch.device(cfg.exp.device if torch.cuda.is_available() else "cpu")

    # VAE 配置
    cfg.vae_cfg = load_config(cfg.vae_config)
    cfg.vae_cfg.exp.vae_ckpt = cfg.vae_ckpt

    print(f"[Config] Device: {device}")
    print(f"[Config] Data root: {cfg.data.root_dir}")

    # Dataset & Dataloader
    dataset = SecondaryFilterDataset(args.input_jsons, cfg, data_root=args.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 加载模型
    print("\n[1/2] Loading VAE...")
    vae = load_vae_model(cfg, device)

    print("\n[2/2] Loading Evaluator...")
    eval_cfg = load_config(cfg.evaluator.config_path)
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
    eval_wrapper.eval()

    # 运行筛选
    print("\n[3/3] Running secondary filter...")
    run_secondary_filter(
        cfg,
        dataloader,
        vae,
        eval_wrapper,
        device,
        args.output_dir,
        args.min_align_improvement,
        args.min_preserve,
    )


if __name__ == "__main__":
    main()
