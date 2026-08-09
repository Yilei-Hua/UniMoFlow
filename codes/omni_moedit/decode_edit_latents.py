#!/usr/bin/env python3
import sys as _release_sys
from pathlib import Path as _ReleasePath

_CODES_ROOT = _ReleasePath(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_CODES_ROOT))

# decode_edit_latents.py
"""
将编辑数据集中的 source / edited latent 通过 VAE 解码回动作数据，
并生成记录新路径的新 JSON。
"""

import os
import sys
import json
import argparse
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

# ========== 以下导入请根据你的实际项目结构调整 ==========
from config.load_config import load_config
from models_flow.hrvae import HRVAE
from common.skeleton import Skeleton
from utils import bvh_io
from utils.motion_process_bvh import recover_pos_from_rot
from utils.paramUtil import kinematic_chain
# =========================================================


def replace_top3(path: Path, new_root: Optional[str]) -> Path:
    """
    若 path 不存在且提供了 new_root，将 path 的前三级目录替换为 new_root。
    例：
        ../data/SnapMoGen/latents_hrvae_detail/train/xxx.npy
        + new_root='../data/SnapMoGen'
        -> ../data/SnapMoGen/latents_hrvae_detail/train/xxx.npy
    """
    if new_root is None or path.exists():
        return path

    parts = path.parts
    # 至少需要: 根目录 + 3 级目录 + 1 个后续路径元素
    if len(parts) >= 5:
        # parts[1:4] 为前三级目录，parts[4:] 为剩余相对路径
        rel = Path(*parts[4:])
        candidate = Path(new_root) / rel
        if candidate.exists():
            return candidate
    return path


def load_vae(vae_yaml_path: str, vae_ckpt_path: str, device: torch.device):
    """
    加载 VAE 模型（与 train_unimoflow.py 中的逻辑保持一致）
    """
    vae_cfg = load_config(vae_yaml_path)

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

    ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vae.load_state_dict(ckpt[model_key])

    vae.to(device)
    vae.eval()
    print(f"[VAE] Loaded from {vae_ckpt_path} (epoch {ckpt.get('ep', 'unknown')})")
    return vae, vae_cfg


def load_skeleton(bvh_template_path: str, device: torch.device):
    """
    加载 Skeleton 模板（用于前向运动学）
    """
    if not Path(bvh_template_path).exists():
        raise FileNotFoundError(f"Skeleton BVH template not found: {bvh_template_path}")
    template_anim = bvh_io.load(bvh_template_path)
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)
    return skeleton


def inv_transform(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    反归一化（与 MotionEditDataset 保持一致）
    data: (..., D)
    """
    D = data.shape[-1]
    return data * std[:D] + mean[:D]


def decode_latent(vae, latent_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    将单个 latent 文件解码回动作特征空间。
    latent_np: (T, D) 或 (T, P, D)
    Returns: (T, input_dim)  numpy
    """
    # 转为 tensor 并加 batch 维度
    x = torch.from_numpy(latent_np).float().to(device)
    if x.dim() == 2:
        x = x.unsqueeze(0)          # (1, T, D)
    elif x.dim() == 3:
        x = x.unsqueeze(0)          # (1, T, P, D)
    else:
        raise ValueError(f"Unsupported latent dim: {x.dim()}, shape: {latent_np.shape}")

    with torch.no_grad():
        # 兼容多种可能的 decode 接口
        if hasattr(vae, 'decode'):
            recon = vae.decode(x)
        elif hasattr(vae, 'decoder'):
            recon = vae.decoder(x)
        else:
            out = vae(x)
            recon = out[0] if isinstance(out, (tuple, list)) else out

    # 去掉 batch 维并转回 numpy
    recon = recon.squeeze(0).cpu().numpy()
    return recon


def latent_to_positions(
    motion_data: np.ndarray,
    skeleton: Skeleton,
    joints_num: int,
    device: torch.device
) -> np.ndarray:
    """
    通过前向运动学将动作表示转为全局关节位置。
    motion_data: (T, D)  numpy 或 torch
    Returns: (T, joints_num, 3) numpy
    """
    if isinstance(motion_data, np.ndarray):
        motion_data = torch.from_numpy(motion_data).float().to(device)

    # recover_pos_from_rot 通常支持 (B, T, D) 或 (T, D)
    if motion_data.dim() == 2:
        motion_data = motion_data.unsqueeze(0)   # (1, T, D)

    global_pos = recover_pos_from_rot(
        motion_data,
        joints_num=joints_num,
        skeleton=skeleton
    )
    # 返回 (T, J, 3)
    return global_pos.squeeze(0).cpu().numpy()


def process_single_pair(
    pair: Dict[str, Any],
    vae,
    vae_cfg,
    device: torch.device,
    out_dir: Path,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    skeleton: Optional[Skeleton],
    joints_num: int,
    output_mode: str,      # 'motion', 'position', 'both'
    data_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    处理单个编辑对：解码 source & edited，保存文件，返回更新后的字典。
    """
    new_pair = dict(pair)   # 深拷贝，避免污染原数据

    # 保留原始 latent 路径到新字段（方便追溯）
    new_pair["source_latent_path"] = pair.get("source_path", "")
    new_pair["edited_latent_path"] = pair.get("edited_path", "")

    split = pair.get("split", "train")
    original_key = pair.get("original_key", "unknown")
    variation_idx = pair.get("variation_idx", 0)

    # 构建输出目录
    motion_dir = out_dir / "motion" / split
    position_dir = out_dir / "position" / split
    motion_dir.mkdir(parents=True, exist_ok=True)
    if output_mode in ("position", "both"):
        position_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 处理 Source ----------
    src_latent_path = Path(pair["source_path"])
    src_latent_path = replace_top3(src_latent_path, data_root)

    if not src_latent_path.exists():
        raise FileNotFoundError(
            f"Source latent not found: {src_latent_path} "
            f"(original: {pair.get('source_path', 'N/A')})"
        )

    src_latent = np.load(src_latent_path)
    src_motion = decode_latent(vae, src_latent, device)   # (T, D)

    # 反归一化（如果提供了 mean/std）
    if mean is not None and std is not None:
        src_motion = inv_transform(src_motion, mean, std)

    # 保存 motion
    if output_mode in ("motion", "both"):
        src_motion_name = src_latent_path.name
        src_motion_path = motion_dir / "source" / src_motion_name
        src_motion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(src_motion_path, src_motion)
        new_pair["source_path"] = str(src_motion_path)

    # 保存 position（FK）
    if output_mode in ("position", "both"):
        if skeleton is None:
            raise ValueError("Skeleton is required for output_mode='position' or 'both'")
        src_pos = latent_to_positions(src_motion, skeleton, joints_num, device)
        src_pos_name = src_latent_path.stem + "_pos.npy"
        src_pos_path = position_dir / "source" / src_pos_name
        src_pos_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(src_pos_path, src_pos)

        if output_mode == "position":
            new_pair["source_path"] = str(src_pos_path)
        else:
            new_pair["source_position_path"] = str(src_pos_path)

    # ---------- 处理 Edited ----------
    tgt_latent_path = Path(pair["edited_path"])
    tgt_latent_path = replace_top3(tgt_latent_path, data_root)

    if not tgt_latent_path.exists():
        raise FileNotFoundError(
            f"Edited latent not found: {tgt_latent_path} "
            f"(original: {pair.get('edited_path', 'N/A')})"
        )

    tgt_latent = np.load(tgt_latent_path)
    tgt_motion = decode_latent(vae, tgt_latent, device)

    if mean is not None and std is not None:
        tgt_motion = inv_transform(tgt_motion, mean, std)

    if output_mode in ("motion", "both"):
        tgt_motion_name = tgt_latent_path.name
        tgt_motion_path = motion_dir / "edited" / tgt_motion_name
        tgt_motion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tgt_motion_path, tgt_motion)
        new_pair["edited_path"] = str(tgt_motion_path)

    if output_mode in ("position", "both"):
        if skeleton is None:
            raise ValueError("Skeleton is required for output_mode='position' or 'both'")
        tgt_pos = latent_to_positions(tgt_motion, skeleton, joints_num, device)
        tgt_pos_name = tgt_latent_path.stem + "_pos.npy"
        tgt_pos_path = position_dir / "edited" / tgt_pos_name
        tgt_pos_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tgt_pos_path, tgt_pos)

        if output_mode == "position":
            new_pair["edited_path"] = str(tgt_pos_path)
        else:
            new_pair["edited_position_path"] = str(tgt_pos_path)

    return new_pair


def main():
    parser = argparse.ArgumentParser(description="Decode edit dataset latents back to motion/position")
    parser.add_argument("--input_json", required=True, help="输入的 JSON 文件路径（如 accepted_secondary.json）")
    parser.add_argument("--out_dir", required=True, help="输出根目录")
    parser.add_argument("--vae_yaml", required=True, help="VAE 配置文件路径（如 vae_detail.yaml）")
    parser.add_argument("--vae_ckpt", required=True, help="VAE checkpoint 路径")
    parser.add_argument("--mean_path", default=None, help="mean.npy 路径（用于反归一化）")
    parser.add_argument("--std_path", default=None, help="std.npy 路径（用于反归一化）")
    parser.add_argument("--skeleton_bvh", default=None, help="Skeleton BVH 模板路径（FK 用）")
    parser.add_argument("--joints_num", type=int, default=24, help="关节数量（FK 用，默认 24）")
    parser.add_argument("--output_mode", default="motion", choices=["motion", "position", "both"],
                        help="输出类型：motion=动作表示, position=全局关节位置, both=两者都存")
    parser.add_argument("--data_root", default=None,
                        help="数据根目录替换路径。若原始路径不存在，将前三级目录替换为此路径后重试。")
    parser.add_argument("--device", default="cuda", help="计算设备")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 VAE
    vae, vae_cfg = load_vae(args.vae_yaml, args.vae_ckpt, device)

    # 2. 加载归一化参数（可选）
    mean, std = None, None
    if args.mean_path and args.std_path:
        mean = np.load(args.mean_path)
        std = np.load(args.std_path)
        print(f"[Norm] Loaded mean={mean.shape}, std={std.shape}")

    # 3. 加载 Skeleton（可选，仅 position/both 需要）
    skeleton = None
    if args.output_mode in ("position", "both"):
        if not args.skeleton_bvh:
            raise ValueError("--skeleton_bvh is required when output_mode is 'position' or 'both'")
        skeleton = load_skeleton(args.skeleton_bvh, device)
        print(f"[Skeleton] Loaded from {args.skeleton_bvh}")

    # 4. 读取输入 JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())
    elif not isinstance(data, list):
        raise ValueError("Input JSON must be a list or a dict of records")

    print(f"[JSON] Loaded {len(data)} records from {args.input_json}")

    # 5. 逐条处理
    new_data: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []

    for idx, pair in enumerate(tqdm(data, desc="Decoding")):
        if not isinstance(pair, dict):
            continue

        try:
            new_pair = process_single_pair(
                pair=pair,
                vae=vae,
                vae_cfg=vae_cfg,
                device=device,
                out_dir=out_dir,
                mean=mean,
                std=std,
                skeleton=skeleton,
                joints_num=args.joints_num,
                output_mode=args.output_mode,
                data_root=args.data_root,
            )
            new_data.append(new_pair)
        except Exception as e:
            print(f"\n[Error] Failed at record {idx} (key={pair.get('original_key', 'N/A')}): {e}")
            traceback.print_exc()
            failed_records.append({
                "index": idx,
                "original_key": pair.get("original_key", "unknown"),
                "error": str(e),
                "record": pair,
            })

    # 6. 保存新 JSON
    out_json_path = out_dir / Path(args.input_json).name
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] New JSON saved to {out_json_path}")
    print(f"       Total records: {len(data)} | Success: {len(new_data)} | Failed: {len(failed_records)}")

    # 7. 保存失败日志（如果有）
    if failed_records:
        fail_log_path = out_dir / "failed_records.json"
        with open(fail_log_path, 'w', encoding='utf-8') as f:
            json.dump(failed_records, f, indent=2, ensure_ascii=False)
        print(f"       Failed log saved to {fail_log_path}")


if __name__ == "__main__":
    main()
