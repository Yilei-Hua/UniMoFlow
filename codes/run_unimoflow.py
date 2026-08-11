#!/usr/bin/env python3
"""
Interactive visualization and generation script for unimoflow.

Compared with edit_vis_interactive.py, this version uses
models_flow/unimoflow.py and supports:

- edit: source motion/latent + edit instruction -> edited motion
- t2m: text prompt -> generated motion
- t2m_edit: text prompt -> generated motion -> further edit
- interactive loop with "last" support, so a T2M result can be edited directly

Run from ../codes or from the repository root.
"""

import argparse
import copy
import datetime
import json
import math
import os
import sys
import warnings
from os.path import exists as pexists
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config.load_config import load_config
from common.skeleton import Skeleton
from common.animation import Animation
from einops import repeat
from utils import bvh_io
from utils.motion_process_bvh import recover_pos_from_rot, recover_bvh_from_rot
from models_flow.hrvae import HRVAE
from models_flow.unimoflow import UniMoFlow
from utils.paramUtil import kinematic_chain



def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"yes", "true", "t", "1"}:
        return True
    if value in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def parse_edit_texts(edit_arg: str) -> List[str]:
    if not edit_arg:
        return []
    candidate = Path(edit_arg)
    if candidate.is_file():
        if candidate.suffix.lower() == ".json":
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("edit_texts", payload.get("commands", []))
            if not isinstance(payload, list):
                raise ValueError("Edit JSON must contain a list of commands")
            return [str(item).strip() for item in payload if str(item).strip()]
        return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [part.strip() for part in edit_arg.split("||") if part.strip()]

def code_relative(path: str) -> str:
    if path is None:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return str(SCRIPT_DIR / path)


def maybe_code_relative(path: str) -> str:
    if path is None:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path) or pexists(path):
        return path
    candidate = SCRIPT_DIR / path
    return str(candidate) if candidate.exists() else path


def load_trusted_checkpoint(path: str, map_location="cpu"):
    try:
        safe_globals = [
            np._core.multiarray.scalar,
            np.core.multiarray.scalar,
            np.dtype,
            np.float32,
            np.float64,
            np.int64,
            np.int32,
        ]
        torch.serialization.add_safe_globals(safe_globals)
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        msg = str(exc)
        if "Weights only load failed" not in msg and "Unsupported global" not in msg:
            raise
        print(f"[Checkpoint] Falling back to weights_only=False for trusted local file: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)


def cfg_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def ensure_list_text(text_arg: str) -> List[str]:
    if text_arg is None:
        return []
    return parse_edit_texts(text_arg)


class UniMoFlowInteractiveGenerator:
    def _init_skeleton(self) -> Optional[Skeleton]:
            template_path = pjoin(self.cfg['data']['root_dir'], 'renamed_bvhs', 'm_ep2_00086.bvh')
            try:
                template_anim = bvh_io.load(template_path)
                skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=self.device)
                return skeleton
            except Exception as e:
                print(f"[Warning] Failed to load skeleton: {e}")
                return None

    def _motion_to_global_pos(self, motion) -> np.ndarray:
            """
            将 motion 转为全局 3D 关节位置。兼容 [T,D] 和 [B,T,D]。
            修复：所有 tensor 先 detach() 再转 numpy，避免 requires_grad 报错。
            """
            if self.skeleton is None:
                if isinstance(motion, torch.Tensor):
                    # 关键修复：先 detach()
                    return motion[:, :72].reshape(-1, 24, 3).detach().cpu().numpy()
                return motion[:, :72].reshape(-1, 24, 3)

            if not isinstance(motion, torch.Tensor):
                motion = torch.from_numpy(motion).float().to(self.device)
            else:
                motion = motion.to(self.device)

            if motion.dim() == 2:
                motion = motion.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            try:
                global_pos = recover_pos_from_rot(motion, joints_num=24, skeleton=self.skeleton)
                if squeeze_output and global_pos.shape[0] == 1:
                    global_pos = global_pos.squeeze(0)
                # 关键修复：先 detach()
                return global_pos.detach().cpu().numpy()
            except Exception as e:
                print(f"[Warning] recover_pos_from_rot failed: {e}, fallback to reshape")
                if motion.dim() == 3:
                    motion = motion.squeeze(0)
                # 关键修复：先 detach()
                return motion[:, :72].reshape(-1, 24, 3).detach().cpu().numpy()

    def _save_gif_comparison(self, source_pos: np.ndarray, edited_pos: np.ndarray,
                                 source_caption: str, target_caption: str, edit_command: str,
                                 output_dir: str, radius: float = 150.0) -> str:
            T = source_pos.shape[0]

            all_joints = np.concatenate([source_pos, edited_pos], axis=0)
            data_mean = all_joints.mean(axis=(0, 1))
            y_min = all_joints[:, :, 1].min()
            y_max = all_joints[:, :, 1].max()
            y_center = (y_min + y_max) / 2

            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(1, 2, wspace=0.1)

            ax_source = fig.add_subplot(gs[0, 0], projection='3d')
            ax_edited = fig.add_subplot(gs[0, 1], projection='3d')

            def draw_skeleton(ax, joints, color='blue', alpha=0.7):
                for chain in kinematic_chain:
                    x = [joints[j, 0] for j in chain]
                    y = [joints[j, 1] for j in chain]
                    z = [joints[j, 2] for j in chain]
                    ax.plot(x, y, z, marker='o', markersize=4,
                            linewidth=2, color=color, alpha=alpha)
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                           c='red', s=20, alpha=0.8)

            def update(frame):
                ax_source.clear()
                ax_edited.clear()

                for ax in [ax_source, ax_edited]:
                    ax.set_xlim3d([data_mean[0] - radius, data_mean[0] + radius])
                    ax.set_ylim3d([max(0, y_center - radius * 0.3), y_center + radius])
                    ax.set_zlim3d([data_mean[2] - radius, data_mean[2] + radius])
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y (Up)')
                    ax.set_zlabel('Z')
                    ax.view_init(elev=20, azim=-60, vertical_axis='y')

                # 关键：仅显示编辑命令，完整文本，不截断
                ax_source.set_title("Source", fontsize=11)
                ax_edited.set_title(f"{edit_command}", fontsize=11)

                draw_skeleton(ax_source, source_pos[frame], 'royalblue')
                draw_skeleton(ax_edited, edited_pos[frame], 'seagreen')

                fig.suptitle(f"Frame {frame}/{T - 1}", fontsize=12)
                return []

            anim = FuncAnimation(fig, update, frames=T, interval=1000 / 30, blit=False)

            gif_path = pjoin(output_dir, "comparison.gif")
            writer = PillowWriter(fps=30)
            anim.save(gif_path, writer=writer)
            plt.close(fig)

            return gif_path

    def _save_bvh(self, motion, output_dir: str, prefix: str) -> Optional[str]:
            try:
                if not isinstance(motion, torch.Tensor):
                    motion = torch.from_numpy(motion).float().to(self.device)
                else:
                    motion = motion.to(self.device)

                if motion.dim() == 2:
                    motion = motion.unsqueeze(0)
                    squeeze_output = True
                else:
                    squeeze_output = False

                result = recover_bvh_from_rot(motion, 24, self.skeleton, keep_shape=False)

                if isinstance(result, tuple):
                    if len(result) == 3:
                        _, local_quats, r_pos = result
                    elif len(result) == 2:
                        local_quats, r_pos = result
                    else:
                        raise ValueError(f"Unexpected return length: {len(result)}")
                else:
                    local_quats = result.local_quats if hasattr(result, 'local_quats') else result[0]
                    r_pos = result.r_pos if hasattr(result, 'r_pos') else result[1]

                if isinstance(local_quats, torch.Tensor):
                    local_quats = local_quats.cpu().detach().numpy()
                if isinstance(r_pos, torch.Tensor):
                    r_pos = r_pos.cpu().detach().numpy()

                if squeeze_output:
                    if local_quats.ndim == 3 and local_quats.shape[0] == 1:
                        local_quats = local_quats[0]
                    if r_pos.ndim == 2 and r_pos.shape[0] == 1:
                        r_pos = r_pos[0]

                template_path = pjoin(self.cfg['data']['root_dir'], 'renamed_bvhs', 'm_ep2_00086.bvh')
                template_anim = bvh_io.load(template_path)

                if r_pos.ndim == 2:
                    r_pos = r_pos[:, None, :]

                r_pos_repeated = repeat(r_pos, 't 1 d -> t k d', k=len(template_anim))

                anim = Animation(
                    local_quats, r_pos_repeated,
                    template_anim.orients, template_anim.offsets,
                    template_anim.parents, template_anim.names,
                    template_anim.frametime
                )

                bvh_path = pjoin(output_dir, f"{prefix}.bvh")
                bvh_io.save(bvh_path, anim, names=anim.names,
                            frametime=anim.frametime, order='xyz', quater=True)

                return bvh_path

            except Exception as e:
                print(f"[Warning] Failed to save BVH for {prefix}: {e}")
                import traceback
                traceback.print_exc()
                return None

    def __init__(self, cfg, device: torch.device, which_epoch: str):
        self.cfg = cfg
        self.device = device
        self.which_epoch = which_epoch
        self.downsample_ratio = int(cfg.data.get("downsample_ratio", 4))
        self.dim_pose = int(cfg.data.get("dim_pose", 296))

        self.vae = self._load_vae()
        self.edit_model = self._load_unimoflow_model()

        meta_dir = pjoin(cfg.data.root_dir, cfg.data.get("meta_dir", "meta_data"))
        self.mean = np.load(pjoin(meta_dir, "mean.npy"))
        self.std = np.load(pjoin(meta_dir, "std.npy"))
        self.mean_torch = torch.from_numpy(self.mean).float().to(device)
        self.std_torch = torch.from_numpy(self.std).float().to(device)

        self.skeleton = self._init_skeleton()
        self.last_result: Optional[Dict[str, Any]] = None

    def _load_vae(self) -> HRVAE:
        vae_cfg_path = code_relative(
            self.cfg.vae_config
        )
        vae_cfg = load_config(vae_cfg_path)

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
        ckpt_path = code_relative(self.cfg.vae_checkpoint)
        ckpt = load_trusted_checkpoint(ckpt_path, map_location="cpu")
        model_key = "vq_model" if "vq_model" in ckpt else "model"
        vae.load_state_dict(ckpt[model_key])
        vae.to(self.device)
        vae.eval()
        print(f"[VAE] Loaded from {ckpt_path}")
        return vae

    def _load_unimoflow_model(self) -> UniMoFlow:
        m_cfg = self.cfg.model
        param_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        model = UniMoFlow(
            checkpoint_path=code_relative(m_cfg.checkpoint_path),
            tokenizer_path=code_relative(m_cfg.tokenizer_path),
            input_dim=m_cfg.input_dim,
            hidden_dim=m_cfg.hidden_dim,
            ffn_dim=m_cfg.ffn_dim,
            num_layers=m_cfg.num_layers,
            num_heads=m_cfg.num_heads,
            num_registers=m_cfg.get("num_registers", 0),
            text_dim=m_cfg.text_dim,
            text_len=m_cfg.text_len,
            dropout_prob=0.0,
            noise_steps=m_cfg.noise_steps,
            cfg_scale=m_cfg.cfg_scale,
            time_scale=m_cfg.get("time_scale", 10.0),
            prediction_type=m_cfg.prediction_type,
            use_logit_normal=m_cfg.get("use_logit_normal", False),
            logit_mean=m_cfg.get("logit_mean", 0.0),
            logit_std=m_cfg.get("logit_std", 1.0),
            fusion_schedule=m_cfg.get("fusion_schedule", "asymmetric"),
            param_dtype=param_dtype,
            spatial_dim=m_cfg.get("spatial_dim", 1),
            use_dynamic_depth=m_cfg.get("use_dynamic_depth", False),
            gen_loss_weight=self.cfg.training.get("gen_loss_weight", 1.0) if "training" in self.cfg else 1.0,
            edit_loss_weight=self.cfg.training.get("edit_loss_weight", 1.0) if "training" in self.cfg else 1.0,
        )

        explicit_path = maybe_code_relative(self.which_epoch)
        if os.path.isfile(explicit_path):
            ckpt_path = explicit_path
        else:
            ckpt_path = maybe_code_relative(
                pjoin(self.cfg.exp.checkpoint_dir, "model", self.which_epoch)
            )
        ckpt = load_trusted_checkpoint(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[UniMoFlow] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[UniMoFlow] Unexpected keys: {len(unexpected)}")
        model.to(self.device)
        model.eval()
        print(f"[UniMoFlow] Loaded from {ckpt_path}")
        return model

    def _latent_len_from_frames(self, length_frames: int) -> int:
        return max(1, int(math.ceil(length_frames / float(self.downsample_ratio))))

    def _frames_from_latent_len(self, latent_len: int) -> int:
        return int(latent_len) * self.downsample_ratio

    def load_motion_from_file(
        self,
        motion_path: str,
        is_latent: bool = False,
        specified_length: Optional[int] = None,
        normalized_motion: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if not pexists(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")

        data = np.load(motion_path)
        motion = torch.from_numpy(data).float().to(self.device)
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)
        elif motion.dim() != 3:
            raise ValueError(f"Motion must be [T,D] or [B,T,D], got {data.shape}")

        B, actual_T, D = motion.shape
        if is_latent:
            if D != int(self.cfg.model.input_dim):
                raise ValueError(f"Latent dim mismatch: file has {D}, model expects {self.cfg.model.input_dim}")
            orig_len = specified_length if specified_length is not None else self._frames_from_latent_len(actual_T)
            latent_len = min(actual_T, self._latent_len_from_frames(orig_len))
            latent = motion[:, :latent_len]
            orig_lengths = [min(orig_len, self._frames_from_latent_len(latent_len))] * B
            with torch.no_grad():
                raw_motion = self.vae.decode(latent)[:, : max(orig_lengths)]
        else:
            if D != self.dim_pose:
                raise ValueError(f"Raw motion dim mismatch: file has {D}, expected {self.dim_pose}")
            orig_len = min(specified_length or actual_T, actual_T)
            motion = motion[:, :orig_len]
            motion_norm = motion if normalized_motion else (motion - self.mean_torch) / self.std_torch
            with torch.no_grad():
                enc_out = self.vae.encode(motion_norm)
                latent = enc_out[0] if isinstance(enc_out, tuple) else enc_out
                raw_motion = self.vae.decode(latent)
            latent_len = latent.shape[1]
            orig_lengths = [orig_len] * B

        latent_lengths = torch.full((B,), latent_len, dtype=torch.long, device=self.device)
        return latent, latent_lengths, raw_motion, orig_lengths

    @torch.no_grad()
    def run_edit(
        self,
        latent: torch.Tensor,
        latent_lengths: torch.Tensor,
        edit_texts: List[str],
        cfg_scale: Optional[float] = None,
        steps: Optional[int] = None,
        use_flowedit: bool = False,
    ) -> Dict[str, Any]:
        x_in = {"source": latent, "edit_text": edit_texts, "length": latent_lengths}
        actual_steps = steps or self.cfg.model.get("eval_noise_steps", self.cfg.model.noise_steps)
        actual_cfg = cfg_scale if cfg_scale is not None else self.cfg.model.cfg_scale
        if use_flowedit:
            return self.edit_model.flow_edit(x_in, num_denoise_steps=actual_steps, cfg_scale=actual_cfg)
        return self.edit_model.generate_edit(x_in, num_denoise_steps=actual_steps, cfg_scale=actual_cfg)

    @torch.no_grad()
    def run_t2m(
        self,
        texts: List[str],
        length_frames: int,
        cfg_scale: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        latent_len = self._latent_len_from_frames(length_frames)
        latent_lengths = torch.full((len(texts),), latent_len, dtype=torch.long, device=self.device)
        actual_steps = steps or self.cfg.model.get("eval_noise_steps", self.cfg.model.noise_steps)
        actual_cfg = cfg_scale if cfg_scale is not None else self.cfg.model.cfg_scale
        out = self.edit_model.generate_gen(
            {"text": texts, "length": latent_lengths},
            num_denoise_steps=actual_steps,
            cfg_scale=actual_cfg,
        )
        out["length"] = latent_lengths
        out["orig_lengths"] = [length_frames] * len(texts)
        return out

    def _denorm(self, motion_norm: torch.Tensor) -> torch.Tensor:
        return motion_norm * self.std_torch + self.mean_torch

    def _save_gif_single(self, joints: np.ndarray, caption: str, output_dir: str, radius: float = 150.0) -> str:
        T = joints.shape[0]
        data_mean = joints.mean(axis=(0, 1))
        y_min = joints[:, :, 1].min()
        y_max = joints[:, :, 1].max()
        y_center = (y_min + y_max) / 2

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        def draw_skeleton(frame_joints):
            for chain in kinematic_chain:
                x = [frame_joints[j, 0] for j in chain]
                y = [frame_joints[j, 1] for j in chain]
                z = [frame_joints[j, 2] for j in chain]
                ax.plot(x, y, z, marker="o", markersize=4, linewidth=2, color="seagreen", alpha=0.75)
            ax.scatter(frame_joints[:, 0], frame_joints[:, 1], frame_joints[:, 2], c="red", s=20, alpha=0.8)

        def update(frame):
            ax.clear()
            ax.set_xlim3d([data_mean[0] - radius, data_mean[0] + radius])
            ax.set_ylim3d([max(0, y_center - radius * 0.3), y_center + radius])
            ax.set_zlim3d([data_mean[2] - radius, data_mean[2] + radius])
            ax.set_xlabel("X")
            ax.set_ylabel("Y (Up)")
            ax.set_zlabel("Z")
            ax.view_init(elev=20, azim=-60, vertical_axis="y")
            ax.set_title(caption, fontsize=11)
            draw_skeleton(joints[frame])
            fig.suptitle(f"Frame {frame}/{T - 1}", fontsize=12)
            return []

        gif_path = pjoin(output_dir, "generated.gif")
        anim = FuncAnimation(fig, update, frames=T, interval=1000 / 30, blit=False)
        anim.save(gif_path, writer=PillowWriter(fps=30))
        plt.close(fig)
        return gif_path

    def process_t2m(
        self,
        texts: List[str],
        length_frames: int,
        output_dir: str,
        save_gif: bool = True,
        save_bvh: bool = True,
        radius: float = 150.0,
        cfg_scale: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        gen_out = self.run_t2m(texts, length_frames, cfg_scale=cfg_scale, steps=steps)
        latents = gen_out["generated"].float()
        motions_norm = self.vae.decode(latents).float()
        orig_lengths = gen_out["orig_lengths"]

        results = []
        for i, text in enumerate(texts):
            length = min(orig_lengths[i], motions_norm.shape[1])
            sample_dir = pjoin(output_dir, f"sample_{i:03d}")
            os.makedirs(sample_dir, exist_ok=True)

            latent_path = pjoin(sample_dir, "generated_latent.npy")
            motion_norm_path = pjoin(sample_dir, "generated_motion_norm.npy")
            motion_denorm_path = pjoin(sample_dir, "generated_motion_denorm.npy")

            motion_norm = motions_norm[i, :length]
            motion_denorm = self._denorm(motion_norm)
            np.save(latent_path, latents[i].detach().cpu().numpy())
            np.save(motion_norm_path, motion_norm.detach().cpu().numpy())
            np.save(motion_denorm_path, motion_denorm.detach().cpu().numpy())

            gif_path = None
            if save_gif:
                joints = self._motion_to_global_pos(motion_denorm.detach().cpu())
                gif_path = self._save_gif_single(joints, text, sample_dir, radius)

            bvh_path = None
            if save_bvh and self.skeleton is not None:
                bvh_path = self._save_bvh(motion_denorm, sample_dir, "generated")

            result = {
                "index": i,
                "text": text,
                "length": int(length),
                "latent_path": latent_path,
                "motion_norm_path": motion_norm_path,
                "motion_denorm_path": motion_denorm_path,
                "gif_path": gif_path,
                "bvh_path": bvh_path,
                "sample_dir": sample_dir,
            }
            results.append(result)

        metadata_path = pjoin(output_dir, "interactive_t2m_results.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": "t2m",
                    "config": {
                        "model_name": self.cfg.exp.name,
                        "checkpoint": self.which_epoch,
                        "length_frames": length_frames,
                        "steps": steps or self.cfg.model.get("eval_noise_steps", self.cfg.model.noise_steps),
                        "cfg_scale": cfg_scale if cfg_scale is not None else self.cfg.model.cfg_scale,
                    },
                    "results": results,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        if results:
            self.last_result = results[-1]
        print(f"\n[Done] T2M results saved to: {output_dir}")
        print(f"[Done] Metadata saved to: {metadata_path}")
        return results

    def process_edit(
        self,
        motion_path: str,
        edit_texts: List[str],
        output_dir: str,
        is_latent: bool = False,
        specified_length: Optional[int] = None,
        normalized_motion: bool = False,
        save_gif: bool = True,
        save_bvh: bool = True,
        radius: float = 150.0,
        cfg_scale: Optional[float] = None,
        steps: Optional[int] = None,
        use_flowedit: bool = False,
    ) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        latent, latent_lengths, raw_motions, orig_lengths = self.load_motion_from_file(
            motion_path,
            is_latent=is_latent,
            specified_length=specified_length,
            normalized_motion=normalized_motion,
        )
        B = latent.shape[0]
        if len(edit_texts) == 1 and B > 1:
            edit_texts = edit_texts * B
        elif len(edit_texts) != B:
            raise ValueError(f"Need 1 edit command or exactly {B}; got {len(edit_texts)}")

        edit_out = self.run_edit(
            latent,
            latent_lengths,
            edit_texts,
            cfg_scale=cfg_scale,
            steps=steps,
            use_flowedit=use_flowedit,
        )
        edited_latents = edit_out["generated"].float()
        edited_motions = self.vae.decode(edited_latents).float()

        results = []
        for i in range(B):
            length = min(orig_lengths[i], edited_motions.shape[1], raw_motions.shape[1])
            sample_dir = pjoin(output_dir, f"sample_{i:03d}")
            os.makedirs(sample_dir, exist_ok=True)

            source_norm = raw_motions[i, :length]
            edited_norm = edited_motions[i, :length]
            source_denorm = self._denorm(source_norm)
            edited_denorm = self._denorm(edited_norm)

            source_latent_path = pjoin(sample_dir, "source_latent.npy")
            edited_latent_path = pjoin(sample_dir, "edited_latent.npy")
            source_motion_path = pjoin(sample_dir, "source_motion_denorm.npy")
            edited_motion_path = pjoin(sample_dir, "edited_motion_denorm.npy")
            edited_motion_norm_path = pjoin(sample_dir, "edited_motion_norm.npy")

            np.save(source_latent_path, latent[i].detach().cpu().numpy())
            np.save(edited_latent_path, edited_latents[i].detach().cpu().numpy())
            np.save(source_motion_path, source_denorm.detach().cpu().numpy())
            np.save(edited_motion_path, edited_denorm.detach().cpu().numpy())
            np.save(edited_motion_norm_path, edited_norm.detach().cpu().numpy())

            gif_path = None
            if save_gif:
                source_pos = self._motion_to_global_pos(source_denorm.detach().cpu())
                edited_pos = self._motion_to_global_pos(edited_denorm.detach().cpu())
                gif_path = self._save_gif_comparison(
                    source_pos,
                    edited_pos,
                    "",
                    "",
                    edit_texts[i],
                    sample_dir,
                    radius,
                )

            source_bvh = None
            edited_bvh = None
            if save_bvh and self.skeleton is not None:
                source_bvh = self._save_bvh(source_denorm, sample_dir, "source")
                edited_bvh = self._save_bvh(edited_denorm, sample_dir, "edited")

            result = {
                "index": i,
                "edit_command": edit_texts[i],
                "length": int(length),
                "source_latent_path": source_latent_path,
                "edited_latent_path": edited_latent_path,
                "source_motion_denorm_path": source_motion_path,
                "edited_motion_denorm_path": edited_motion_path,
                "edited_motion_norm_path": edited_motion_norm_path,
                "gif_path": gif_path,
                "source_bvh": source_bvh,
                "edited_bvh": edited_bvh,
                "sample_dir": sample_dir,
            }
            results.append(result)

        metadata_path = pjoin(output_dir, "interactive_edit_results.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": "edit",
                    "config": {
                        "model_name": self.cfg.exp.name,
                        "checkpoint": self.which_epoch,
                        "motion_file": motion_path,
                        "is_latent": is_latent,
                        "normalized_motion": normalized_motion,
                        "steps": steps or self.cfg.model.get("eval_noise_steps", self.cfg.model.noise_steps),
                        "cfg_scale": cfg_scale if cfg_scale is not None else self.cfg.model.cfg_scale,
                        "use_flowedit": use_flowedit,
                    },
                    "results": results,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        if results:
            self.last_result = {
                "latent_path": results[-1]["edited_latent_path"],
                "motion_denorm_path": results[-1]["edited_motion_denorm_path"],
                "length": results[-1]["length"],
                "sample_dir": results[-1]["sample_dir"],
            }
        print(f"\n[Done] Edit results saved to: {output_dir}")
        print(f"[Done] Metadata saved to: {metadata_path}")
        return results


def timestamp_dir(root: str, prefix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return pjoin(root, f"{prefix}_{ts}")


def read_optional_float(prompt: str, default: Optional[float]):
    raw = input(prompt).strip()
    if raw.lower() in ("q", "quit"):
        raise KeyboardInterrupt
    if not raw:
        return default
    return float(raw)


def read_optional_int(prompt: str, default: Optional[int]):
    raw = input(prompt).strip()
    if raw.lower() in ("q", "quit"):
        raise KeyboardInterrupt
    if not raw:
        return default
    return int(raw)


def interactive_loop(generator: UniMoFlowInteractiveGenerator, args):
    print("\n" + "=" * 72)
    print("UniMoFlow interactive motion tool is ready.")
    print("Modes: t2m/g, edit/e, last/l, t2m_edit/ge, quit/q")
    print("Tip: T2M writes generated_latent.npy; use last/l or that latent path for editing.")
    print("=" * 72)

    round_id = 0
    while True:
        round_id += 1
        try:
            mode = input("\nMode [t2m(g) / edit(e) / last(l) / t2m_edit(ge) / quit(q)]: ").strip().lower()
            if mode in ("q", "quit"):
                break

            out_root = args.output_dir
            if mode in ("t2m", "g"):
                text = input("T2M text prompt (or .txt): ").strip()
                texts = ensure_list_text(text)
                length = read_optional_int(f"Motion length in frames (Enter={args.length}): ", args.length)
                cfg_scale = read_optional_float(f"T2M CFG scale (Enter={args.cfg_scale}): ", args.cfg_scale)
                steps = read_optional_int(f"T2M steps (Enter={args.steps}): ", args.steps)
                out_dir = input("Output dir (Enter=auto): ").strip() or timestamp_dir(out_root, f"t2m_{round_id:03d}")
                generator.process_t2m(texts, length, out_dir, args.save_gif, args.save_bvh, args.radius, cfg_scale, steps)

            elif mode in ("edit", "e", "last", "l"):
                if mode in ("last", "l"):
                    if not generator.last_result:
                        print("[Error] No last T2M/edit result available.")
                        continue
                    motion_file = generator.last_result["latent_path"]
                    is_latent = True
                    length = int(generator.last_result.get("length") or args.length)
                    normalized_motion = False
                    print(f"[Last] Editing latent: {motion_file}")
                else:
                    motion_file = input("Motion/latent .npy path: ").strip()
                    is_latent = input("Is latent? [y/N]: ").strip().lower() in ("y", "yes", "1", "true")
                    normalized_motion = False
                    if not is_latent:
                        normalized_motion = input("Raw motion is already normalized? [y/N]: ").strip().lower() in ("y", "yes", "1", "true")
                    length = read_optional_int(f"Original length in frames (Enter={args.length}/auto): ", args.length)
                edit_text = input("Edit instruction (or .txt): ").strip()
                edit_texts = ensure_list_text(edit_text)
                cfg_scale = read_optional_float(f"Edit CFG scale (Enter={args.cfg_scale}): ", args.cfg_scale)
                steps = read_optional_int(f"Edit steps (Enter={args.steps}): ", args.steps)
                out_dir = input("Output dir (Enter=auto): ").strip() or timestamp_dir(out_root, f"edit_{round_id:03d}")
                generator.process_edit(
                    motion_file,
                    edit_texts,
                    out_dir,
                    is_latent=is_latent,
                    specified_length=length,
                    normalized_motion=normalized_motion,
                    save_gif=args.save_gif,
                    save_bvh=args.save_bvh,
                    radius=args.radius,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    use_flowedit=args.self_flowedit,
                )

            elif mode in ("t2m_edit", "ge"):
                text = input("T2M text prompt (or .txt): ").strip()
                texts = ensure_list_text(text)
                length = read_optional_int(f"Motion length in frames (Enter={args.length}): ", args.length)
                t2m_cfg = read_optional_float(f"T2M CFG scale (Enter={args.cfg_scale}): ", args.cfg_scale)
                steps = read_optional_int(f"Steps (Enter={args.steps}): ", args.steps)
                edit_text = input("Edit instruction after T2M: ").strip()
                edit_texts = ensure_list_text(edit_text)
                out_dir = input("Output dir (Enter=auto): ").strip() or timestamp_dir(out_root, f"t2m_edit_{round_id:03d}")
                t2m_results = generator.process_t2m(
                    texts,
                    length,
                    pjoin(out_dir, "t2m"),
                    args.save_gif,
                    args.save_bvh,
                    args.radius,
                    t2m_cfg,
                    steps,
                )
                if t2m_results:
                    generator.process_edit(
                        t2m_results[0]["latent_path"],
                        edit_texts,
                        pjoin(out_dir, "edit"),
                        is_latent=True,
                        specified_length=length,
                        save_gif=args.save_gif,
                        save_bvh=args.save_bvh,
                        radius=args.radius,
                        cfg_scale=args.cfg_scale,
                        steps=steps,
                        use_flowedit=args.self_flowedit,
                    )
            else:
                print("[Hint] Unknown mode.")
        except KeyboardInterrupt:
            print("\nExit.")
            break
        except Exception as exc:
            print(f"[Error] {exc}")
            import traceback

            traceback.print_exc()


def build_cfg(args):
    config_path = maybe_code_relative(args.config)
    cfg = load_config(config_path)
    cfg.exp.checkpoint_dir = code_relative(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "unimoflow", cfg.exp.name))
    if args.root_dir:
        cfg.data.root_dir = args.root_dir
    if args.cfg_scale is not None:
        cfg.model.cfg_scale = args.cfg_scale
    if args.steps is not None:
        cfg.model.eval_noise_steps = args.steps
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive generation and editing with UniMoFlow")
    parser.add_argument("--config", type=str, default="../configs/unimoflow.yaml")
    parser.add_argument("--which_epoch", type=str, default="latest.tar")
    parser.add_argument("--mode", choices=["interactive", "edit", "t2m", "t2m_edit"], default="interactive")
    parser.add_argument("--text", type=str, default=None, help="T2M text prompt or .txt file")
    parser.add_argument("--edit_text", type=str, default=None, help="Edit command or .txt file")
    parser.add_argument("--motion_file", type=str, default=None, help="Input .npy for edit mode")
    parser.add_argument("--is_latent", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--normalized_motion", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--length", type=int, default=196, help="Original motion length in frames")
    parser.add_argument("--output_dir", type=str, default="../outputs/unimoflow_interactive")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--self_flowedit", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_gif", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--save_bvh", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args)
    device_name = args.device or cfg.exp.get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {maybe_code_relative(args.config)}")
    print(f"Checkpoint: {args.which_epoch}")

    generator = UniMoFlowInteractiveGenerator(cfg, device, args.which_epoch)

    if args.mode == "interactive":
        interactive_loop(generator, args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "t2m":
        if not args.text:
            raise ValueError("--text is required for --mode t2m")
        generator.process_t2m(
            ensure_list_text(args.text),
            args.length,
            args.output_dir,
            args.save_gif,
            args.save_bvh,
            args.radius,
            args.cfg_scale,
            args.steps,
        )
    elif args.mode == "edit":
        if not args.motion_file or not args.edit_text:
            raise ValueError("--motion_file and --edit_text are required for --mode edit")
        generator.process_edit(
            args.motion_file,
            ensure_list_text(args.edit_text),
            args.output_dir,
            is_latent=args.is_latent,
            specified_length=args.length,
            normalized_motion=args.normalized_motion,
            save_gif=args.save_gif,
            save_bvh=args.save_bvh,
            radius=args.radius,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            use_flowedit=args.self_flowedit,
        )
    elif args.mode == "t2m_edit":
        if not args.text or not args.edit_text:
            raise ValueError("--text and --edit_text are required for --mode t2m_edit")
        t2m_results = generator.process_t2m(
            ensure_list_text(args.text),
            args.length,
            pjoin(args.output_dir, "t2m"),
            args.save_gif,
            args.save_bvh,
            args.radius,
            args.cfg_scale,
            args.steps,
        )
        if t2m_results:
            generator.process_edit(
                t2m_results[0]["latent_path"],
                ensure_list_text(args.edit_text),
                pjoin(args.output_dir, "edit"),
                is_latent=True,
                specified_length=args.length,
                save_gif=args.save_gif,
                save_bvh=args.save_bvh,
                radius=args.radius,
                cfg_scale=args.cfg_scale,
                steps=args.steps,
                use_flowedit=args.self_flowedit,
            )


if __name__ == "__main__":
    main()
