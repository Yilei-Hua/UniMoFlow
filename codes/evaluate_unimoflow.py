import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from models_flow.unimoflow import UniMoFlow
from models_flow.hrvae import HRVAE
from config.load_config import load_config
from dataset.edit_dataset import MotionEditDataset, collate_fn
from dataset.dataset import TextMotionDataset
import utils.eval_t2m as eval_t2m
from utils.fixseeds import fixseed
from utils.metrics import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_frechet_distance,
    cosine_similarity_matrix,
)

import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path


def _register_torch_load_safe_globals():
    """Allow numpy scalar objects commonly stored in locally trained checkpoints."""
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


def load_trusted_checkpoint(path, map_location='cpu'):
    """Load a local SnapMoGen checkpoint across PyTorch 2.6 weights_only changes."""
    _register_torch_load_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        msg = str(e)
        if 'Weights only load failed' not in msg and 'Unsupported global' not in msg:
            raise
        print(
            f"[Checkpoint Load] Safe weights_only load failed for {path}; "
            "falling back to weights_only=False for this trusted local checkpoint."
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def calculate_r_precision_top3(embedding1, embedding2):
    top_k = min(3, embedding1.shape[0], embedding2.shape[0])
    if top_k <= 0:
        return np.zeros((embedding1.shape[0], 3), dtype=np.float32)
    scores = calculate_R_precision(embedding1, embedding2, top_k=top_k, sum_all=False)
    if scores.shape[1] == 3:
        return scores
    padded = np.zeros((scores.shape[0], 3), dtype=scores.dtype)
    padded[:, :scores.shape[1]] = scores
    return padded


def load_vae(cfg, device):
    vae_cfg = load_config(cfg.vae_config)

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
    ckpt = load_trusted_checkpoint(ckpt_path, map_location=device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vae.load_state_dict(ckpt[model_key])

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f'Loading VAE Model {vae_cfg.exp.name} from epoch {ckpt["ep"]}')
    vae.to(device)
    vae.eval()
    return vae


def load_unimoflow_model(cfg, device):
    param_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs = dict(
        checkpoint_path=cfg.model.checkpoint_path,
        tokenizer_path=cfg.model.tokenizer_path,
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        ffn_dim=cfg.model.ffn_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_registers=cfg.model.get('num_registers', 0),
        text_dim=cfg.model.text_dim,
        text_len=cfg.model.text_len,
        dropout_prob=0.0,
        noise_steps=cfg.model.noise_steps,
        cfg_scale=cfg.model.cfg_scale,
        time_scale=cfg.model.get('time_scale', 1.0),
        prediction_type=cfg.model.prediction_type,
        use_logit_normal=cfg.model.get('use_logit_normal', False),
        fusion_schedule=cfg.model.get('fusion_schedule', 'asymmetric'),
        param_dtype=param_dtype,
        use_role_tags=True,
        spatial_dim=cfg.model.get('spatial_dim', 1),
        use_dynamic_depth=cfg.model.get('use_dynamic_depth', False),
        gen_loss_weight=1.0,
        edit_loss_weight=1.0,
    )
    model = UniMoFlow(**model_kwargs)

    explicit_path = Path(cfg.exp.which_epoch)
    if not explicit_path.is_absolute():
        explicit_path = (Path(__file__).resolve().parent / explicit_path).resolve()
    if explicit_path.is_file():
        ckpt_path = str(explicit_path)
    else:
        ckpt_path = pjoin(cfg.exp.checkpoint_dir, 'model', cfg.exp.which_epoch)
    ckpt = load_trusted_checkpoint(ckpt_path, map_location='cpu')

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    load_result = model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f'Loading UniMoFlow ({"UniMoFlow"}) from {cfg.exp.which_epoch}, epoch {ckpt.get("ep", "unknown")}')
        missing = getattr(load_result, 'missing_keys', [])
        unexpected = getattr(load_result, 'unexpected_keys', [])
        print(f'[Checkpoint Load] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}')
        if missing:
            print(f'[Checkpoint Load] missing sample: {missing[:20]}')
        if unexpected:
            print(f'[Checkpoint Load] unexpected sample: {unexpected[:20]}')
    return model


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    return local_rank, world_size, device


@torch.no_grad()
def compute_edit_region_alignment(eval_wrapper, source_motions, edited_motions,
                                   edit_commands, orig_lengths,
                                   threshold_percentile=75.0, min_seg_len=8):
    B, T, D = source_motions.shape
    device = source_motions.device

    et_edit, _ = eval_wrapper.encode_text(edit_commands, sample_mean=True)

    delta = edited_motions - source_motions
    delta_norm = torch.norm(delta, dim=-1)

    region_scores = []
    region_coverages = []
    peak_delta_ratios = []

    for i in range(B):
        actual_len = orig_lengths[i].item()
        if actual_len <= 1:
            region_scores.append(0.0)
            region_coverages.append(0.0)
            peak_delta_ratios.append(0.0)
            continue

        d = delta_norm[i, :actual_len].cpu().numpy()
        threshold = np.percentile(d, threshold_percentile)
        sig_mask = d > threshold
        coverage = float(sig_mask.sum()) / actual_len
        region_coverages.append(coverage)
        peak_delta_ratios.append(float(d.max() / (d.mean() + 1e-8)))

        if sig_mask.sum() == 0:
            region_scores.append(0.0)
            continue

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

        best_seg = max(segments, key=lambda s: s[1] - s[0])
        seg_start, seg_end = best_seg

        if seg_end - seg_start < min_seg_len:
            pad = (min_seg_len - (seg_end - seg_start) + 1) // 2
            seg_start = max(0, seg_start - pad)
            seg_end = min(actual_len, seg_end + pad)
            if seg_end - seg_start < min_seg_len:
                seg_end = min(actual_len, seg_start + min_seg_len)

        seg_motion = edited_motions[i, seg_start:seg_end].unsqueeze(0)
        seg_len = torch.tensor([seg_end - seg_start], device=device)

        _, em_seg, _ = eval_wrapper.encode_motion(
            seg_motion[..., :148], seg_len, sample_mean=True)

        cos_sim = F.cosine_similarity(em_seg, et_edit[i:i+1], dim=-1)
        region_scores.append(float(cos_sim.item()))

    return {
        'region_alignment_scores': np.array(region_scores),
        'region_coverages': np.array(region_coverages),
        'peak_delta_ratios': np.array(peak_delta_ratios),
    }


@torch.no_grad()
def evaluate_edit_test(unimoflow_model, vae, eval_wrapper, dataloader, cfg, device,
                       is_main_process=True, self_flowedit=True):
    unimoflow_model.eval()
    vae.eval()
    eval_wrapper.eval()

    num_steps = cfg.model.get('eval_noise_steps', cfg.model.noise_steps)
    cfg_scale_val = cfg.model.cfg_scale
    downsample_ratio = cfg.data.get('downsample_ratio', 4)

    mode_name = 'flowedit' if self_flowedit else 'regular'
    all_matching_scores = []
    all_structure_scores = []
    all_r_precision = []
    all_gt_r_precision = []
    all_gt_matching_scores = []
    all_src_r_precision = []
    all_src_matching_scores = []
    all_r_precision_gen_vs_gt = []
    all_r_precision_gen_vs_src = []
    all_edit_strengths = []

    all_source_target_matches = []
    all_edited_target_matches = []
    all_relative_improvements = []
    all_improvement_ratios = []
    all_region_alignment_scores = []
    all_region_coverages = []
    all_peak_delta_ratios = []
    all_cycle_source_matching_scores = []
    all_cycle_source_r_precision = []
    all_cycle_motion_vs_source_r_precision = []
    all_cycle_structure_scores = []
    all_cycle_forward_gates = []
    all_cycle_consistency_scores = []
    all_em_edited = []
    all_em_target = []

    if is_main_process:
        print(f'\n{"=" * 60}')
        print(f'Edit Evaluation [{mode_name}]')
        print(f'Steps={num_steps}, CFG={cfg_scale_val}')
        print(f'{"=" * 60}')

    iterator = tqdm(dataloader, desc=f"Edit Eval [{mode_name}]") if is_main_process else dataloader

    for batch in iterator:
        if not batch:
            continue
        try:
            source_latents = batch['source'].to(device)
            target_latents = batch['target'].to(device)
            latent_lengths = batch['length'].to(device)
            target_captions = batch['target_caption']
            source_captions = batch['source_caption']
            edit_commands = batch['edit_text']
            reverse_edit_commands = batch.get(
                'reverse_edit_text',
                ['Restore the original source motion'] * len(edit_commands),
            )

            B = source_latents.shape[0]

            x_in = {
                "source": source_latents,
                "edit_text": edit_commands,
                "length": latent_lengths,
            }

            if self_flowedit:
                edit_out = unimoflow_model.flow_edit(
                    x=x_in,
                    num_denoise_steps=num_steps,
                    cfg_scale=cfg_scale_val,
                )
            else:
                edit_out = unimoflow_model.generate(
                    x=x_in,
                    num_denoise_steps=num_steps,
                    cfg_scale=cfg_scale_val,
                )
            edited_latents = edit_out['generated']

            cycle_x_in = {
                "source": edited_latents,
                "edit_text": reverse_edit_commands,
                "length": latent_lengths,
            }
            if self_flowedit:
                cycle_out = unimoflow_model.flow_edit(
                    x=cycle_x_in,
                    num_denoise_steps=num_steps,
                    cfg_scale=cfg_scale_val,
                )
            else:
                cycle_out = unimoflow_model.generate(
                    x=cycle_x_in,
                    num_denoise_steps=num_steps,
                    cfg_scale=cfg_scale_val,
                )
            cycle_latents = cycle_out['generated']

            orig_lengths = latent_lengths * downsample_ratio

            source_motions = vae.decode(source_latents)
            target_motions = vae.decode(target_latents)
            edited_motions = vae.decode(edited_latents)
            cycle_motions = vae.decode(cycle_latents)

            et_target, _ = eval_wrapper.encode_text(target_captions, sample_mean=True)
            et_source, _ = eval_wrapper.encode_text(source_captions, sample_mean=True)
            _, em_edited, _ = eval_wrapper.encode_motion(
                edited_motions[..., :148], orig_lengths, sample_mean=True
            )
            _, em_target, _ = eval_wrapper.encode_motion(
                target_motions[..., :148], orig_lengths, sample_mean=True
            )
            _, em_source, _ = eval_wrapper.encode_motion(
                source_motions[..., :148], orig_lengths, sample_mean=True
            )
            _, em_cycle, _ = eval_wrapper.encode_motion(
                cycle_motions[..., :148], orig_lengths, sample_mean=True
            )

            et_np = et_target.cpu().numpy()
            et_source_np = et_source.cpu().numpy()
            em_edited_np = em_edited.cpu().numpy()
            em_target_np = em_target.cpu().numpy()
            em_source_np = em_source.cpu().numpy()
            em_cycle_np = em_cycle.cpu().numpy()

            r_precision = calculate_r_precision_top3(et_np, em_edited_np)
            match_score = cosine_similarity_matrix(et_np, em_edited_np).diagonal()

            gt_r_precision = calculate_r_precision_top3(et_np, em_target_np)
            gt_match_scores = cosine_similarity_matrix(et_np, em_target_np).diagonal()

            src_r_precision = calculate_r_precision_top3(et_source_np, em_source_np)
            src_match_scores = cosine_similarity_matrix(et_source_np, em_source_np).diagonal()

            cycle_source_r_precision = calculate_r_precision_top3(et_source_np, em_cycle_np)
            cycle_source_match_scores = cosine_similarity_matrix(et_source_np, em_cycle_np).diagonal()
            cycle_motion_vs_source_r_precision = calculate_r_precision_top3(em_cycle_np, em_source_np)

            structure_scores = []
            for i in range(B):
                actual_len = min(
                    int(orig_lengths[i].item()),
                    source_motions.shape[1],
                    edited_motions.shape[1],
                )
                if actual_len <= 0:
                    structure_scores.append(0.0)
                    continue
                src_vec = source_motions[i, :actual_len, :148].flatten()
                edit_vec = edited_motions[i, :actual_len, :148].flatten()
                cos_sim = F.cosine_similarity(src_vec.unsqueeze(0), edit_vec.unsqueeze(0), dim=-1)
                structure_scores.append(cos_sim.item())

            structure_scores = np.array(structure_scores)

            cycle_structure_scores = []
            for i in range(B):
                actual_len = min(
                    int(orig_lengths[i].item()),
                    source_motions.shape[1],
                    cycle_motions.shape[1],
                )
                if actual_len <= 0:
                    cycle_structure_scores.append(0.0)
                    continue
                src_vec = source_motions[i, :actual_len, :148].flatten()
                cycle_vec = cycle_motions[i, :actual_len, :148].flatten()
                cos_sim = F.cosine_similarity(src_vec.unsqueeze(0), cycle_vec.unsqueeze(0), dim=-1)
                cycle_structure_scores.append(cos_sim.item())

            cycle_structure_scores = np.array(cycle_structure_scores)

            r_prec_gen_vs_gt = calculate_r_precision_top3(em_edited_np, em_target_np)
            r_prec_gen_vs_src = calculate_r_precision_top3(em_edited_np, em_source_np)

            edit_strength = match_score - structure_scores

            source_target_match_arr = cosine_similarity_matrix(et_np, em_source_np).diagonal()
            edited_target_match_arr = match_score
            rel_imp = edited_target_match_arr - source_target_match_arr
            imp_ratio = edited_target_match_arr / (source_target_match_arr + 1e-8)

            region_metrics = compute_edit_region_alignment(
                eval_wrapper, source_motions, edited_motions,
                edit_commands, orig_lengths)

            cycle_return_score = (
                0.40 * cycle_structure_scores +
                0.30 * cycle_source_r_precision[:, 0] +
                0.30 * cycle_motion_vs_source_r_precision[:, 0]
            )
            semantic_gate = np.clip(rel_imp / 0.05, 0.0, 1.0)
            change_gate = np.clip((1.0 - structure_scores) / 0.05, 0.0, 1.0)
            cycle_forward_gate = np.sqrt(semantic_gate * change_gate)
            cycle_consistency_score = cycle_return_score * cycle_forward_gate

            all_r_precision.append(r_precision)
            all_matching_scores.extend(match_score.tolist())
            all_gt_r_precision.append(gt_r_precision)
            all_gt_matching_scores.extend(gt_match_scores.tolist())
            all_src_r_precision.append(src_r_precision)
            all_src_matching_scores.extend(src_match_scores.tolist())
            all_r_precision_gen_vs_gt.append(r_prec_gen_vs_gt)
            all_r_precision_gen_vs_src.append(r_prec_gen_vs_src)
            all_structure_scores.extend(structure_scores.tolist())
            all_edit_strengths.extend(edit_strength.tolist())
            all_source_target_matches.extend(source_target_match_arr.tolist())
            all_edited_target_matches.extend(edited_target_match_arr.tolist())
            all_relative_improvements.extend(rel_imp.tolist())
            all_improvement_ratios.extend(imp_ratio.tolist())
            all_region_alignment_scores.extend(region_metrics['region_alignment_scores'].tolist())
            all_region_coverages.extend(region_metrics['region_coverages'].tolist())
            all_peak_delta_ratios.extend(region_metrics['peak_delta_ratios'].tolist())
            all_cycle_source_matching_scores.extend(cycle_source_match_scores.tolist())
            all_cycle_source_r_precision.append(cycle_source_r_precision)
            all_cycle_motion_vs_source_r_precision.append(cycle_motion_vs_source_r_precision)
            all_cycle_structure_scores.extend(cycle_structure_scores.tolist())
            all_cycle_forward_gates.extend(cycle_forward_gate.tolist())
            all_cycle_consistency_scores.extend(cycle_consistency_score.tolist())
            all_em_edited.append(em_edited_np)
            all_em_target.append(em_target_np)

            if is_main_process:
                avg_match = np.mean(all_matching_scores)
                avg_struct = np.mean(all_structure_scores)
                avg_gen_vs_gt = np.mean([r[:, 0].mean() for r in all_r_precision_gen_vs_gt]) if all_r_precision_gen_vs_gt else 0
                iterator.set_postfix({
                    'Match': f'{avg_match:.3f}',
                    'Struct': f'{avg_struct:.3f}',
                    'Gen_vs_GT': f'{avg_gen_vs_gt:.3f}',
                })

        except Exception as e:
            print(f"\n[Error] Edit eval batch failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    all_r_precision = np.concatenate(all_r_precision, axis=0)
    all_gt_r_precision = np.concatenate(all_gt_r_precision, axis=0)
    all_src_r_precision = np.concatenate(all_src_r_precision, axis=0)
    all_r_precision_gen_vs_gt = np.concatenate(all_r_precision_gen_vs_gt, axis=0)
    all_r_precision_gen_vs_src = np.concatenate(all_r_precision_gen_vs_src, axis=0)
    all_cycle_source_r_precision = np.concatenate(all_cycle_source_r_precision, axis=0)
    all_cycle_motion_vs_source_r_precision = np.concatenate(all_cycle_motion_vs_source_r_precision, axis=0)
    all_em_edited = np.concatenate(all_em_edited, axis=0)
    all_em_target = np.concatenate(all_em_target, axis=0)
    mu_edited, cov_edited = calculate_activation_statistics(all_em_edited)
    mu_target, cov_target = calculate_activation_statistics(all_em_target)
    edit_fid = calculate_frechet_distance(mu_target, cov_target, mu_edited, cov_edited)

    prefix = f'edit_{mode_name}_'

    metrics = {
        f'{prefix}r_precision_top1': float(np.mean(all_r_precision[:, 0])),
        f'{prefix}r_precision_top2': float(np.mean(all_r_precision[:, 1])),
        f'{prefix}r_precision_top3': float(np.mean(all_r_precision[:, 2])),
        f'{prefix}matching_score': float(np.mean(all_matching_scores)),
        f'{prefix}fid': float(edit_fid),
        f'{prefix}gt_r_precision_top1': float(np.mean(all_gt_r_precision[:, 0])),
        f'{prefix}gt_r_precision_top2': float(np.mean(all_gt_r_precision[:, 1])),
        f'{prefix}gt_r_precision_top3': float(np.mean(all_gt_r_precision[:, 2])),
        f'{prefix}gt_matching_score': float(np.mean(all_gt_matching_scores)),
        f'{prefix}src_r_precision_top1': float(np.mean(all_src_r_precision[:, 0])),
        f'{prefix}src_r_precision_top2': float(np.mean(all_src_r_precision[:, 1])),
        f'{prefix}src_r_precision_top3': float(np.mean(all_src_r_precision[:, 2])),
        f'{prefix}src_matching_score': float(np.mean(all_src_matching_scores)),
        f'{prefix}structure_preservation': float(np.mean(all_structure_scores)),
        f'{prefix}edit_strength': float(np.mean(all_edit_strengths)),
        f'{prefix}r_precision_gen_vs_gt_top1': float(np.mean(all_r_precision_gen_vs_gt[:, 0])),
        f'{prefix}r_precision_gen_vs_gt_top2': float(np.mean(all_r_precision_gen_vs_gt[:, 1])),
        f'{prefix}r_precision_gen_vs_gt_top3': float(np.mean(all_r_precision_gen_vs_gt[:, 2])),
        f'{prefix}r_precision_gen_vs_src_top1': float(np.mean(all_r_precision_gen_vs_src[:, 0])),
        f'{prefix}r_precision_gen_vs_src_top2': float(np.mean(all_r_precision_gen_vs_src[:, 1])),
        f'{prefix}r_precision_gen_vs_src_top3': float(np.mean(all_r_precision_gen_vs_src[:, 2])),
        f'{prefix}source_target_match': float(np.mean(all_source_target_matches)),
        f'{prefix}edited_target_match': float(np.mean(all_edited_target_matches)),
        f'{prefix}relative_improvement': float(np.mean(all_relative_improvements)),
        f'{prefix}improvement_ratio_mean': float(np.mean(all_improvement_ratios)),
        f'{prefix}positive_improvement_ratio': float(np.mean(np.array(all_relative_improvements) > 0.0)),
        f'{prefix}region_alignment_mean': float(np.mean(all_region_alignment_scores)),
        f'{prefix}region_alignment_std': float(np.std(all_region_alignment_scores)),
        f'{prefix}region_coverage_mean': float(np.mean(all_region_coverages)),
        f'{prefix}peak_delta_ratio_mean': float(np.mean(all_peak_delta_ratios)),
        f'{prefix}cycle_source_matching_score': float(np.mean(all_cycle_source_matching_scores)),
        f'{prefix}cycle_source_r_precision_top1': float(np.mean(all_cycle_source_r_precision[:, 0])),
        f'{prefix}cycle_source_r_precision_top2': float(np.mean(all_cycle_source_r_precision[:, 1])),
        f'{prefix}cycle_source_r_precision_top3': float(np.mean(all_cycle_source_r_precision[:, 2])),
        f'{prefix}cycle_motion_vs_source_r_precision_top1': float(np.mean(all_cycle_motion_vs_source_r_precision[:, 0])),
        f'{prefix}cycle_motion_vs_source_r_precision_top2': float(np.mean(all_cycle_motion_vs_source_r_precision[:, 1])),
        f'{prefix}cycle_motion_vs_source_r_precision_top3': float(np.mean(all_cycle_motion_vs_source_r_precision[:, 2])),
        f'{prefix}cycle_structure_to_source': float(np.mean(all_cycle_structure_scores)),
        f'{prefix}cycle_forward_edit_gate': float(np.mean(all_cycle_forward_gates)),
        f'{prefix}cycle_consistency_score': float(np.mean(all_cycle_consistency_scores)),
        f'{prefix}total_samples': len(all_matching_scores),
    }

    metrics[f'{prefix}overall_score'] = (
        0.30 * metrics[f'{prefix}r_precision_top1'] +
        0.25 * metrics[f'{prefix}structure_preservation'] +
        0.25 * metrics[f'{prefix}region_alignment_mean'] +
        0.20 * max(0, metrics[f'{prefix}relative_improvement'])
    )
    metrics[f'{prefix}overall_score_with_cycle'] = (
        0.75 * metrics[f'{prefix}overall_score'] +
        0.25 * metrics[f'{prefix}cycle_consistency_score']
    )

    if is_main_process:
        print(f"\n{'=' * 60}")
        print(f"Edit Evaluation Results [{mode_name}]")
        print(f"{'=' * 60}")
        print(f"Total Samples: {metrics[f'{prefix}total_samples']}")
        print(f"\nSemantic Alignment (Target Text vs Generated):")
        print(f"  R-Precision:  Top1={metrics[f'{prefix}r_precision_top1']:.4f}, "
              f"Top2={metrics[f'{prefix}r_precision_top2']:.4f}, Top3={metrics[f'{prefix}r_precision_top3']:.4f}")
        print(f"  Matching Score: {metrics[f'{prefix}matching_score']:.4f}")
        print(f"  FID: {metrics[f'{prefix}fid']:.4f}")
        print(f"\nGT Reference (Target Text vs GT Target Motion):")
        print(f"  R-Precision:  Top1={metrics[f'{prefix}gt_r_precision_top1']:.4f}, "
              f"Top2={metrics[f'{prefix}gt_r_precision_top2']:.4f}, Top3={metrics[f'{prefix}gt_r_precision_top3']:.4f}")
        print(f"  Matching Score: {metrics[f'{prefix}gt_matching_score']:.4f}")
        print(f"\nSource Reference (Source Text vs Source Motion):")
        print(f"  R-Precision:  Top1={metrics[f'{prefix}src_r_precision_top1']:.4f}, "
              f"Top2={metrics[f'{prefix}src_r_precision_top2']:.4f}, Top3={metrics[f'{prefix}src_r_precision_top3']:.4f}")
        print(f"  Matching Score: {metrics[f'{prefix}src_matching_score']:.4f}")
        print(f"\nMotion-to-Motion (Gen vs GT Target):")
        print(f"  R-Precision:  Top1={metrics[f'{prefix}r_precision_gen_vs_gt_top1']:.4f}, "
              f"Top2={metrics[f'{prefix}r_precision_gen_vs_gt_top2']:.4f}, Top3={metrics[f'{prefix}r_precision_gen_vs_gt_top3']:.4f}")
        print(f"\nMotion-to-Motion (Gen vs Source):")
        print(f"  R-Precision:  Top1={metrics[f'{prefix}r_precision_gen_vs_src_top1']:.4f}, "
              f"Top2={metrics[f'{prefix}r_precision_gen_vs_src_top2']:.4f}, Top3={metrics[f'{prefix}r_precision_gen_vs_src_top3']:.4f}")
        print(f"\nStructure Preservation: {metrics[f'{prefix}structure_preservation']:.4f}")
        print(f"Edit Strength: {metrics[f'{prefix}edit_strength']:.4f}")
        print(f"\nEdit Localization & Semantic Alignment:")
        print(f"  Region Alignment: {metrics[f'{prefix}region_alignment_mean']:.4f} +/- {metrics[f'{prefix}region_alignment_std']:.4f}")
        print(f"  Region Coverage:  {metrics[f'{prefix}region_coverage_mean']:.4f}")
        print(f"\nRelative Improvement:")
        print(f"  Source->Target:     {metrics[f'{prefix}source_target_match']:.4f}")
        print(f"  Edited->Target:     {metrics[f'{prefix}edited_target_match']:.4f}")
        print(f"  Improvement:        {metrics[f'{prefix}relative_improvement']:.4f}")
        print(f"  Positive Ratio:     {metrics[f'{prefix}positive_improvement_ratio']:.1%}")
        print(f"\nCycle Consistency (Edited + Reverse Edit -> Source):")
        print(f"  Source Text Matching: {metrics[f'{prefix}cycle_source_matching_score']:.4f}")
        print(f"  Source Text R@1/2/3:  {metrics[f'{prefix}cycle_source_r_precision_top1']:.4f}, "
              f"{metrics[f'{prefix}cycle_source_r_precision_top2']:.4f}, "
              f"{metrics[f'{prefix}cycle_source_r_precision_top3']:.4f}")
        print(f"  Motion->Source R@1:   {metrics[f'{prefix}cycle_motion_vs_source_r_precision_top1']:.4f}")
        print(f"  Struct->Source:       {metrics[f'{prefix}cycle_structure_to_source']:.4f}")
        print(f"  Anti-Identity Gate:   {metrics[f'{prefix}cycle_forward_edit_gate']:.4f}")
        print(f"  Cycle Score:          {metrics[f'{prefix}cycle_consistency_score']:.4f}")
        print(f"\n  Overall Score: {metrics[f'{prefix}overall_score']:.4f}")
        print(f"  Overall + Cycle: {metrics[f'{prefix}overall_score_with_cycle']:.4f}")
        print(f"{'=' * 60}")

    return metrics


def evaluate_gen_test(unimoflow_model, vae, eval_wrapper, dataloader, cfg, device,
                      is_main_process=True, repeat_time=3, cal_mm=True):
    unimoflow_model.eval()
    if vae is not None:
        vae.eval()
    eval_wrapper.eval()

    num_steps = cfg.model.get('eval_noise_steps', cfg.model.noise_steps)
    cfg_scale_val = cfg.model.cfg_scale

    if is_main_process:
        print(f'\n{"=" * 60}')
        print(f'Generation Evaluation (Test Set)')
        print(f'Steps={num_steps}, CFG={cfg_scale_val}, Repeat={repeat_time}')
        print(f'{"=" * 60}')

    fid_list = []
    div_list = []
    top1_list = []
    top2_list = []
    top3_list = []
    matching_list = []
    mm_list = []

    for repeat_id in range(repeat_time):
        if is_main_process:
            print(f"Repeat {repeat_id + 1}/{repeat_time}")

        fid, diversity, R_precision, matching_score, multimodality = \
            eval_t2m.evaluation_diff_withMM(
                val_loader=dataloader,
                diffusion_model=unimoflow_model,
                vae_model=vae,
                repeat_id=repeat_id,
                eval_wrapper=eval_wrapper,
                num_denoise_steps=num_steps,
                cfg_scale=cfg_scale_val,
                device=device,
                cal_mm=cal_mm,
                distributed=False,
                world_size=1,
                local_rank=0,
            )

        if is_main_process:
            fid_list.append(fid)
            div_list.append(diversity)
            top1_list.append(R_precision[0])
            top2_list.append(R_precision[1])
            top3_list.append(R_precision[2])
            matching_list.append(matching_score)
            if cal_mm:
                mm_list.append(multimodality)

    if is_main_process:
        fid_arr = np.array(fid_list)
        div_arr = np.array(div_list)
        top1_arr = np.array(top1_list)
        top2_arr = np.array(top2_list)
        top3_arr = np.array(top3_list)
        matching_arr = np.array(matching_list)

        conf_factor = 1.96 / np.sqrt(repeat_time)

        msg = (
            f"\nGen Final Results (CFG={cfg_scale_val}, Steps={num_steps}):\n"
            f"FID: {np.mean(fid_arr):.3f} +/- {np.std(fid_arr) * conf_factor:.3f}\n"
            f"Diversity: {np.mean(div_arr):.3f} +/- {np.std(div_arr) * conf_factor:.3f}\n"
            f"Top-1: {np.mean(top1_arr):.3f} +/- {np.std(top1_arr) * conf_factor:.3f}\n"
            f"Top-2: {np.mean(top2_arr):.3f} +/- {np.std(top2_arr) * conf_factor:.3f}\n"
            f"Top-3: {np.mean(top3_arr):.3f} +/- {np.std(top3_arr) * conf_factor:.3f}\n"
            f"Matching Score: {np.mean(matching_arr):.3f} +/- {np.std(matching_arr) * conf_factor:.3f}\n"
        )

        if cal_mm and mm_list:
            mm_arr = np.array(mm_list)
            msg += f"Multimodality: {np.mean(mm_arr):.3f} +/- {np.std(mm_arr) * conf_factor:.3f}\n"

        print(msg)

        fid_ci95 = float(np.std(fid_arr) * conf_factor)
        div_ci95 = float(np.std(div_arr) * conf_factor)
        top1_ci95 = float(np.std(top1_arr) * conf_factor)
        top2_ci95 = float(np.std(top2_arr) * conf_factor)
        top3_ci95 = float(np.std(top3_arr) * conf_factor)
        matching_ci95 = float(np.std(matching_arr) * conf_factor)

        gen_metrics = {
            'gen_fid': float(np.mean(fid_arr)),
            'gen_fid_std': fid_ci95,
            'gen_fid_ci95': fid_ci95,
            'gen_diversity': float(np.mean(div_arr)),
            'gen_diversity_std': div_ci95,
            'gen_diversity_ci95': div_ci95,
            'gen_top1': float(np.mean(top1_arr)),
            'gen_top1_ci95': top1_ci95,
            'gen_top2': float(np.mean(top2_arr)),
            'gen_top2_ci95': top2_ci95,
            'gen_top3': float(np.mean(top3_arr)),
            'gen_top3_ci95': top3_ci95,
            'gen_matching_score': float(np.mean(matching_arr)),
            'gen_matching_score_ci95': matching_ci95,
        }
        if cal_mm and mm_list:
            mm_arr = np.array(mm_list)
            gen_metrics['gen_multimodality'] = float(np.mean(mm_arr))
            gen_metrics['gen_multimodality_ci95'] = float(np.std(mm_arr) * conf_factor)

        return gen_metrics
    else:
        return {}


def main():
    parser = argparse.ArgumentParser(description='Evaluate UniMoFlow on Test Set (Edit + Gen)')
    parser.add_argument('--config', type=str, default='../configs/unimoflow.yaml',
                        help='Path to training config')
    parser.add_argument('--which_epoch', type=str, default='gen_best/net_best_fid.tar',
                        help='Checkpoint to evaluate')
    parser.add_argument('--output_dir', type=str, default='../outputs/unimoflow_eval',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Deprecated alias: set both edit/gen batch size when specific flags are not provided')
    parser.add_argument('--edit_batch_size', type=int, default=None,
                        help='Batch size for edit evaluation (default: 32)')
    parser.add_argument('--gen_batch_size', type=int, default=None,
                        help='Batch size for T2M/gen evaluation (default: 100)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Denoise steps for generation')
    parser.add_argument('--cfg_scale', type=float, default=7.5,
                        help='CFG scale')
    parser.add_argument('--repeat_time', type=int, default=1,
                        help='Repeat times for gen evaluation')
    parser.add_argument('--cal_mm', action='store_true', default=False,
                        help='Calculate Multimodality for gen eval')
    parser.add_argument('--skip_edit', action='store_true', default=False,
                        help='Skip edit evaluation')
    parser.add_argument('--skip_gen', action='store_true', default=False,
                        help='Skip gen evaluation')
    parser.add_argument('--edit_test_files', type=str, nargs='+',
                        default=None,
                        help='Test JSON files for edit evaluation. Defaults to config data.edit_test_files or data.edit_data_files.')
    parser.add_argument('--seed', type=int, default=10306,
                        help='Random seed')

    args = parser.parse_args()
    args.edit_batch_size = args.edit_batch_size or args.batch_size or 32
    args.gen_batch_size = args.gen_batch_size or args.batch_size or 100

    local_rank, world_size, device = setup_distributed()
    is_main_process = (local_rank == 0)

    try:
        fixseed(args.seed + local_rank)

        cfg = load_config(args.config)
        cfg.exp.which_epoch = args.which_epoch
        edit_test_files = args.edit_test_files
        if edit_test_files is None:
            edit_test_files = cfg.data.get('edit_test_files', None)
        if edit_test_files is None:
            edit_test_files = cfg.data.get('edit_data_files', None)
        if edit_test_files is None:
            edit_test_files = []

        cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'unimoflow', cfg.exp.name)

        if args.steps is not None:
            cfg.model.eval_noise_steps = args.steps
        if args.cfg_scale is not None:
            cfg.model.cfg_scale = args.cfg_scale

        data_root = cfg.data.root_dir
        meta_dir = pjoin(data_root, 'meta_data')

        mean = np.load(pjoin(meta_dir, 'mean.npy'))
        std = np.load(pjoin(meta_dir, 'std.npy'))

        vae = load_vae(cfg, device)

        unimoflow_model = load_unimoflow_model(cfg, device)

        eval_cfg = load_config(cfg.evaluator.config_path)
        eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
        eval_wrapper.eval()

        all_metrics = {
            'model_name': cfg.exp.name,
            'checkpoint': args.which_epoch,
            'config': args.config,
            'steps': cfg.model.get('eval_noise_steps', cfg.model.noise_steps),
            'cfg_scale': cfg.model.cfg_scale,
            'edit_batch_size': args.edit_batch_size,
            'gen_batch_size': args.gen_batch_size,
            'edit_test_files': list(edit_test_files),
        }

        # ==============================
        # Edit Evaluation
        # ==============================
        if not args.skip_edit:
            edit_dataset = MotionEditDataset(
                data_root=cfg.data.root_dir,
                split="test",
                data_files=edit_test_files,
                cycle_aug_prob=0.0,
                caption_as_edit_prob=0.0,
                max_length=cfg.data.max_motion_length,
                mean=None,
                std=None,
            )

            edit_loader = DataLoader(
                edit_dataset,
                batch_size=args.edit_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn,
                pin_memory=True,
            )

            if is_main_process:
                print(f"Edit test dataset size: {len(edit_dataset)}, batch_size={args.edit_batch_size}")

            # flow_edit mode
            edit_metrics_flowedit = evaluate_edit_test(
                unimoflow_model=unimoflow_model,
                vae=vae,
                eval_wrapper=eval_wrapper,
                dataloader=edit_loader,
                cfg=cfg,
                device=device,
                is_main_process=is_main_process,
                self_flowedit=True,
            )

            # regular generate mode
            edit_metrics_regular = evaluate_edit_test(
                unimoflow_model=unimoflow_model,
                vae=vae,
                eval_wrapper=eval_wrapper,
                dataloader=edit_loader,
                cfg=cfg,
                device=device,
                is_main_process=is_main_process,
                self_flowedit=False,
            )

            if is_main_process:
                all_metrics.update(edit_metrics_flowedit)
                all_metrics.update(edit_metrics_regular)

        # ==============================
        # Gen Evaluation
        # ==============================
        if not args.skip_gen:
            cfg.data.feat_dir = pjoin(data_root, 'renamed_feats')
            data_split_dir = pjoin(data_root, 'data_split_info')
            all_caption_path = pjoin(data_root, 'all_caption_clean.json')
            test_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
            test_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')

            gen_dataset = TextMotionDataset(cfg, mean, std, test_mid_split_file, test_cid_split_file, all_caption_path)

            gen_loader = DataLoader(
                gen_dataset,
                batch_size=args.gen_batch_size,
                drop_last=False,
                num_workers=4,
                shuffle=False,
                pin_memory=True,
            )

            if is_main_process:
                print(f"Gen test dataset size: {len(gen_dataset)}, batch_size={args.gen_batch_size}")

            gen_metrics = evaluate_gen_test(
                unimoflow_model=unimoflow_model,
                vae=vae,
                eval_wrapper=eval_wrapper,
                dataloader=gen_loader,
                cfg=cfg,
                device=device,
                is_main_process=is_main_process,
                repeat_time=args.repeat_time,
                cal_mm=args.cal_mm,
            )

            if is_main_process:
                all_metrics.update(gen_metrics)

        # ==============================
        # Save Results
        # ==============================
        if is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)

            results_path = pjoin(args.output_dir, 'eval_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {results_path}")

            if not args.skip_edit and not args.skip_gen:
                flow_match = all_metrics.get('edit_flowedit_matching_score', 0)
                flow_struct = all_metrics.get('edit_flowedit_structure_preservation', 0)
                flow_relimp = all_metrics.get('edit_flowedit_relative_improvement', 0)
                flow_gen_vs_gt = all_metrics.get('edit_flowedit_r_precision_gen_vs_gt_top1', 0)
                flow_cycle = all_metrics.get('edit_flowedit_cycle_consistency_score', 0)
                reg_match = all_metrics.get('edit_regular_matching_score', 0)
                reg_struct = all_metrics.get('edit_regular_structure_preservation', 0)
                reg_relimp = all_metrics.get('edit_regular_relative_improvement', 0)
                reg_gen_vs_gt = all_metrics.get('edit_regular_r_precision_gen_vs_gt_top1', 0)
                reg_cycle = all_metrics.get('edit_regular_cycle_consistency_score', 0)
                summary = (
                    f"\n{'=' * 60}\n"
                    f"SUMMARY\n"
                    f"{'=' * 60}\n"
                    f"Edit-FlowEdit | Matching: {flow_match:.4f}  "
                    f"Struct: {flow_struct:.4f}  "
                    f"Gen_vs_GT: {flow_gen_vs_gt:.4f}  "
                    f"RelImp: {flow_relimp:.4f}  "
                    f"Cycle: {flow_cycle:.4f}\n"
                    f"Edit-Regular  | Matching: {reg_match:.4f}  "
                    f"Struct: {reg_struct:.4f}  "
                    f"Gen_vs_GT: {reg_gen_vs_gt:.4f}  "
                    f"RelImp: {reg_relimp:.4f}  "
                    f"Cycle: {reg_cycle:.4f}\n"
                    f"Gen   | FID: {all_metrics.get('gen_fid', 0):.3f}  "
                    f"Div: {all_metrics.get('gen_diversity', 0):.3f}  "
                    f"Top1: {all_metrics.get('gen_top1', 0):.3f}  "
                    f"Matching: {all_metrics.get('gen_matching_score', 0):.3f}\n"
                    f"{'=' * 60}"
                )
                print(summary)

                summary_path = pjoin(args.output_dir, 'summary.txt')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary + '\n')
                    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    except Exception as e:
        import traceback
        print(f"[Rank {local_rank}] Error: {e}")
        traceback.print_exc()
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
