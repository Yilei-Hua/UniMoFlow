# edit_evaluator.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .metrics import (
    calculate_R_precision, cosine_similarity_matrix,
    calculate_activation_statistics, calculate_frechet_distance,
    calculate_diversity
)
import torch.distributed as dist


def length_to_mask(length, max_len, device: torch.device = None):
    """长度转mask"""
    if device is None:
        device = "cpu"
    if isinstance(length, list):
        length = torch.tensor(length)
    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


# ==================== 新增：无GT指标辅助函数 ====================

@torch.no_grad()
def _compute_edit_region_alignment(source_motions, edited_motions, edit_commands,
                                   orig_lengths, eval_wrapper, threshold_percentile=75.0, min_seg_len=8):
    """
    指标1：差异热力图 + 区域语义匹配 (Edit Localization & Semantic Alignment)
    """
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

        _, em_seg, _ = eval_wrapper.encode_motion(seg_motion[..., :148], seg_len, sample_mean=True)
        cos_sim = F.cosine_similarity(em_seg, et_edit[i:i+1], dim=-1)
        region_scores.append(float(cos_sim.item()))

    return {
        'region_alignment_scores': np.array(region_scores),
        'region_coverages': np.array(region_coverages),
        'peak_delta_ratios': np.array(peak_delta_ratios),
    }


@torch.no_grad()
def _compute_relative_improvement(source_motions, edited_motions, target_captions,
                                  orig_lengths, eval_wrapper):
    """
    指标2：相对改进评估 (Relative Improvement)
    """
    et_target, _ = eval_wrapper.encode_text(target_captions, sample_mean=True)
    _, em_source, _ = eval_wrapper.encode_motion(source_motions[..., :148], orig_lengths, sample_mean=True)
    _, em_edited, _ = eval_wrapper.encode_motion(edited_motions[..., :148], orig_lengths, sample_mean=True)

    source_target_match = F.cosine_similarity(em_source, et_target, dim=-1).cpu().numpy()
    edited_target_match = F.cosine_similarity(em_edited, et_target, dim=-1).cpu().numpy()
    relative_improvement = edited_target_match - source_target_match
    improvement_ratio = edited_target_match / (source_target_match + 1e-8)

    return {
        'source_target_match': source_target_match,
        'edited_target_match': edited_target_match,
        'relative_improvement': relative_improvement,
        'improvement_ratio': improvement_ratio,
    }

# ============================================================


@torch.no_grad()
def evaluate_edit_model(
        out_dir,
        val_loader,
        edit_model,
        vae_model,
        eval_wrapper,
        writer,
        ep,
        best_fid,
        best_matching,
        best_structure_preservation,
        best_overall_score,
        device,
        plot_func=None,
        num_denoise_steps=20,
        cfg_scale=4.0,
        save_ckpt=True,
        save_anim=False,
        distributed=False,
        world_size=1,
        local_rank=0,
        structure_weight=0.5,
        matching_weight=0.5,
        max_visualize_samples=4,
        return_positive_improve=False
):
    """
    编辑模型评估函数 - 修复版（集成无GT指标）

    评估指标：
    1. 语义匹配度（Matching）: 目标文本 vs 生成动作 (R-precision, Matching Score)
    2. 结构保留度（Structure）: 源动作 vs 生成动作 (Latent空间余弦相似度)
    3. 生成质量（FID）: 生成动作 vs GT目标动作
    4. GT参考指标: 目标文本 vs GT目标动作（用于对比）
    5. [NEW] 区域语义匹配: 变化区域 vs 指令文本
    6. [NEW] 相对改进: 编辑后 vs 目标文本 相对 源 vs 目标文本
    """
    edit_model.eval()
    if vae_model is not None:
        vae_model.eval()

    # 收集用于FID计算的特征
    motion_annotation_list = []  # GT目标动作特征
    motion_pred_list = []  # 生成动作特征

    # 语义匹配度指标 - 生成结果
    R_precision_pred = np.array([0., 0., 0.])
    matching_score_pred = 0.0

    # 语义匹配度指标 - GT目标（参考值）
    R_precision_real = np.array([0., 0., 0.])
    matching_score_real = 0.0

    # 结构保留度指标
    structure_preservation_scores = []
    structure_details = []  # 存储详细结构信息用于分析

    # 新增：指标1 收集器
    all_region_alignment_scores = []
    all_region_coverages = []
    all_peak_delta_ratios = []

    # 新增：指标2 收集器
    all_source_target_matches = []
    all_edited_target_matches = []
    all_relative_improvements = []
    all_improvement_ratios = []

    nb_sample = 0

    # 可视化数据缓存 - 确保收集所有需要的信息
    viz_data_list = []

    if local_rank == 0:
        print(f"--> \t Evaluating Edit Diffusion Model...")
        print(f"       Steps: {num_denoise_steps}, CFG: {cfg_scale}")

    # 设置sampler epoch
    if distributed and hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(ep)

    for batch in tqdm(val_loader, disable=(local_rank != 0)):
        # 解包数据
        source_latents = batch["source"].to(device).float().detach()
        target_latents = batch["target"].to(device).float().detach()
        edit_texts = batch["edit_text"]
        m_lengths = batch["length"].to(device).long().detach()
        source_captions = batch.get("source_caption", [""] * len(edit_texts))
        target_captions = batch.get("target_caption", [""] * len(edit_texts))

        bs = source_latents.shape[0]
        latent_len = source_latents.shape[1]

        # 解码latent获取motion（用于evaluator计算特征）
        if vae_model is not None:
            with torch.no_grad():
                # VAE解码需要原始motion长度（考虑下采样因子）
                ds_factor = getattr(vae_model, 'downsample_factor', 4)
                orig_lengths = m_lengths * ds_factor

                source_motions = vae_model.forward_decoder(source_latents, orig_lengths.clone())
                target_motions = vae_model.forward_decoder(target_latents, orig_lengths.clone())
        else:
            source_motions = source_latents
            target_motions = target_latents
            ds_factor = 1
            orig_lengths = m_lengths

        # 1. 编码文本和GT目标动作（用于语义匹配度计算）
        et_target, _ = eval_wrapper.encode_text(target_captions, sample_mean=True)
        fid_em_target, em_target, _ = eval_wrapper.encode_motion(
            target_motions[..., :148], orig_lengths, sample_mean=True
        )

        # 2. 编辑模型生成
        x_in = {
            "source": source_latents,
            "edit_text": edit_texts,
            "length": m_lengths
        }

        # 临时设置cfg_scale
        raw_model = edit_model.module if hasattr(edit_model, 'module') else edit_model
        old_cfg = raw_model.cfg_scale if hasattr(raw_model, 'cfg_scale') else cfg_scale
        if hasattr(raw_model, 'cfg_scale'):
            raw_model.cfg_scale = cfg_scale

        with torch.no_grad():
            gen_output = raw_model.generate(x_in, num_denoise_steps=num_denoise_steps)
        pred_latents = gen_output["generated"] if isinstance(gen_output, dict) else gen_output

        if hasattr(raw_model, 'cfg_scale'):
            raw_model.cfg_scale = old_cfg

        # 3. 解码生成结果
        if vae_model is not None:
            with torch.no_grad():
                pred_motions = vae_model.forward_decoder(pred_latents, orig_lengths.clone())
        else:
            pred_motions = pred_latents

        # 4. 编码生成结果特征
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(
            pred_motions[..., :148], orig_lengths, sample_mean=True
        )

        # 收集FID统计用特征
        motion_annotation_list.append(fid_em_target.cpu())
        motion_pred_list.append(fid_em_pred.cpu())

        # 5. 计算语义匹配度（目标文本 vs 生成动作）
        # 修复：动态调整top_k防止batch size < 3时的IndexError
        actual_top_k = min(3, bs)
        temp_R_pred = calculate_R_precision(
            et_target.cpu().numpy(), em_pred.cpu().numpy(),
            top_k=actual_top_k, sum_all=True, is_cosine_sim=True
        )
        # 补齐到3维以保持与累加器R_precision_pred维度一致
        if temp_R_pred.shape[0] < 3:
            temp_R_pred = np.pad(temp_R_pred, (0, 3 - temp_R_pred.shape[0]), mode='constant')

        temp_match_pred = cosine_similarity_matrix(
            et_target.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()

        R_precision_pred += temp_R_pred
        matching_score_pred += float(temp_match_pred)

        # 6. 计算GT语义匹配度（目标文本 vs GT目标动作）- 作为参考基准
        temp_R_real = calculate_R_precision(
            et_target.cpu().numpy(), em_target.cpu().numpy(),
            top_k=actual_top_k, sum_all=True, is_cosine_sim=True
        )
        if temp_R_real.shape[0] < 3:
            temp_R_real = np.pad(temp_R_real, (0, 3 - temp_R_real.shape[0]), mode='constant')

        temp_match_real = cosine_similarity_matrix(
            et_target.cpu().numpy(), em_target.cpu().numpy()
        ).trace()
        R_precision_real += temp_R_real
        matching_score_real += float(temp_match_real)

        # 7. 计算结构保留度（源动作 vs 生成动作）- 在Latent空间
        # 使用mask处理变长序列
        latent_mask = length_to_mask(m_lengths, latent_len, device).unsqueeze(-1)  # [B, T, 1]

        # 计算每个样本的余弦相似度（考虑实际长度）
        for i in range(bs):
            actual_len = m_lengths[i].item()

            # 提取有效长度部分
            src_vec = source_latents[i, :actual_len].flatten()  # [actual_len * D]
            pred_vec = pred_latents[i, :actual_len].flatten()

            # 余弦相似度（结构越相似越接近1）
            cos_sim = F.cosine_similarity(src_vec.unsqueeze(0), pred_vec.unsqueeze(0), dim=-1)
            structure_preservation_scores.append(cos_sim.item())

            # 记录详细信息用于调试
            structure_details.append({
                'sample_id': batch.get('pair_id', [f'ep{ep}_{idx}' for idx in range(bs)])[i],
                'cos_sim': cos_sim.item(),
                'length': actual_len,
                'is_cycle': batch.get('is_cycle', [False] * bs)[i]
            })

        # 新增：计算无GT指标
        region_metrics = _compute_edit_region_alignment(
            source_motions, pred_motions, edit_texts, orig_lengths, eval_wrapper
        )
        all_region_alignment_scores.extend(region_metrics['region_alignment_scores'].tolist())
        all_region_coverages.extend(region_metrics['region_coverages'].tolist())
        all_peak_delta_ratios.extend(region_metrics['peak_delta_ratios'].tolist())

        imp_metrics = _compute_relative_improvement(
            source_motions, pred_motions, target_captions, orig_lengths, eval_wrapper
        )
        all_source_target_matches.extend(imp_metrics['source_target_match'].tolist())
        all_edited_target_matches.extend(imp_metrics['edited_target_match'].tolist())
        all_relative_improvements.extend(imp_metrics['relative_improvement'].tolist())
        all_improvement_ratios.extend(imp_metrics['improvement_ratio'].tolist())

        nb_sample += bs

        # 8. 缓存可视化数据（限制样本数，仅主进程）
        if local_rank == 0 and len(viz_data_list) < max_visualize_samples:
            # 收集当前batch的所有样本
            for i in range(min(bs, max_visualize_samples - len(viz_data_list))):
                viz_data_list.append({
                    "source_motion": source_motions[i].cpu(),
                    "pred_motion": pred_motions[i].cpu(),
                    "target_motion": target_motions[i].cpu(),
                    "edit_text": edit_texts[i],
                    "source_caption": source_captions[i],
                    "target_caption": target_captions[i],
                    "length": orig_lengths[i].cpu().item(),
                    "pair_id": batch.get('pair_id', [f'ep{ep}_{i}'])[i],
                    "structure_score": structure_preservation_scores[-bs + i] if i < len(
                        structure_preservation_scores[-bs:]) else 0.0
                })

    # 9. 分布式聚合所有指标
    if distributed and dist.is_initialized():
        # 聚合FID特征
        local_annotation = torch.cat(motion_annotation_list, dim=0) if motion_annotation_list else torch.empty(0, 512,
                                                                                                               device=device)
        local_pred = torch.cat(motion_pred_list, dim=0) if motion_pred_list else torch.empty(0, 512, device=device)

        gathered_annotations = [torch.zeros_like(local_annotation) for _ in range(world_size)]
        gathered_preds = [torch.zeros_like(local_pred) for _ in range(world_size)]

        dist.all_gather(gathered_annotations, local_annotation)
        dist.all_gather(gathered_preds, local_pred)

        if local_rank == 0:
            motion_annotation_list = gathered_annotations
            motion_pred_list = gathered_preds
        else:
            motion_annotation_list = []
            motion_pred_list = []

        # 聚合统计指标
        rp_pred_tensor = torch.from_numpy(R_precision_pred).to(device)
        rp_real_tensor = torch.from_numpy(R_precision_real).to(device)

        dist.all_reduce(rp_pred_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(rp_real_tensor, op=dist.ReduceOp.SUM)

        R_precision_pred = rp_pred_tensor.cpu().numpy()
        R_precision_real = rp_real_tensor.cpu().numpy()

        metrics_tensor = torch.tensor([
            float(matching_score_pred),
            float(matching_score_real),
            float(np.sum(structure_preservation_scores)),
            float(np.sum(all_region_alignment_scores)) if all_region_alignment_scores else 0.0,
            float(np.sum(all_relative_improvements)) if all_relative_improvements else 0.0,
            float(np.sum(np.array(all_relative_improvements) > 0.0)) if all_relative_improvements else 0.0,
            float(nb_sample)
        ], device=device, dtype=torch.float64)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        matching_score_pred = metrics_tensor[0].item()
        matching_score_real = metrics_tensor[1].item()
        total_structure_score = metrics_tensor[2].item()
        total_region_align = metrics_tensor[3].item()
        total_rel_imp = metrics_tensor[4].item()
        total_positive_imp = metrics_tensor[5].item()
        total_samples = int(metrics_tensor[6].item())
    else:
        total_samples = nb_sample
        total_structure_score = np.sum(structure_preservation_scores)
        total_region_align = np.sum(all_region_alignment_scores) if all_region_alignment_scores else 0.0
        total_rel_imp = np.sum(all_relative_improvements) if all_relative_improvements else 0.0
        total_positive_imp = np.sum(np.array(all_relative_improvements) > 0.0) if all_relative_improvements else 0.0
        # 非分布式时转为numpy数组
        motion_annotation_list = [
            torch.cat(motion_annotation_list, dim=0).cpu().numpy()] if motion_annotation_list else [np.zeros((1, 512))]
        motion_pred_list = [torch.cat(motion_pred_list, dim=0).cpu().numpy()] if motion_pred_list else [
            np.zeros((1, 512))]

    # 10. 计算最终指标（仅主进程）
    if not distributed or local_rank == 0:
        # 合并特征
        motion_annotation_np = np.concatenate([x for x in motion_annotation_list if len(x) > 0], axis=0)
        motion_pred_np = np.concatenate([x for x in motion_pred_list if len(x) > 0], axis=0)

        # FID（生成质量：生成动作 vs GT目标动作）
        if len(motion_annotation_np) > 1 and len(motion_pred_np) > 1:
            gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
            mu, cov = calculate_activation_statistics(motion_pred_np)
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        else:
            fid = 9999.0

        # Diversity
        diversity_real = calculate_diversity(motion_annotation_np, min(300, len(motion_annotation_np)))
        diversity = calculate_diversity(motion_pred_np, min(300, len(motion_pred_np)))

        # 平均指标计算
        R_precision_pred_avg = R_precision_pred / total_samples
        matching_score_pred_avg = matching_score_pred / total_samples

        R_precision_real_avg = R_precision_real / total_samples
        matching_score_real_avg = matching_score_real / total_samples

        structure_preservation_avg = total_structure_score / total_samples
        region_alignment_avg = total_region_align / total_samples
        rel_imp_avg = total_rel_imp / total_samples
        positive_improvement_ratio = total_positive_imp / total_samples

        # 综合评分（语义匹配度 + 结构保留度 + 区域语义匹配 + 相对改进）
        overall_score = (
                matching_weight * R_precision_pred_avg[0] +
                structure_weight * structure_preservation_avg +
                0.25 * region_alignment_avg +
                0.20 * max(0, rel_imp_avg)
        )

        # 11. 详细日志输出（包含GT参考值对比 + 无GT指标）
        msg = (
            f"\n{'=' * 60}\n"
            f"Epoch {ep} Evaluation Results:\n"
            f"{'=' * 60}\n"
            f"Generation Quality:\n"
            f"  FID (Pred vs GT):      {fid:.4f} (lower is better)\n"
            f"  Diversity (Pred):      {diversity:.4f} | Real: {diversity_real:.4f}\n"
            f"\nSemantic Matching (Target Text vs Generated):\n"
            f"  R-Precision:           Top1={R_precision_pred_avg[0]:.4f}, "
            f"Top2={R_precision_pred_avg[1]:.4f}, Top3={R_precision_pred_avg[2]:.4f}\n"
            f"  Matching Score:        {matching_score_pred_avg:.4f}\n"
            f"\nGT Reference (Target Text vs GT Target Motion):\n"
            f"  R-Precision:           Top1={R_precision_real_avg[0]:.4f}, "
            f"Top2={R_precision_real_avg[1]:.4f}, Top3={R_precision_real_avg[2]:.4f}\n"
            f"  Matching Score:        {matching_score_real_avg:.4f}\n"
            f"\nStructure Preservation (Source vs Generated):\n"
            f"  Avg Cosine Sim:        {structure_preservation_avg:.4f}\n"
            f"  Samples > 0.7:         {sum(1 for s in structure_preservation_scores if s > 0.7)}/{len(structure_preservation_scores)}\n"
            f"\n[NEW] Edit Localization & Semantic Alignment (Metric 1):\n"
            f"  Region Alignment:      {region_alignment_avg:.4f}\n"
            f"  Region Coverage:       {np.mean(all_region_coverages):.4f}\n"
            f"  Peak Delta Ratio:      {np.mean(all_peak_delta_ratios):.4f}\n"
            f"\n[NEW] Relative Improvement (Metric 2):\n"
            f"  Source→Target Match:   {np.mean(all_source_target_matches):.4f}\n"
            f"  Edited→Target Match:   {np.mean(all_edited_target_matches):.4f}\n"
            f"  Relative Improvement:  {rel_imp_avg:.4f}\n"
            f"  Positive Improvement:  {positive_improvement_ratio:.1%} of samples\n"
            f"\nOverall Score:           {overall_score:.4f}\n"
            f"{'=' * 60}"
        )
        print(msg)

        if writer is not None:
            # 生成质量
            writer.add_scalar('EditEval/FID', fid, ep)
            writer.add_scalar('EditEval/Diversity', diversity, ep)
            writer.add_scalar('EditEval/Diversity_Real', diversity_real, ep)

            # 语义匹配度（生成）
            writer.add_scalar('EditEval/R_Top1_Pred', R_precision_pred_avg[0], ep)
            writer.add_scalar('EditEval/R_Top2_Pred', R_precision_pred_avg[1], ep)
            writer.add_scalar('EditEval/R_Top3_Pred', R_precision_pred_avg[2], ep)
            writer.add_scalar('EditEval/Matching_Score_Pred', matching_score_pred_avg, ep)

            # GT参考（用于监控过拟合）
            writer.add_scalar('EditEval/R_Top1_Real', R_precision_real_avg[0], ep)
            writer.add_scalar('EditEval/Matching_Score_Real', matching_score_real_avg, ep)

            # 结构保留度
            writer.add_scalar('EditEval/Structure_Preservation', structure_preservation_avg, ep)

            # 新增无GT指标
            writer.add_scalar('EditEval/Region_Alignment', region_alignment_avg, ep)
            writer.add_scalar('EditEval/Relative_Improvement', rel_imp_avg, ep)
            writer.add_scalar('EditEval/Positive_Improvement_Ratio', positive_improvement_ratio, ep)
            writer.add_scalar('EditEval/Overall_Score', overall_score, ep)

        # 12. 最佳模型保存逻辑（多维度监控）
        updated = False

        # 基于综合评分保存
        if overall_score > best_overall_score:
            msg = f"--> --> \t Overall Score Improved from {best_overall_score:.5f} to {overall_score:.5f} !!!"
            print(msg)
            best_overall_score = overall_score
            if save_ckpt:
                save_path = os.path.join(out_dir, 'net_best_overall.tar')
                torch.save({
                    "model": edit_model.module.state_dict() if hasattr(edit_model,
                                                                       'module') else edit_model.state_dict(),
                    "ep": ep,
                    "overall_score": overall_score,
                    "fid": fid,
                    "matching_score": matching_score_pred_avg,
                    "structure_preservation": structure_preservation_avg,
                    "metrics": {
                        "R_precision_pred": R_precision_pred_avg.tolist(),
                        "R_precision_real": R_precision_real_avg.tolist(),
                        "matching_pred": matching_score_pred_avg,
                        "matching_real": matching_score_real_avg,
                        "structure": structure_preservation_avg,
                        "region_alignment": region_alignment_avg,
                        "relative_improvement": rel_imp_avg,
                    }
                }, save_path)
            updated = True

        # 单独保存最佳语义匹配
        if matching_score_pred_avg > best_matching:
            msg = f"--> --> \t Matching Score Improved from {best_matching:.5f} to {matching_score_pred_avg:.5f} !!!"
            print(msg)
            best_matching = matching_score_pred_avg
            if save_ckpt and not updated:
                save_path = os.path.join(out_dir, 'net_best_matching.tar')
                torch.save({
                    "model": edit_model.module.state_dict() if hasattr(edit_model,
                                                                       'module') else edit_model.state_dict(),
                    "ep": ep,
                    "metrics": {
                        "matching": matching_score_pred_avg,
                        "R_precision": R_precision_pred_avg.tolist()
                    }
                }, save_path)

        # 单独保存最佳结构保留
        if structure_preservation_avg > best_structure_preservation:
            msg = f"--> --> \t Structure Preservation Improved from {best_structure_preservation:.5f} to {structure_preservation_avg:.5f} !!!"
            print(msg)
            best_structure_preservation = structure_preservation_avg

        # 保存最新
        if save_ckpt:
            latest_path = os.path.join(out_dir, 'net_last.tar')
            torch.save({
                "model": edit_model.module.state_dict() if hasattr(edit_model, 'module') else edit_model.state_dict(),
                "ep": ep,
                "metrics": {
                    "overall": overall_score,
                    "fid": fid,
                    "matching": matching_score_pred_avg,
                    "structure": structure_preservation_avg,
                    "region_alignment": region_alignment_avg,
                    "relative_improvement": rel_imp_avg,
                }
            }, latest_path)

        # 13. 可视化保存（修复：正确分别保存源、生成、目标）
        if save_anim and plot_func is not None and len(viz_data_list) > 0:
            try:
                print(f"--> \t Generating edit visualization for Epoch {ep}...")

                anim_dir = os.path.join(out_dir, 'animation', f'E{ep:04d}')
                os.makedirs(anim_dir, exist_ok=True)

                # 分别保存每个样本的三个视角
                for idx, viz_data in enumerate(viz_data_list[:max_visualize_samples]):
                    sample_dir = os.path.join(anim_dir, f'sample_{idx:02d}')
                    os.makedirs(sample_dir, exist_ok=True)

                    length = viz_data["length"]
                    ds_factor = getattr(vae_model, 'downsample_factor', 4) if vae_model else 1

                    # 准备caption信息（包含指标）
                    struct_score = viz_data.get('structure_score', 0.0)

                    # 源动作
                    source_cap = f"[SOURCE] {viz_data['source_caption']}"
                    source_data = viz_data["source_motion"].unsqueeze(0).numpy()

                    # 生成动作（包含编辑命令和结构保留度）
                    pred_cap = f"[EDITED] {viz_data['edit_text']}\nStructSim: {struct_score:.3f}"
                    pred_data = viz_data["pred_motion"].unsqueeze(0).numpy()

                    # 目标动作（GT）
                    target_cap = f"[TARGET GT] {viz_data['target_caption']}"
                    target_data = viz_data["target_motion"].unsqueeze(0).numpy()

                    # 修复：直接传入目录路径和prefix，避免文件名冲突
                    # Source
                    plot_func(
                        source_data,
                        sample_dir,  # 传入目录
                        [source_cap],
                        [length],
                        prefix="source_"  # 指定前缀避免覆盖
                    )

                    # Generated
                    plot_func(
                        pred_data,
                        sample_dir,
                        [pred_cap],
                        [length],
                        prefix="generated_"
                    )

                    # Target
                    plot_func(
                        target_data,
                        sample_dir,
                        [target_cap],
                        [length],
                        prefix="target_"
                    )

                    # 保存文本信息到txt方便查看
                    info_path = os.path.join(sample_dir, 'info.txt')
                    with open(info_path, 'w') as f:
                        f.write(f"Pair ID: {viz_data.get('pair_id', 'unknown')}\n")
                        f.write(f"Structure Preservation: {struct_score:.4f}\n")
                        f.write(f"\nSource: {viz_data['source_caption']}\n")
                        f.write(f"Edit: {viz_data['edit_text']}\n")
                        f.write(f"Target: {viz_data['target_caption']}\n")

                print(f"--> \t Visualizations saved to {anim_dir}")

            except Exception as e:
                print(f"[!] Error saving animation: {e}")
                import traceback
                traceback.print_exc()

        result = (
            fid, best_matching, best_structure_preservation, best_overall_score,
            R_precision_pred_avg[0], R_precision_pred_avg[1], R_precision_pred_avg[2],
            matching_score_pred_avg, structure_preservation_avg
        )
        if return_positive_improve:
            result = result + (positive_improvement_ratio,)
        return result

    else:
        # 非主进程返回原值
        result = (best_fid, best_matching, best_structure_preservation, best_overall_score,
                  0, 0, 0, 0, 0)
        if return_positive_improve:
            result = result + (0,)
        return result
