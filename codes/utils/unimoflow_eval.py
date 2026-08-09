import torch
import os
from os.path import join as pjoin
import numpy as np
from collections import OrderedDict
from utils.edit_evaluator import evaluate_edit_model
from utils.eval_t2m import evaluation_diffusion_model
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class UniMoFlowMetricsTracker:
    """
    混合训练指标追踪器：分别追踪编辑和生成两种模式的最佳指标
    """

    def __init__(self):
        # 编辑模式最佳指标
        self.edit_best_fid = float('inf')
        self.edit_best_matching = -float('inf')
        self.edit_best_structure = -float('inf')
        self.edit_best_overall = -float('inf')
        self.edit_best_top1 = -float('inf')
        self.edit_best_positive_improve = -float('inf')

        # 生成模式最佳指标
        self.gen_best_fid = float('inf')
        self.gen_best_div = -float('inf')
        self.gen_best_top1 = -float('inf')
        self.gen_best_top2 = -float('inf')
        self.gen_best_top3 = -float('inf')
        self.gen_best_matching = -float('inf')

    def update_edit_best(self, fid, match_score, struct_score, overall_score, top1, positive_improve, model_dir, model_state_dict, ep):
        """更新编辑模式最佳指标并保存最优checkpoint"""
        updated = False
        save_dir = pjoin(model_dir, 'edit_best')
        os.makedirs(save_dir, exist_ok=True)

        if match_score > self.edit_best_matching:
            self.edit_best_matching = match_score
            torch.save({"model": model_state_dict, "ep": ep, "matching_score": match_score,
                        "top1": top1, "positive_improve": positive_improve, "fid": fid,
                        "structure": struct_score},
                       pjoin(save_dir, 'net_best_matching.tar'))
            updated = True

        if top1 > self.edit_best_top1:
            self.edit_best_top1 = top1
            torch.save({"model": model_state_dict, "ep": ep, "top1": top1,
                        "matching_score": match_score, "positive_improve": positive_improve,
                        "fid": fid, "structure": struct_score},
                       pjoin(save_dir, 'net_best_top1.tar'))
            updated = True

        if positive_improve > self.edit_best_positive_improve:
            self.edit_best_positive_improve = positive_improve
            torch.save({"model": model_state_dict, "ep": ep,
                        "positive_improve": positive_improve, "matching_score": match_score,
                        "top1": top1, "fid": fid, "structure": struct_score},
                       pjoin(save_dir, 'net_best_positive_improve.tar'))
            updated = True

        self.edit_best_fid = min(self.edit_best_fid, fid)
        self.edit_best_overall = max(self.edit_best_overall, overall_score)
        if struct_score > self.edit_best_structure:
            self.edit_best_structure = struct_score

        return updated

    def update_gen_best(self, fid, diversity, top1, top2, top3, matching, model_dir, model_state_dict, ep):
        """更新生成模式最佳指标并保存最优checkpoint"""
        updated = False
        save_dir = pjoin(model_dir, 'gen_best')
        os.makedirs(save_dir, exist_ok=True)

        if fid < self.gen_best_fid:
            self.gen_best_fid = fid
            torch.save({"model": model_state_dict, "ep": ep, "fid": fid},
                       pjoin(save_dir, 'net_best_fid.tar'))
            updated = True

        if matching > self.gen_best_matching:
            self.gen_best_matching = matching
            torch.save({"model": model_state_dict, "ep": ep, "matching_score": matching},
                       pjoin(save_dir, 'net_best_matching.tar'))
            updated = True

        if top1 > self.gen_best_top1:
            self.gen_best_top1 = top1
            torch.save({"model": model_state_dict, "ep": ep, "top1": top1},
                       pjoin(save_dir, 'net_best_top1.tar'))
            updated = True

        self.gen_best_div = max(self.gen_best_div, diversity) if diversity > 0 else self.gen_best_div
        self.gen_best_top2 = max(self.gen_best_top2, top2)
        self.gen_best_top3 = max(self.gen_best_top3, top3)

        return updated

    def state_dict(self):
        return {
            'edit_best_fid': self.edit_best_fid,
            'edit_best_matching': self.edit_best_matching,
            'edit_best_structure': self.edit_best_structure,
            'edit_best_overall': self.edit_best_overall,
            'edit_best_top1': self.edit_best_top1,
            'edit_best_positive_improve': self.edit_best_positive_improve,
            'gen_best_fid': self.gen_best_fid,
            'gen_best_div': self.gen_best_div,
            'gen_best_top1': self.gen_best_top1,
            'gen_best_top2': self.gen_best_top2,
            'gen_best_top3': self.gen_best_top3,
            'gen_best_matching': self.gen_best_matching,
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


def get_raw_model(model):
    """获取底层模型（解包DDP包装）"""
    return model.module if isinstance(model, DDP) else model


def get_model_state(model):
    """获取模型state_dict（解包DDP）"""
    return get_raw_model(model).state_dict()


def evaluate_edit(
        out_dir,
        val_loader,
        unimoflow_model,
        vae_model,
        eval_wrapper,
        writer,
        ep,
        tracker: UniMoFlowMetricsTracker,
        device,
        plot_func=None,
        num_denoise_steps=10,
        cfg_scale=5.0,
        save_ckpt=True,
        save_anim=False,
        distributed=False,
        world_size=1,
        local_rank=0,
        structure_weight=0.5,
        matching_weight=0.5,
):
    """
    编辑模式评估：封装 evaluate_edit_model，兼容混合模型的 generate 接口

    返回: (fid, match_score, struct_score, overall_score, top1, top2, top3)
    """
    raw_model = get_raw_model(unimoflow_model)

    fid, best_matching, best_structure, best_overall, top1, top2, top3, match_score, struct_score, positive_improve = \
        evaluate_edit_model(
            out_dir=out_dir,
            val_loader=val_loader,
            edit_model=raw_model,
            vae_model=vae_model.module if isinstance(vae_model, DDP) else vae_model,
            eval_wrapper=eval_wrapper,
            writer=writer,
            ep=ep,
            best_fid=tracker.edit_best_fid,
            best_matching=tracker.edit_best_matching,
            best_structure_preservation=tracker.edit_best_structure,
            best_overall_score=tracker.edit_best_overall,
            device=device,
            plot_func=plot_func,
            num_denoise_steps=num_denoise_steps,
            cfg_scale=cfg_scale,
            save_ckpt=False,
            save_anim=save_anim,
            distributed=distributed,
            world_size=world_size,
            local_rank=local_rank,
            structure_weight=structure_weight,
            matching_weight=matching_weight,
            return_positive_improve=True,
        )

    if local_rank == 0 and writer is not None:
        writer.add_scalar('EditEval/FID', fid, ep)
        writer.add_scalar('EditEval/Matching', match_score, ep)
        writer.add_scalar('EditEval/Structure', struct_score, ep)
        writer.add_scalar('EditEval/Overall', best_overall, ep)
        writer.add_scalar('EditEval/Top1', top1, ep)
        writer.add_scalar('EditEval/Top2', top2, ep)
        writer.add_scalar('EditEval/Top3', top3, ep)
        writer.add_scalar('EditEval/PositiveImprove', positive_improve, ep)

    return fid, match_score, struct_score, best_overall, top1, top2, top3, positive_improve


def evaluate_gen(
        out_dir,
        val_loader,
        unimoflow_model,
        vae_model,
        eval_wrapper,
        writer,
        ep,
        tracker: UniMoFlowMetricsTracker,
        device,
        plot_func=None,
        num_denoise_steps=10,
        cfg_scale=5.0,
        save_ckpt=False,
        save_anim=False,
        distributed=False,
        world_size=1,
        local_rank=0,
):
    """
    生成模式评估：封装 evaluation_diffusion_model，兼容混合模型的 generate 接口

    返回: (fid, diversity, top1, top2, top3, matching)
    """
    raw_model = get_raw_model(unimoflow_model)

    # evaluation_diffusion_model 内部会调用 model.generate()
    # 我们的 model.generate() 会根据输入自动路由到 generate_gen
    # 返回值: (best_fid, best_div, best_top1, best_top2, best_top3, best_matching)
    fid, diversity, top1, top2, top3, matching_score = evaluation_diffusion_model(
        out_dir=out_dir,
        val_loader=val_loader,
        diffusion_model=raw_model,
        vae_model=vae_model.module if isinstance(vae_model, DDP) else vae_model,
        writer=writer,
        ep=ep,
        best_fid=tracker.gen_best_fid,
        best_div=tracker.gen_best_div,
        best_top1=tracker.gen_best_top1,
        best_top2=tracker.gen_best_top2,
        best_top3=tracker.gen_best_top3,
        best_matching=tracker.gen_best_matching,
        eval_wrapper=eval_wrapper,
        device=device,
        plot_func=plot_func,
        num_denoise_steps=num_denoise_steps,
        cfg_scale=cfg_scale,
        save_ckpt=False,
        save_anim=save_anim,
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )

    if local_rank == 0 and writer is not None:
        writer.add_scalar('GenEval/FID', fid, ep)
        writer.add_scalar('GenEval/Diversity', diversity, ep)
        writer.add_scalar('GenEval/Top1', top1, ep)
        writer.add_scalar('GenEval/Top2', top2, ep)
        writer.add_scalar('GenEval/Top3', top3, ep)
        writer.add_scalar('GenEval/Matching', matching_score, ep)

    return fid, diversity, top1, top2, top3, matching_score
