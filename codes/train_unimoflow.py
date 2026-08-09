import os
import argparse
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from models_flow.unimoflow import UniMoFlow
from trainers.unimoflow_trainer import UniMoFlowTrainer
from config.load_config import load_config
from dataset.edit_dataset import MotionEditDataset, collate_fn
from dataset.dataset import LatentTextMotionDataset, TextMotionDataset
from utils.paramUtil import kinematic_chain
from utils import bvh_io
from common.skeleton import Skeleton
from utils.motion_process_bvh import recover_pos_from_rot
from models_flow.hrvae import HRVAE
import shutil
import traceback
import signal
from datetime import timedelta
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image


def plot_3d_motion_safe(save_path, kinematic_chain, joints, title="", fps=30, radius=100):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joints = np.array(joints)
    if joints.shape[-1] == 3:
        joints = joints.reshape(joints.shape[0], -1, 3)
    M, N = joints.shape[:2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    data_mean = joints.mean(axis=(0, 1))
    data_max = np.abs(joints - data_mean).max()

    def init():
        ax.set_xlim3d([data_mean[0] - radius, data_mean[0] + radius])
        ax.set_ylim3d([data_mean[1] - radius, data_mean[1] + radius])
        ax.set_zlim3d([0, radius * 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        return []

    def animate(i):
        ax.clear()
        init()
        for chain in kinematic_chain:
            x = [joints[i, j, 0] for j in chain]
            y = [joints[i, j, 1] for j in chain]
            z = [joints[i, j, 2] for j in chain]
            ax.plot(x, y, z, marker='o', markersize=3, linewidth=2)
        return []

    ani = FuncAnimation(fig, animate, frames=M, init_func=init, interval=1000 / fps, blit=False)
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
    except Exception as e:
        print(f"Error saving animation to {save_path}: {e}")
        try:
            img_dir = save_path.replace('.mp4', '_frames')
            os.makedirs(img_dir, exist_ok=True)
            for i in range(min(M, 10)):
                animate(i)
                fig.savefig(pjoin(img_dir, f'frame_{i:04d}.png'))
            print(f"Saved debug frames to {img_dir}")
        except Exception as e2:
            print(f"Failed to save frames: {e2}")
    plt.close(fig)


def plot_edit_t2m(data, save_dir, captions, m_lengths, train_dataset, prefix=""):
    try:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        global_pos = forward_kinematic_func(data, train_dataset)
        if isinstance(global_pos, torch.Tensor):
            global_pos = global_pos.detach().cpu().numpy()
        for i in range(len(global_pos)):
            length = min(int(m_lengths[i]), global_pos.shape[1])
            motion_data = global_pos[i, :length]
            filename = f'{prefix}{i:02d}.mp4' if prefix else f'{i:02d}.mp4'
            save_path = pjoin(save_dir, filename)
            plot_3d_motion_safe(
                save_path, kinematic_chain, motion_data,
                title=captions[i] if i < len(captions) else "",
                fps=30, radius=100
            )
    except Exception as e:
        print(f"Error in plot_edit_t2m for dir {save_dir}: {e}")
        import traceback
        traceback.print_exc()


def forward_kinematic_func(data, train_dataset):
    device = train_dataset.device if hasattr(train_dataset, 'device') else torch.device('cuda')
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().to(device)
    elif isinstance(data, torch.Tensor):
        data = data.float().to(device)
    if hasattr(train_dataset, 'inv_transform'):
        motions = train_dataset.inv_transform(data)
    else:
        motions = data
    if hasattr(train_dataset, 'skeleton'):
        global_pos = recover_pos_from_rot(
            motions,
            joints_num=getattr(train_dataset.cfg.data, 'joint_num', 24),
            skeleton=train_dataset.skeleton
        )
        return global_pos
    return motions


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
    ckpt = torch.load(
        cfg.vae_checkpoint,
        map_location=device, weights_only=True
    )
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vae.load_state_dict(ckpt[model_key])
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f'Loading VAE Model {vae_cfg.exp.name} from epoch {ckpt["ep"]}')
    vae.to(device)
    vae.eval()
    if dist.is_initialized() and dist.get_world_size() > 1:
        vae = DDP(vae, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    return vae, vae_cfg


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
        timeout = dist.default_pg_timeout if hasattr(dist, 'default_pg_timeout') else \
            dist.InitProcessGroupKwargs(timeout=timedelta(seconds=60))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout if isinstance(timeout, timedelta) else None
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (local_rank == 0)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        local_rank = 0
    return local_rank, world_size, device, is_main_process


def emergency_abort(local_rank, world_size, exit_code=1):
    print(f"[Rank {local_rank}] Emergency abort...")
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    try:
        torch.cuda.synchronize()
    except:
        pass
    os._exit(exit_code)


def main(config_path=None):
    local_rank, world_size, device, is_main_process = setup_distributed()

    def signal_handler(sig, frame):
        print(f"\n[Rank {local_rank}] Interrupted by signal {sig}")
        emergency_abort(local_rank, world_size)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        config_path = config_path or 'configs/unimoflow.yaml'
        cfg = load_config(config_path)
        cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'unimoflow', cfg.exp.name)

        if is_main_process:
            if cfg.exp.is_continue:
                n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, 'train_unimoflow.yaml'))
                n_cfg.exp.is_continue = True
                n_cfg.exp.device = cfg.exp.device
                n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
                cfg = n_cfg
            else:
                os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
                shutil.copy(config_path, pjoin(cfg.exp.checkpoint_dir, 'train_unimoflow.yaml'))
                if os.path.basename(config_path) != 'train_unimoflow.yaml':
                    shutil.copy(config_path, pjoin(cfg.exp.checkpoint_dir, os.path.basename(config_path)))

        if world_size > 1:
            dist.barrier()

        cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
        cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
        cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'unimoflow', cfg.exp.name)

        if is_main_process:
            os.makedirs(cfg.exp.model_dir, exist_ok=True)
            os.makedirs(cfg.exp.eval_dir, exist_ok=True)
            os.makedirs(cfg.exp.log_dir, exist_ok=True)

        data_root = cfg.data.root_dir
        meta_dir = pjoin(data_root, 'meta_data')

        mean = np.load(pjoin(meta_dir, 'mean.npy'))
        std = np.load(pjoin(meta_dir, 'std.npy'))

        vae, vae_cfg = load_vae(cfg, device=device)

        template_anim = bvh_io.load(pjoin(data_root, 'renamed_bvhs', 'm_ep2_00086.bvh'))
        skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)

        # 初始化混合模型
        unimoflow_model = UniMoFlow(
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
            dropout_prob=cfg.model.dropout_prob,
            noise_steps=cfg.model.noise_steps,
            cfg_scale=cfg.model.cfg_scale,
            time_scale=cfg.model.get('time_scale', 1.0),
            prediction_type=cfg.model.prediction_type,
            use_logit_normal=cfg.model.get('use_logit_normal', False),
            fusion_schedule=cfg.model.get('fusion_schedule', 'asymmetric'),
            param_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            use_role_tags=True,
            spatial_dim=1,
            use_dynamic_depth=False,
            gen_loss_weight=cfg.training.get('gen_loss_weight', 1.0),
            edit_loss_weight=cfg.training.get('edit_loss_weight', 1.0),
        )

        unimoflow_model.to(device)

        if is_main_process:
            model_to_count = unimoflow_model.module if isinstance(unimoflow_model, DDP) else unimoflow_model
            pc = sum(param.numel() for param in model_to_count.parameters())
            print(f"UniMoFlow architecture:\n{model_to_count}")
            print(f'Total parameters: {pc / 1e6:.2f}M')
            print(f"World Size: {world_size}")
            print(f"Running on Device: {device}")

        # Trainer初始化
        trainer = UniMoFlowTrainer(
            cfg, unimoflow_model, vae=vae, device=device,
            local_rank=local_rank, world_size=world_size,
            gen_prob=cfg.training.get('gen_prob', 0.5),
            mixing_strategy=cfg.training.get('mixing_strategy', 'probabilistic'),
            geo_loss=False,
        )

        # ========== 编辑数据集 ==========
        train_edit_dataset = MotionEditDataset(
            data_root=data_root,
            split="train",
            data_files=cfg.data.get('edit_data_files', []),
            cycle_aug_prob=cfg.data.get('cycle_aug_prob', 0.5),
            caption_as_edit_prob=0.2,
            max_length=cfg.data.max_motion_length,
            mean=mean,
            std=std,
        )

        val_edit_dataset = MotionEditDataset(
            data_root=data_root,
            split="val",
            data_files=cfg.data.get('edit_val_files', []),
            cycle_aug_prob=0.0,
            max_length=cfg.data.max_motion_length,
            mean=mean,
            std=std,
        )

        # ========== 生成数据集 ==========
        cfg.data.feat_dir = pjoin(data_root, 'renamed_feats')
        data_split_dir = pjoin(data_root, 'data_split_info')
        all_caption_path = pjoin(data_root, 'all_caption_clean.json')
        train_cid_split_file = pjoin(data_split_dir, 'train_ids.txt')
        val_cid_split_file = pjoin(data_split_dir, 'val_ids.txt')
        val_mid_split_file = pjoin(data_split_dir, 'val_fnames.txt')

        train_gen_dataset = LatentTextMotionDataset(
            cfg, train_cid_split_file, all_caption_path,
            latent_dir=cfg.data.get('gen_latent_dir', "/latents_hrvae_detail"),
        )

        val_gen_dataset = LatentTextMotionDataset(
            cfg, val_cid_split_file, all_caption_path,
            latent_dir=cfg.data.get('gen_latent_dir', "/latents_hrvae_detail"),
        )

        # ========== Samplers ==========
        if world_size > 1:
            train_edit_sampler = DistributedSampler(train_edit_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
            val_edit_sampler = DistributedSampler(val_edit_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
            train_gen_sampler = DistributedSampler(train_gen_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
            val_gen_sampler = DistributedSampler(val_gen_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        else:
            train_edit_sampler = val_edit_sampler = None
            train_gen_sampler = val_gen_sampler = None

        # ========== DataLoaders ==========
        train_edit_loader = DataLoader(
            train_edit_dataset,
            batch_size=cfg.training.batch_size,
            drop_last=True,
            num_workers=cfg.training.get('num_workers', 4),
            shuffle=(train_edit_sampler is None),
            sampler=train_edit_sampler,
            pin_memory=True,
            persistent_workers=True if train_edit_sampler else False,
            collate_fn=collate_fn,
            pin_memory_device=str(device) if torch.cuda.is_available() else "",
        )

        val_edit_loader = DataLoader(
            val_edit_dataset,
            batch_size=cfg.training.batch_size,
            drop_last=True,
            num_workers=cfg.training.get('num_workers', 4),
            shuffle=False,
            sampler=val_edit_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
            pin_memory_device=str(device) if torch.cuda.is_available() else "",
        )

        train_gen_loader = DataLoader(
            train_gen_dataset,
            batch_size=cfg.training.batch_size,
            drop_last=True,
            num_workers=cfg.training.get('num_workers', 4),
            shuffle=(train_gen_sampler is None),
            sampler=train_gen_sampler,
            pin_memory=True,
            persistent_workers=True if train_gen_sampler else False,
            pin_memory_device=str(device) if torch.cuda.is_available() else "",
        )

        # ========== 评估配置 ==========
        eval_cfg = load_config(cfg.evaluator.config_path)
        eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

        eval_loader = DataLoader(
            val_edit_dataset,
            batch_size=cfg.evaluator.get('batch_size', cfg.training.batch_size),
            drop_last=False,
            num_workers=cfg.training.get('num_workers', 4),
            shuffle=False,
            sampler=val_edit_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
            pin_memory_device=str(device) if torch.cuda.is_available() else "",
        )

        val_edit_dataset.skeleton = skeleton
        val_edit_dataset.cfg = cfg
        val_edit_dataset.device = device

        plot_eval = partial(plot_edit_t2m, train_dataset=val_edit_dataset)

        # ========== 生成模式验证/评估 DataLoaders（用于 T2M 评估）==========
        # 验证 loss 使用 LatentTextMotionDataset（32-dim latents，与训练一致）
        val_gen_loader = DataLoader(
            val_gen_dataset,
            batch_size=cfg.training.batch_size,
            drop_last=False,
            num_workers=cfg.training.get('num_workers', 4),
            shuffle=False,
            sampler=val_gen_sampler,
            pin_memory=True,
            pin_memory_device=str(device) if torch.cuda.is_available() else "",
        )

        # 综合评估使用 TextMotionDataset（148-dim 原始动作空间），
        # 因为 evaluation_diffusion_model 内部用 evaluator 编码 motions[..., :148]
        gen_eval_dataset = TextMotionDataset(cfg, mean, std, val_mid_split_file, val_cid_split_file, all_caption_path)

        if world_size > 1:
            gen_eval_sampler = DistributedSampler(gen_eval_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        else:
            gen_eval_sampler = None

        gen_eval_loader = DataLoader(
            gen_eval_dataset,
            batch_size=cfg.evaluator.get('batch_size', cfg.training.batch_size),
            drop_last=False,
            num_workers=cfg.training.get('num_workers', 4),
            shuffle=False,
            sampler=gen_eval_sampler,
            pin_memory=True,
            pin_memory_device=str(device) if torch.cuda.is_available() else "",
        )

        # 生成模式可视化函数（仅解码并保存视频）
        gen_eval_dataset.skeleton = skeleton
        gen_eval_dataset.cfg = cfg
        gen_plot_eval = partial(plot_edit_t2m, train_dataset=gen_eval_dataset)

        print(f"[Rank {local_rank}/{world_size}] DataLoaders ready. "
              f"Edit Train: {len(train_edit_dataset)}, Gen Train: {len(train_gen_dataset)}, "
              f"Val Edit: {len(val_edit_dataset)}, Val Gen: {len(val_gen_dataset)}")

        # 开始混合训练
        trainer.train(
            edit_loader=train_edit_loader,
            gen_loader=train_gen_loader,
            val_loader=val_edit_loader,
            eval_val_loader=eval_loader,
            eval_wrapper=eval_wrapper,
            plot_eval=plot_eval,
            gen_val_loader=val_gen_loader,
            gen_eval_val_loader=gen_eval_loader,
            gen_plot_eval=gen_plot_eval,
        )

    except Exception as e:
        print(f"\n[Rank {local_rank}] Training failed:")
        traceback.print_exc()
        emergency_abort(local_rank, world_size)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/unimoflow.yaml')
    args = parser.parse_args()
    main(args.config)
