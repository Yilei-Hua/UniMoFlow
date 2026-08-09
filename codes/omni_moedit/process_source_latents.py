import sys as _release_sys
from pathlib import Path as _ReleasePath

_CODES_ROOT = _ReleasePath(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_CODES_ROOT))

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join as pjoin

from config.load_config import load_config
from dataset.dataset import TextMotionDataset
# from models_flow.ST_VAE import HRVAE
from models_flow.hrvae import HRVAE
import utils.bvh_io as bvh_io
from common.skeleton import Skeleton


def load_vae(cfg, device):
    # 复用 train_diffusion.py 中的逻辑
    vae_cfg = load_config(cfg.vae_config)
    # vae_cfg = load_config(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vae', cfg.vae_name, 'stvae.yaml'))
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
    # vae = HRVAE(
    #     # cfg=cfg,
    #     input_width=vae_cfg.data.dim_pose,
    #     z_dim=vae_cfg.model.z_dim,
    #     dim=vae_cfg.model.dim,
    #     dec_dim=vae_cfg.model.dec_dim,
    #     num_res_blocks=vae_cfg.model.num_res_blocks,
    #     dropout=vae_cfg.model.dropout,
    #     dim_mult=vae_cfg.model.dim_mult,
    #     temperal_downsample=vae_cfg.model.temperal_downsample,
    #     num_joints=24,  # 内部关节粒度（不暴露给外部索引）
    #     joint_dim=16,  # 每个关节的特征维度
    #     num_parts=6,
    # )
    ckpt = torch.load(
        cfg.vae_checkpoint,
        map_location=device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vae.load_state_dict(ckpt[model_key])
    vae.to(device)
    vae.eval()
    return vae, vae_cfg


@torch.no_grad()
def process_split(split_name, dataset, vae, device, save_root):
    print(f"Processing {split_name} split...")
    save_dir = pjoin(save_root, split_name)
    os.makedirs(save_dir, exist_ok=True)

    # 必须使用 shuffle=False，这样 DataLoader 的迭代顺序才与 dataset.cid_list 一致
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 1. 既然 dataset.py 中明确有 cid_list，我们直接获取引用
    if hasattr(dataset, 'cid_list'):
        id_list = dataset.cid_list
    else:
        raise ValueError("Dataset does not have 'cid_list' attribute!")

    for i, batch in enumerate(tqdm(loader)):
        # 2. 根据 TextMotionDataset.__getitem__ 的返回内容解包
        # 返回值是: caption, motion, m_length
        texts, motions, m_lengths = batch

        motions = motions.to(device).float()
        m_lengths = m_lengths.to(device).long()

        # 3. 获取 dim_pose
        # dataset.py 中没有 self.dim_pose，但有 self.cfg
        # 或者直接使用 motions 的最后一维，通常 VAE 需要特定的输入维度
        dim_pose = dataset.cfg.data.dim_pose

        # VAE Encode
        # [Batch, Length, Dim] -> [Batch, Latent_Length, Latent_Dim]
        latents = vae.encode(motions[..., :dim_pose], m_lengths)
        # 转回 CPU Numpy
        latents = latents[:, :m_lengths[0]//vae.downsample_factor, :]
        latent_np = latents.cpu().numpy()[0]  # 取 batch 第一个 (BatchSize=1)

        # 4. 获取文件名 ID
        # 因为 shuffle=False，当前的 index i 对应 cid_list[i]
        cid = id_list[i]
        # 你的 cid 格式通常是 "000012#000#120"
        # 直接用 cid 作为文件名，方便 LatentTextMotionDataset 读取
        file_id = cid
        # 保存
        save_path = pjoin(save_dir, f"{file_id}.npy")
        np.save(save_path, latent_np)


if __name__ == "__main__":
    # 1. 配置加载
    cfg = load_config('../configs/omni_moedit_filter.yaml')
    device = torch.device(cfg.exp.device if torch.cuda.is_available() else 'cpu')

    # 路径设置
    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

    # 2. 加载 VAE
    vae, vae_cfg = load_vae(cfg, device)

    # 3. 定义输出路径
    # 例如保存到: ../data/SnapMoGen/latents_hrvae
    output_root = pjoin(cfg.data.root_dir, f"latents_{cfg.vae_name}")
    os.makedirs(output_root, exist_ok=True)
    print(f"Latents will be saved to: {output_root}")

    # 4. 处理 Train Set
    train_mid_split_file = pjoin(data_split_dir, 'train_fnames.txt')
    train_cid_split_file = pjoin(data_split_dir, 'train_ids.txt')
    train_dataset = TextMotionDataset(cfg, mean, std, train_mid_split_file, train_cid_split_file, all_caption_path)

    # 关键：手动给 Dataset 注入 id_list 属性，以便上面函数能读到
    # 这一步依赖于 dataset 是如何读取 split_file 的
    # 通常 dataset 会读取 split_file 每一行作为一个 ID
    with open(train_mid_split_file, 'r') as f:
        train_dataset.id_list = [line.strip() for line in f.readlines()]

    process_split('train', train_dataset, vae, device, output_root)

    # 5. 处理 Val Set
    val_mid_split_file = pjoin(data_split_dir, 'val_fnames.txt')
    val_cid_split_file = pjoin(data_split_dir, 'val_ids.txt')
    val_dataset = TextMotionDataset(cfg, mean, std, val_mid_split_file, val_cid_split_file, all_caption_path)

    with open(val_mid_split_file, 'r') as f:
        val_dataset.id_list = [line.strip() for line in f.readlines()]

    process_split('val', val_dataset, vae, device, output_root)

    test_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    test_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')
    test_dataset = TextMotionDataset(cfg, mean, std, test_mid_split_file, test_cid_split_file, all_caption_path)

    with open(test_mid_split_file, 'r') as f:
        test_dataset.id_list = [line.strip() for line in f.readlines()]

    process_split('test', test_dataset, vae, device, output_root)
    print("Preprocessing Done!")
