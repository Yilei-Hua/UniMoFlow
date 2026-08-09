import collections

import torch
import numpy as np
from torch.utils import data
from os.path import join as pjoin
import random
from tqdm import tqdm
import json

# from utils.paramUtil import style_enumerator, style_inv_enumerator
class CommonMotionDataset(data.Dataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path):
        self.cfg = cfg
        mid_list = []
        cid_list = []
        total_frames = 0

        data_dict = {}

        with open(mid_list_path, "r") as f:
            for line in f.readlines():
                mid_list.append(line.strip())

        with open(cid_list_path, "r") as f:
            for line in f.readlines():
                cid = line.strip()
                _, start, end = cid.split("#")

                if int(end) - int(start) >= cfg.data.min_motion_length:
                    cid_list.append(cid)
                    total_frames += int(end) - int(start)

        # for fid in fids_list:

        total_count = len(cid_list)

        for i, mid in tqdm(enumerate(mid_list)):
            data_path = pjoin(cfg.data.feat_dir, "%s.npy" % mid)
            data = np.load(data_path)
            data_dict[mid] = data

        # if cfg.is_train and (not fix_bias):
        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.cfg = cfg
        self.mid_list = mid_list
        self.cid_list = cid_list

        print(
            "Loading %d motions, %d frames, %03f hours"
            % (total_count, total_frames, total_frames / 30.0 / 60.0 / 60.0)
        )
        # print("Loading %d style motions, %d style frames, %03f hours"%(num_style_motions, total_style_frames, total_style_frames/30./60./60.))

    def inv_transform(self, data):
        if isinstance(data, np.ndarray):
            return data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        elif isinstance(data, torch.Tensor):
            return data * torch.from_numpy(self.std[:data.shape[-1]]).float().to(
                data.device
            ) + torch.from_numpy(self.mean[:data.shape[-1]]).float().to(data.device)
        else:
            raise TypeError("Expected data to be either np.ndarray or torch.Tensor")

    def __len__(self):
        return len(self.cid_list)

    def __getitem__(self, item):
        cid = self.cid_list[item]
        mid, start, end = cid.split("#")
        motion = self.data_dict[mid][int(start) : int(end)]

        # Z Normalization
        motion_data = (motion - self.mean) / self.std

        # print(self.std)
        return motion_data, cid


class TextMotionDataset(CommonMotionDataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, all_caption_path):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path)

        with open(all_caption_path, "r") as f:
            self.all_captions = json.load(f)

    def __getitem__(self, item):
        motion, cid = super().__getitem__(item)
        captions = self.all_captions[cid]["manual"] + self.all_captions[cid]["gpt"]
        caption = random.choice(captions)
        m_length = (
            len(motion)
            if len(motion) < self.cfg.data.max_motion_length
            else self.cfg.data.max_motion_length
        )

        # coin2 = np.random.choice(["single", "single", "double"])
        # if coin2 == "double":
        #     m_length = (
        #         m_length // self.cfg.data.unit_length - 1
        #     ) * self.cfg.data.unit_length
        # else:
        m_length = (
                m_length // self.cfg.data.unit_length
            ) * self.cfg.data.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        if m_length < self.cfg.data.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (self.cfg.data.max_motion_length - m_length, motion.shape[1])
                    ),
                ],
                axis=0,
            )
        return caption, motion, m_length

class MotionDataset(CommonMotionDataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path)
        lengths = [0]
        n_cid_list = []
        for cid in self.cid_list:
            _, start, end = cid.split("#")
            length = int(end) - int(start) - self.cfg.data.motion_length
            if length >= 0:
                lengths.append(length)
                n_cid_list.append(cid)

        self.cid_list = n_cid_list
        self.cumsum = np.cumsum(lengths)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        cid_idx = np.searchsorted(self.cumsum, item + 1) - 1
        # cid =
        idx = item - self.cumsum[cid_idx]
        motion, _ = super().__getitem__(cid_idx)
        motion_clip = motion[idx : idx + self.cfg.data.motion_length]

        return motion_clip

class WindowedMotionDataset(CommonMotionDataset):
    """
    支持窗口化采样的动作数据集
    - 训练时：随机截取 window_length 长度的动作片段
    - 验证/测试时：提取足够长度用于重建评估（类似 TextMotionDataset）
    """

    def __init__(
        self,
        cfg,
        mean,
        std,
        mid_list_path,
        cid_list_path,
        all_caption_path,
        split="train"
    ):
        # 先调用父类初始化加载基础数据
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path)

        self.split = split
        self.cfg = cfg

        # 根据 split 设置参数
        if split == "train":
            self.window_length = cfg.data.get("window_length", cfg.data.motion_length)
            self.random_crop = True
            # 可选：是否使用随机长度（类似 TextMotionDataset 的 coin2 逻辑）
            self.use_variable_length = cfg.data.get("use_variable_length", False)
        else:  # val or test
            self.window_length = cfg.data.max_motion_length
            self.random_crop = False
            self.use_variable_length = False

        # 加载文本数据
        with open(all_caption_path, "r") as f:
            self.all_captions = json.load(f)

    def __getitem__(self, item):
        # 获取基础动作数据（从 CommonMotionDataset）
        # motion, cid = super().__getitem__(item)
        cid = self.cid_list[item]
        mid, start, end = cid.split("#")
        motion = self.data_dict[mid][int(start):int(end)]
        original_length = len(motion)

        # 获取文本描述
        captions = self.all_captions[cid]["manual"] + self.all_captions[cid]["gpt"]
        caption = random.choice(captions)

        # 根据 split 决定如何处理动作长度
        if self.split == "train":
            # 训练模式：随机窗口裁剪
            motion_processed, m_length = self._train_process(motion)
        else:
            # 验证/测试模式：提取足够长度用于重建
            motion_processed, m_length = self._eval_process(motion)

        # Z 标准化（父类已经完成了标准化，但这里我们处理的是裁剪后的片段）
        motion_data = (motion_processed - self.mean) / self.std
        return caption, motion_data, m_length

    def _train_process(self, motion):
        """训练时处理：随机裁剪到 window_length"""
        motion_length = len(motion)

        # 如果 motion 比 window 长，随机裁剪
        if motion_length > self.window_length:
            start_idx = random.randint(0, motion_length - self.window_length)
            motion = motion[start_idx:start_idx + self.window_length]
            m_length = self.window_length
        else:
            # 如果 motion 比 window 短，保留原长（后续会 padding）
            m_length = motion_length

        # 可选：类似 TextMotionDataset 的 unit_length 对齐
        unit_length = self.cfg.data.get("unit_length", 1)
        if self.use_variable_length and random.random() < 0.5:
            # 随机缩短长度，但保持 unit_length 对齐（类似 double 逻辑）
            m_length = (m_length // unit_length - 1) * unit_length if m_length > unit_length else m_length
        else:
            m_length = (m_length // unit_length) * unit_length

        # 随机选择起始点（在可用范围内）
        if m_length < len(motion):
            start_idx = random.randint(0, len(motion) - m_length)
            motion = motion[start_idx:start_idx + m_length]
        return motion, m_length

    def _eval_process(self, motion):
        """验证/测试时处理：与 TextMotionDataset 完全一致"""
        # 步骤1：确定有效长度（与 TextMotionDataset 完全一致）
        m_length = (
            len(motion)
            if len(motion) < self.cfg.data.max_motion_length
            else self.cfg.data.max_motion_length
        )

        # 步骤2：unit_length 对齐（向下取整）
        unit_length = self.cfg.data.get("unit_length", 1)
        m_length = (m_length // unit_length) * unit_length

        # 步骤3：随机选择起始点（与 TextMotionDataset 一致）
        if m_length < len(motion):
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx: idx + m_length]
        else:
            motion = motion[:m_length]

        # 步骤4：Padding 到 max_motion_length（与 TextMotionDataset 一致）
        if m_length < self.cfg.data.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (self.cfg.data.max_motion_length - m_length, motion.shape[1])
                    ),
                ],
                axis=0,
            )

        return motion, m_length

class LatentTextMotionDataset(data.Dataset):
    def __init__(self, cfg, split_file, caption_path, latent_dir="/latents_hrvae_detail"):
        """
        Args:
            cfg: 配置对象
            split_file: 包含 data IDs 的 txt 文件路径 (对应原 train_ids.txt 或 val_ids.txt)
            caption_path: all_caption_clean.json 路径
            latent_dir: 预处理好的 latent .npy 文件所在目录
        """
        self.cfg = cfg
        tag = split_file.split('/')[-1].split('_')[0]
        self.latent_dir = cfg.data.root_dir + latent_dir + "/" + tag
        # 1. 读取 ID 列表
        # 假设 split_file 里每一行就是一个 cid (例如: 000012#000#120)
        # 这些 ID 应该与预处理时保存的文件名一致
        with open(split_file, "r") as f:
            self.id_list = [line.strip() for line in f.readlines()]

        # 2. 加载文本描述
        with open(caption_path, "r") as f:
            self.all_captions = json.load(f)

        # 3. 计算 Latent 空间的最大长度
        # 假设 VAE 下采样倍率为 4 (HumanML3D/Kit 的标准 VAE)
        # 如果你的 VAE 倍率不同，请修改这个 downsample_ratio
        self.downsample_ratio = 4
        self.max_latent_length = self.cfg.data.max_motion_length // self.downsample_ratio
        # 空间维度 P（固定部位/部件数），默认为 1 兼容旧版 2D latent (L, D)
        self.spatial_dim = getattr(cfg.model, 'spatial_dim', 1)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        # 获取 ID (对应文件名)
        cid = self.id_list[idx]

        # 1. 加载 Latent 数据
        # 路径: latent_dir/cid.npy
        # 支持 shape: [L, D] 或 [L, P, D]
        latent_path = pjoin(self.latent_dir, f"{cid}.npy")
        try:
            latent = np.load(latent_path)
        except FileNotFoundError:
            # 容错处理：如果找不到文件，随机返回一个由零组成的 Dummy
            # 根据 spatial_dim 构造对应维度
            if self.spatial_dim > 1:
                latent = np.zeros((self.max_latent_length, self.spatial_dim, self.cfg.model.input_dim))
            else:
                latent = np.zeros((self.max_latent_length, self.cfg.model.input_dim))

        # 统一为 3D (L, P, D)：兼容旧数据 (L, D) -> (L, 1, D)
        if latent.ndim == 2:
            latent = latent[:, np.newaxis, :]

        # 如果实际存储的 P 与配置不一致，做自适应 pad / 截断
        if latent.shape[1] != self.spatial_dim:
            if latent.shape[1] < self.spatial_dim:
                pad_width = [(0, 0), (0, self.spatial_dim - latent.shape[1]), (0, 0)]
                latent = np.pad(latent, pad_width, mode='constant')
            else:
                latent = latent[:, :self.spatial_dim, :]

        # 2. 获取文本 (逻辑与 TextMotionDataset 一致)
        captions = self.all_captions[cid]["manual"] + self.all_captions[cid]["gpt"]
        caption = random.choice(captions)

        # 3. 长度处理 (Crop & Pad) —— 仅在时间维度 L 上操作
        seq_len = latent.shape[0]

        if seq_len >= self.max_latent_length:
            # 随机裁剪 (Random Crop)
            start = random.randint(0, seq_len - self.max_latent_length)
            latent = latent[start: start + self.max_latent_length]
            out_len = self.max_latent_length
        else:
            # 填充 (Padding) —— 在时间轴 (axis 0) 后面补零，保留 (P, D) 结构
            padding_len = self.max_latent_length - seq_len
            latent = np.concatenate(
                [latent, np.zeros((padding_len, latent.shape[1], latent.shape[2]))], axis=0
            )
            out_len = seq_len

        # 转为 Tensor
        latent_tensor = torch.from_numpy(latent).float()

        # 返回: caption, latent_tensor, latent_length
        return caption, latent_tensor, out_len

# Collate function for DataLoader
def windowed_motion_collate_fn(batch):
    """批处理函数，处理变长数据"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    # 堆叠 motion 张量
    motions = [torch.from_numpy(b["motion"]) for b in batch]
    output["motion"] = torch.stack(motions, dim=0)

    # 长度信息
    output["motion_length"] = torch.tensor([b["motion_length"] for b in batch])
    output["original_length"] = [b["original_length"] for b in batch]

    # 文本信息
    output["caption"] = [b["caption"] for b in batch]
    output["cid"] = [b["cid"] for b in batch]
    return output
