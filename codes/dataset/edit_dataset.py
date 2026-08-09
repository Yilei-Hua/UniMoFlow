# dataset/edit_dataset.py
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


class MotionEditDataset(Dataset):
    """
    动作编辑数据集（多源版本）

    支持多种数据源：
    1. data_files: 多个JSON文件路径列表，每个文件包含编辑对数据
       格式示例： [{"original_key": "cap_000006", "source_path": "...", "edited_path": "...",
                   "source_caption": "...", "target_caption": "...", "edit_command": "...", ...}, ...]
    2. data_list: 直接传入数据列表（内存中已加载的数据）
    3. 向后兼容：保留 accepted_pairs_file 和 real_pairs_file 参数

    支持循环一致性增强：随机交换源/目标，并替换为逆命令
    支持Caption增强：以一定概率直接使用target caption作为edit command
    """

    def __init__(
            self,
            data_root: str,
            split: str = "train",
            # 新增：支持多源数据文件
            data_files: Optional[List[str]] = None,  # 多个JSON文件路径列表
            data_list: Optional[List[Dict]] = None,  # 直接传入数据列表
            # 向后兼容
            accepted_pairs_file: Optional[str] = None,
            real_pairs_file: Optional[str] = None,
            # 配置参数
            cycle_aug_prob: float = 0.5,
            caption_as_edit_prob: float = 0.0,
            max_length: int = 320,
            mean=None, std=None,
            prioritize_real_pairs: bool = False,
            # 数据筛选
            min_metrics: Optional[Dict[str, float]] = None,  # 根据metrics筛选，如 {"r_align_edit": 0.7}
            only_accepted: bool = True,  # 只加载 status == "accepted" 的数据
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.cycle_aug_prob = cycle_aug_prob if split == "train" else 0.0
        self.caption_as_edit_prob = caption_as_edit_prob if split == "train" else 0.0
        self.max_length = max_length // 4
        self.mean = mean
        self.std = std
        self.prioritize_real_pairs = prioritize_real_pairs

        # 筛选条件
        self.min_metrics = min_metrics or {}
        self.only_accepted = only_accepted

        all_pairs = []

        # ========== 1. 处理新格式的多源数据文件 ==========
        if data_files is not None:
            for file_idx, data_file in enumerate(data_files):
                # 支持绝对路径或相对于 data_root 的路径
                file_path = Path(data_file) if Path(data_file).is_absolute() else self.data_root / data_file

                if not file_path.exists():
                    print(f"[Warning] data_file not found: {file_path}")
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 统一转换为列表格式
                    if isinstance(data, dict):
                        # 如果JSON是字典格式（如 {"cap_000006": {...}, ...}）
                        data = list(data.values())
                    elif not isinstance(data, list):
                        print(f"[Warning] Unsupported data format in {file_path}, expected list or dict")
                        continue

                    # 为每个数据添加来源标识
                    source_tag = f"file_{file_idx}_{file_path.stem}"
                    loaded_count = 0
                    filtered_count = 0

                    for item in data:
                        if not isinstance(item, dict):
                            continue

                        # 检查 split
                        if item.get("split", "train") != split:
                            continue

                        # 检查 status (如果 only_accepted=True)
                        if self.only_accepted and item.get("status", "accepted") != "accepted":
                            filtered_count += 1
                            continue

                        # 检查 metrics 条件
                        metrics = item.get("metrics", {})
                        metrics_pass = all(
                            metrics.get(k, float('-inf')) >= v
                            for k, v in self.min_metrics.items()
                        )
                        if not metrics_pass:
                            filtered_count += 1
                            continue

                        # 添加元数据
                        item["_data_source"] = source_tag
                        item["_file_idx"] = file_idx

                        # 确保必要字段存在（兼容处理）
                        self._normalize_item(item)

                        all_pairs.append(item)
                        loaded_count += 1

                    print(f"[MotionEditDataset] Loaded {loaded_count} pairs from {file_path.name} "
                          f"(filtered {filtered_count}, tag: {source_tag})")

                except Exception as e:
                    print(f"[Error] Failed to load {file_path}: {e}")

        # ========== 2. 处理直接传入的数据列表 ==========
        if data_list is not None:
            loaded_count = 0
            for item in data_list:
                if item.get("split", "train") != split:
                    continue
                if self.only_accepted and item.get("status", "accepted") != "accepted":
                    continue

                item["_data_source"] = "data_list"
                self._normalize_item(item)
                all_pairs.append(item)
                loaded_count += 1

            print(f"[MotionEditDataset] Loaded {loaded_count} pairs from data_list")

        # ========== 3. 向后兼容：处理旧的参数格式 ==========
        # 加载 accepted_pairs（FlowEdit生成）
        if accepted_pairs_file is not None:
            # pairs_path = Path(data_root) / "edit_latents_filtered_new" / accepted_pairs_file
            pairs_path = self.data_root / "edit_filtered_6var" / accepted_pairs_file
            if pairs_path.exists():
                with open(pairs_path, 'r', encoding='utf-8') as f:
                    accepted_pairs = json.load(f)

                loaded_count = 0
                for p in accepted_pairs:
                    if p.get("split", "train") != split:
                        continue
                    if self.only_accepted and p.get("status", "accepted") != "accepted":
                        continue

                    p["_data_source"] = "accepted_pairs"
                    self._normalize_item(p)
                    all_pairs.append(p)
                    loaded_count += 1

                print(f"[MotionEditDataset] Loaded {loaded_count} pairs from {accepted_pairs_file} (legacy)")
            else:
                print(f"[Warning] accepted_pairs_file not found: {pairs_path}")

        # 加载 real_motion_pairs（真实动作对）
        if real_pairs_file is not None:
            real_path = self.data_root / real_pairs_file
            if real_path.exists():
                with open(real_path, 'r', encoding='utf-8') as f:
                    real_pairs = json.load(f)

                loaded_count = 0
                for p in real_pairs:
                    if p.get("split", "train") != split:
                        continue

                    p["_data_source"] = "real_pairs"
                    if "original_key" not in p:
                        p["original_key"] = p.get("source_key", p.get("target_key", "unknown"))
                    self._normalize_item(p)
                    all_pairs.append(p)
                    loaded_count += 1

                print(f"[MotionEditDataset] Loaded {loaded_count} pairs from {real_pairs_file} (legacy)")
            else:
                print(f"[Warning] real_pairs_file not found: {real_path}")

        if len(all_pairs) == 0:
            raise ValueError(f"No data loaded for split '{split}'. Please check file paths.")

        # ========== 4. 去重逻辑（优先保留real_pairs） ==========
        if prioritize_real_pairs:
            seen_keys = {}
            filtered_pairs = []
            for p in all_pairs:
                key = p.get("original_key", str(id(p)))
                if key in seen_keys:
                    # 如果当前是real_pairs，替换已有的
                    if p["_data_source"] == "real_pairs":
                        for i, existing in enumerate(filtered_pairs):
                            if existing.get("original_key") == key:
                                filtered_pairs[i] = p
                                break
                else:
                    seen_keys[key] = True
                    filtered_pairs.append(p)
            all_pairs = filtered_pairs
            print(f"[MotionEditDataset] After deduplication: {len(all_pairs)} pairs")

        self.pairs = all_pairs

        # 统计信息
        self._print_statistics()

    def _normalize_item(self, item: Dict):
        """
        规范化数据项，确保必要字段存在
        处理不同来源的数据格式差异
        """
        # 确保 original_key 存在
        if "original_key" not in item:
            item["original_key"] = item.get("sample_id", item.get("source_key", str(id(item))))

        # ========== 新增：处理 edit_pairs_aligned.json 的字段名 ==========
        # 处理源路径字段映射 (source_motion_path -> source_path/source_latent_path)
        if "source_path" not in item and "source_motion_path" in item:
            item["source_path"] = item["source_motion_path"]
        if "source_latent_path" not in item and "source_motion_path" in item:
            item["source_latent_path"] = item["source_motion_path"]

        # 处理目标路径字段映射 (target_motion_path -> edited_path/target_latent_path)
        if "edited_path" not in item and "target_motion_path" in item:
            item["edited_path"] = item["target_motion_path"]
        if "target_latent_path" not in item and "target_motion_path" in item:
            item["target_latent_path"] = item["target_motion_path"]
        # ==============================================================

        # 统一路径字段（原有的兼容逻辑）
        if "edited_path" not in item and "target_path" in item:
            item["edited_path"] = item["target_path"]
        if "target_latent_path" not in item and "edited_path" in item:
            item["target_latent_path"] = item["edited_path"]
        if "source_latent_path" not in item and "source_path" in item:
            item["source_latent_path"] = item["source_path"]

        # 确保caption字段存在
        if "source_caption" not in item:
            item["source_caption"] = item.get("original_caption", "No source caption")
        if "target_caption" not in item:
            item["target_caption"] = item.get("edited_caption", item.get("new_caption", "No target caption"))

        # 确保edit_command字段存在
        if "edit_command" not in item:
            item["edit_command"] = item.get("edit_text", "Transform the motion")
        if "reverse_edit_command" not in item:
            item["reverse_edit_command"] = item.get("reverse_command", "Revert the changes")

    def _print_statistics(self):
        """打印数据集统计信息"""
        print(f"[MotionEditDataset] Total {len(self.pairs)} pairs for {self.split}")

        # 统计各数据源数量
        source_counts = {}
        for p in self.pairs:
            src = p["_data_source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        if len(source_counts) > 1:
            print("  Data source distribution:")
            for src, count in sorted(source_counts.items()):
                print(f"    - {src}: {count}")

        # 循环增强概率
        print(f"  Cycle augmentation prob: {self.cycle_aug_prob}")
        print(f"  Caption-as-edit prob: {self.caption_as_edit_prob}")

        # metrics统计（如果有）
        if any("metrics" in p for p in self.pairs):
            align_scores = [p["metrics"].get("r_align_edit", 0) for p in self.pairs if "metrics" in p]
            if align_scores:
                print(f"  Avg r_align_edit: {np.mean(align_scores):.4f} "
                      f"(min: {np.min(align_scores):.4f}, max: {np.max(align_scores):.4f})")

    def __len__(self):
        return len(self.pairs)

    def inv_transform(self, data):
        if isinstance(data, np.ndarray):
            return data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        elif isinstance(data, torch.Tensor):
            return data * torch.from_numpy(self.std[:data.shape[-1]]).float().to(
                data.device
            ) + torch.from_numpy(self.mean[:data.shape[-1]]).float().to(data.device)
        else:
            raise TypeError("Expected data to be either np.ndarray or torch.Tensor")

    def _load_latent(self, path: str) -> torch.Tensor:
        """加载latent文件，支持相对路径、绝对路径，以及前三级路径替换为data_root"""
        if not path:
            raise ValueError("Empty path provided")

        # 1. 尝试原始路径（绝对或相对）
        full_path = Path(path)
        if full_path.exists():
            try:
                data = np.load(full_path)
                return torch.from_numpy(data).float()
            except Exception as e:
                raise RuntimeError(f"Failed to load {full_path}: {e}")

        # 2. 原始路径不存在：将前三级目录替换为 data_root
        # 例如 ../data/SnapMoGen/latents_hrvae_detail/test/xxx.npy
        # 前三级: root, datasets, SnapMoGen -> 替换为 data_root
        parts = full_path.parts
        if len(parts) > 3:
            if parts[0] == '/':
                # 绝对路径：跳过 '/' 和前三层目录，保留剩余部分
                relative_parts = parts[4:]
            else:
                # 相对路径：跳过前三层目录
                relative_parts = parts[3:]

            new_path = self.data_root / Path(*relative_parts)
            if new_path.exists():
                try:
                    data = np.load(new_path)
                    return torch.from_numpy(data).float()
                except Exception as e:
                    raise RuntimeError(f"Failed to load {new_path}: {e}")

        # 3. 仍不存在：尝试作为相对于 data_root 的路径
        relative_path = self.data_root / path
        if relative_path.exists():
            try:
                data = np.load(relative_path)
                return torch.from_numpy(data).float()
            except Exception as e:
                raise RuntimeError(f"Failed to load {relative_path}: {e}")

        # 4. 所有尝试均失败
        raise FileNotFoundError(
            f"Latent file not found: {path}\n"
            f"  Tried: {full_path}\n"
            f"  Tried (replace top-3 dirs with data_root): {new_path if len(parts) > 3 else 'N/A'}\n"
            f"  Tried (relative to data_root): {relative_path}"
        )

    def _get_latent_paths(self, pair: Dict, use_cycle: bool = False) -> tuple:
        """
        根据数据来源和是否循环增强，获取对应的latent路径
        Returns: (source_path, target_path)
        """
        if use_cycle:
            # 循环增强时交换源和目标
            src_path = pair.get("edited_path", pair.get("target_latent_path", ""))
            tgt_path = pair.get("source_path", pair.get("source_latent_path", ""))
        else:
            src_path = pair.get("source_path", pair.get("source_latent_path", ""))
            tgt_path = pair.get("edited_path", pair.get("target_latent_path", ""))

        return src_path, tgt_path

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        data_source = pair["_data_source"]

        # 循环增强决策
        use_cycle = (self.split == "train" and random.random() < self.cycle_aug_prob)

        # 根据循环增强状态获取路径和文本
        src_path, tgt_path = self._get_latent_paths(pair, use_cycle)

        if use_cycle:
            source_caption = pair["target_caption"]
            target_caption = pair["source_caption"]
            edit_text = pair.get("reverse_edit_command", "Restore the previous action")
            is_cycle = True
        else:
            source_caption = pair["source_caption"]
            target_caption = pair["target_caption"]
            edit_text = pair["edit_command"]
            is_cycle = False

        # Caption增强：以一定概率直接使用target caption作为edit command
        use_caption_as_edit = (self.split == "train" and
                               random.random() < self.caption_as_edit_prob)
        if use_caption_as_edit:
            edit_text = target_caption

        # 加载latents
        try:
            source_latent = self._load_latent(src_path)
            target_latent = self._load_latent(tgt_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[Error] Failed to load pair {pair.get('original_key', idx)}: {e}")
            # 返回一个随机的其他样本作为替代（避免训练中断）
            return self.__getitem__((idx + 1) % len(self.pairs))

        # 【关键修复】维度一致性校验：source 与 target 必须同维同空间形状
        if source_latent.dim() != target_latent.dim():
            print(f"[Error] Dimension mismatch in pair {pair.get('original_key', idx)} ({data_source}): "
                  f"source_dim={source_latent.dim()} (shape={source_latent.shape}), "
                  f"target_dim={target_latent.dim()} (shape={target_latent.shape})")
            return self.__getitem__((idx + 1) % len(self.pairs))

        # 对于 2D+ 的 latent，空间/特征维度必须严格匹配（第一维是时间，允许长度不同）
        if source_latent.dim() >= 2 and source_latent.shape[1:] != target_latent.shape[1:]:
            print(f"[Error] Spatial dimension mismatch in pair {pair.get('original_key', idx)} ({data_source}): "
                  f"source_shape={source_latent.shape}, target_shape={target_latent.shape}")
            return self.__getitem__((idx + 1) % len(self.pairs))

        # 长度对齐（仅裁剪时间维度，空间维度已验证一致）
        source_len = source_latent.shape[0]
        target_len = target_latent.shape[0]

        if source_len != target_len:
            min_len = min(source_len, target_len, self.max_length)
            source_latent = source_latent[:min_len]
            target_latent = target_latent[:min_len]
            length = min_len
            if self.split == "train" and random.random() < 0.01:  # 减少日志频率
                pair_id = pair.get("original_key", idx)
                print(f"[Warning] Length mismatch in {pair_id} ({data_source}): "
                      f"src={source_len}, tgt={target_len}, using {min_len}")
        else:
            length = min(source_len, self.max_length)
            if length < source_len:
                source_latent = source_latent[:length]
                target_latent = target_latent[:length]

        return {
            "source": source_latent,
            "target": target_latent,
            "edit_text": edit_text,
            "reverse_edit_text": pair.get("reverse_edit_command", "Restore the previous action"),
            "length": length,
            "source_caption": source_caption,
            "target_caption": target_caption,
            "is_cycle": is_cycle,
            "is_caption_edit": use_caption_as_edit,
            "pair_id": pair.get("original_key", str(idx)),
            "data_source": data_source,
            # 附加信息（可用于分析）
            # "metrics": pair.get("metrics", {}),
            # "variation_idx": pair.get("variation_idx", 0),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义collate函数处理变长序列，兼容3D (L,D) 与 4D (L,P,D) latent"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples in batch are None")

    max_len = max([b["length"] for b in batch])
    B = len(batch)
    sample = batch[0]["source"]

    # 【关键修复】增强防御性检查：确保batch内所有样本维度一致，且target与source匹配
    for i, b in enumerate(batch):
        # 1. source 维度一致性
        if b["source"].dim() != sample.dim():
            raise ValueError(
                f"Batch中样本source维度不一致! sample[0] source dim={sample.dim()}, "
                f"sample[{i}] source dim={b['source'].dim()}. "
                f"请确保所有latent文件格式统一（均为2D (L,D) 或 3D (L,P,D)）。"
            )
        # 2. target 与 source 维度必须匹配（已由 __getitem__ 兜底，此处二次确认）
        if b["target"].dim() != b["source"].dim():
            raise ValueError(
                f"Batch中sample[{i}]的source和target维度不匹配! "
                f"source dim={b['source'].dim()}, target dim={b['target'].dim()}. "
                f"pair_id={b.get('pair_id', 'unknown')}"
            )
        # 3. 空间/特征维度必须一致（第一维时间已对齐，后续维度必须相同）
        if b["source"].shape[1:] != b["target"].shape[1:]:
            raise ValueError(
                f"Batch中sample[{i}]的source和target空间维度不匹配! "
                f"source shape={b['source'].shape}, target shape={b['target'].shape}. "
                f"pair_id={b.get('pair_id', 'unknown')}"
            )

    if sample.dim() == 3:
        # 4D latent: (L, P, D) -> batch后 (B, max_len, P, D)
        P = sample.shape[1]
        D = sample.shape[2]
        source = torch.zeros(B, max_len, P, D)
        target = torch.zeros(B, max_len, P, D)
        mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, b in enumerate(batch):
            L = b["length"]
            source[i, :L] = b["source"][:L]
            target[i, :L] = b["target"][:L]
            mask[i, :L] = True

    elif sample.dim() == 2:
        # 3D latent: (L, D) -> batch后 (B, max_len, D)
        D = sample.shape[-1]
        source = torch.zeros(B, max_len, D)
        target = torch.zeros(B, max_len, D)
        mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, b in enumerate(batch):
            L = b["length"]
            source[i, :L] = b["source"][:L]
            target[i, :L] = b["target"][:L]
            mask[i, :L] = True
    else:
        raise ValueError(
            f"不支持的latent维度: {sample.dim()}，期望2D (L,D) 或 3D (L,P,D)。"
            f"实际shape: {sample.shape}"
        )

    return {
        "source": source,
        "target": target,
        "edit_text": [b["edit_text"] for b in batch],
        "reverse_edit_text": [b["reverse_edit_text"] for b in batch],
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "mask": mask,
        "source_caption": [b["source_caption"] for b in batch],
        "target_caption": [b["target_caption"] for b in batch],
        "is_cycle": [b["is_cycle"] for b in batch],
        "is_caption_edit": [b["is_caption_edit"] for b in batch],
        "pair_id": [b["pair_id"] for b in batch],
        "data_source": [b["data_source"] for b in batch],
    }
# # dataset/edit_dataset.py
# import json
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Union
#
#
# class MotionEditDataset(Dataset):
#     """
#     动作编辑数据集（多源版本）
#
#     支持多种数据源：
#     1. data_files: 多个JSON文件路径列表，每个文件包含编辑对数据
#        格式示例： [{"original_key": "cap_000006", "source_path": "...", "edited_path": "...",
#                    "source_caption": "...", "target_caption": "...", "edit_command": "...", ...}, ...]
#     2. data_list: 直接传入数据列表（内存中已加载的数据）
#     3. 向后兼容：保留 accepted_pairs_file 和 real_pairs_file 参数
#
#     支持循环一致性增强：随机交换源/目标，并替换为逆命令
#     支持Caption增强：以一定概率直接使用target caption作为edit command
#     """
#
#     def __init__(
#             self,
#             data_root: str,
#             split: str = "train",
#             # 新增：支持多源数据文件
#             data_files: Optional[List[str]] = None,  # 多个JSON文件路径列表
#             data_list: Optional[List[Dict]] = None,  # 直接传入数据列表
#             # 向后兼容
#             accepted_pairs_file: Optional[str] = None,
#             real_pairs_file: Optional[str] = None,
#             # 配置参数
#             cycle_aug_prob: float = 0.5,
#             caption_as_edit_prob: float = 0.0,
#             max_length: int = 320,
#             mean=None, std=None,
#             prioritize_real_pairs: bool = False,
#             # 数据筛选
#             min_metrics: Optional[Dict[str, float]] = None,  # 根据metrics筛选，如 {"r_align_edit": 0.7}
#             only_accepted: bool = True,  # 只加载 status == "accepted" 的数据
#     ):
#         super().__init__()
#         self.split = split
#         self.cycle_aug_prob = cycle_aug_prob if split == "train" else 0.0
#         self.caption_as_edit_prob = caption_as_edit_prob if split == "train" else 0.0
#         self.max_length = max_length // 4
#         self.mean = mean
#         self.std = std
#         self.prioritize_real_pairs = prioritize_real_pairs
#
#         # 筛选条件
#         self.min_metrics = min_metrics or {}
#         self.only_accepted = only_accepted
#
#         all_pairs = []
#
#         # ========== 1. 处理新格式的多源数据文件 ==========
#         if data_files is not None:
#             for file_idx, data_file in enumerate(data_files):
#                 # 支持绝对路径或相对于 data_root 的路径
#                 file_path = Path(data_file) if Path(data_file).is_absolute() else Path(data_root) / data_file
#
#                 if not file_path.exists():
#                     print(f"[Warning] data_file not found: {file_path}")
#                     continue
#
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
#
#                     # 统一转换为列表格式
#                     if isinstance(data, dict):
#                         # 如果JSON是字典格式（如 {"cap_000006": {...}, ...}）
#                         data = list(data.values())
#                     elif not isinstance(data, list):
#                         print(f"[Warning] Unsupported data format in {file_path}, expected list or dict")
#                         continue
#
#                     # 为每个数据添加来源标识
#                     source_tag = f"file_{file_idx}_{file_path.stem}"
#                     loaded_count = 0
#                     filtered_count = 0
#
#                     for item in data:
#                         if not isinstance(item, dict):
#                             continue
#
#                         # 检查 split
#                         if item.get("split", "train") != split:
#                             continue
#
#                         # 检查 status (如果 only_accepted=True)
#                         if self.only_accepted and item.get("status", "accepted") != "accepted":
#                             filtered_count += 1
#                             continue
#
#                         # 检查 metrics 条件
#                         metrics = item.get("metrics", {})
#                         metrics_pass = all(
#                             metrics.get(k, float('-inf')) >= v
#                             for k, v in self.min_metrics.items()
#                         )
#                         if not metrics_pass:
#                             filtered_count += 1
#                             continue
#
#                         # 添加元数据
#                         item["_data_source"] = source_tag
#                         item["_file_idx"] = file_idx
#
#                         # 确保必要字段存在（兼容处理）
#                         self._normalize_item(item)
#
#                         all_pairs.append(item)
#                         loaded_count += 1
#
#                     print(f"[MotionEditDataset] Loaded {loaded_count} pairs from {file_path.name} "
#                           f"(filtered {filtered_count}, tag: {source_tag})")
#
#                 except Exception as e:
#                     print(f"[Error] Failed to load {file_path}: {e}")
#
#         # ========== 2. 处理直接传入的数据列表 ==========
#         if data_list is not None:
#             loaded_count = 0
#             for item in data_list:
#                 if item.get("split", "train") != split:
#                     continue
#                 if self.only_accepted and item.get("status", "accepted") != "accepted":
#                     continue
#
#                 item["_data_source"] = "data_list"
#                 self._normalize_item(item)
#                 all_pairs.append(item)
#                 loaded_count += 1
#
#             print(f"[MotionEditDataset] Loaded {loaded_count} pairs from data_list")
#
#         # ========== 3. 向后兼容：处理旧的参数格式 ==========
#         # 加载 accepted_pairs（FlowEdit生成）
#         if accepted_pairs_file is not None:
#             # pairs_path = Path(data_root) / "edit_latents_filtered_new" / accepted_pairs_file
#             pairs_path = Path(data_root) / "edit_filtered_6var" / accepted_pairs_file
#             if pairs_path.exists():
#                 with open(pairs_path, 'r', encoding='utf-8') as f:
#                     accepted_pairs = json.load(f)
#
#                 loaded_count = 0
#                 for p in accepted_pairs:
#                     if p.get("split", "train") != split:
#                         continue
#                     if self.only_accepted and p.get("status", "accepted") != "accepted":
#                         continue
#
#                     p["_data_source"] = "accepted_pairs"
#                     self._normalize_item(p)
#                     all_pairs.append(p)
#                     loaded_count += 1
#
#                 print(f"[MotionEditDataset] Loaded {loaded_count} pairs from {accepted_pairs_file} (legacy)")
#             else:
#                 print(f"[Warning] accepted_pairs_file not found: {pairs_path}")
#
#         # 加载 real_motion_pairs（真实动作对）
#         if real_pairs_file is not None:
#             real_path = Path(data_root) / real_pairs_file
#             if real_path.exists():
#                 with open(real_path, 'r', encoding='utf-8') as f:
#                     real_pairs = json.load(f)
#
#                 loaded_count = 0
#                 for p in real_pairs:
#                     if p.get("split", "train") != split:
#                         continue
#
#                     p["_data_source"] = "real_pairs"
#                     if "original_key" not in p:
#                         p["original_key"] = p.get("source_key", p.get("target_key", "unknown"))
#                     self._normalize_item(p)
#                     all_pairs.append(p)
#                     loaded_count += 1
#
#                 print(f"[MotionEditDataset] Loaded {loaded_count} pairs from {real_pairs_file} (legacy)")
#             else:
#                 print(f"[Warning] real_pairs_file not found: {real_path}")
#
#         if len(all_pairs) == 0:
#             raise ValueError(f"No data loaded for split '{split}'. Please check file paths.")
#
#         # ========== 4. 去重逻辑（优先保留real_pairs） ==========
#         if prioritize_real_pairs:
#             seen_keys = {}
#             filtered_pairs = []
#             for p in all_pairs:
#                 key = p.get("original_key", str(id(p)))
#                 if key in seen_keys:
#                     # 如果当前是real_pairs，替换已有的
#                     if p["_data_source"] == "real_pairs":
#                         for i, existing in enumerate(filtered_pairs):
#                             if existing.get("original_key") == key:
#                                 filtered_pairs[i] = p
#                                 break
#                 else:
#                     seen_keys[key] = True
#                     filtered_pairs.append(p)
#             all_pairs = filtered_pairs
#             print(f"[MotionEditDataset] After deduplication: {len(all_pairs)} pairs")
#
#         self.pairs = all_pairs
#
#         # 统计信息
#         self._print_statistics()
#
#     def _normalize_item(self, item: Dict):
#         """
#         规范化数据项，确保必要字段存在
#         处理不同来源的数据格式差异
#         """
#         # 确保 original_key 存在
#         if "original_key" not in item:
#             item["original_key"] = item.get("sample_id", item.get("source_key", str(id(item))))
#
#         # ========== 新增：处理 edit_pairs_aligned.json 的字段名 ==========
#         # 处理源路径字段映射 (source_motion_path -> source_path/source_latent_path)
#         if "source_path" not in item and "source_motion_path" in item:
#             item["source_path"] = item["source_motion_path"]
#         if "source_latent_path" not in item and "source_motion_path" in item:
#             item["source_latent_path"] = item["source_motion_path"]
#
#         # 处理目标路径字段映射 (target_motion_path -> edited_path/target_latent_path)
#         if "edited_path" not in item and "target_motion_path" in item:
#             item["edited_path"] = item["target_motion_path"]
#         if "target_latent_path" not in item and "target_motion_path" in item:
#             item["target_latent_path"] = item["target_motion_path"]
#         # ==============================================================
#
#         # 统一路径字段（原有的兼容逻辑）
#         if "edited_path" not in item and "target_path" in item:
#             item["edited_path"] = item["target_path"]
#         if "target_latent_path" not in item and "edited_path" in item:
#             item["target_latent_path"] = item["edited_path"]
#         if "source_latent_path" not in item and "source_path" in item:
#             item["source_latent_path"] = item["source_path"]
#
#         # 确保caption字段存在
#         if "source_caption" not in item:
#             item["source_caption"] = item.get("original_caption", "No source caption")
#         if "target_caption" not in item:
#             item["target_caption"] = item.get("edited_caption", item.get("new_caption", "No target caption"))
#
#         # 确保edit_command字段存在
#         if "edit_command" not in item:
#             item["edit_command"] = item.get("edit_text", "Transform the motion")
#         if "reverse_edit_command" not in item:
#             item["reverse_edit_command"] = item.get("reverse_command", "Revert the changes")
#
#     def _print_statistics(self):
#         """打印数据集统计信息"""
#         print(f"[MotionEditDataset] Total {len(self.pairs)} pairs for {self.split}")
#
#         # 统计各数据源数量
#         source_counts = {}
#         for p in self.pairs:
#             src = p["_data_source"]
#             source_counts[src] = source_counts.get(src, 0) + 1
#
#         if len(source_counts) > 1:
#             print("  Data source distribution:")
#             for src, count in sorted(source_counts.items()):
#                 print(f"    - {src}: {count}")
#
#         # 循环增强概率
#         print(f"  Cycle augmentation prob: {self.cycle_aug_prob}")
#         print(f"  Caption-as-edit prob: {self.caption_as_edit_prob}")
#
#         # metrics统计（如果有）
#         if any("metrics" in p for p in self.pairs):
#             align_scores = [p["metrics"].get("r_align_edit", 0) for p in self.pairs if "metrics" in p]
#             if align_scores:
#                 print(f"  Avg r_align_edit: {np.mean(align_scores):.4f} "
#                       f"(min: {np.min(align_scores):.4f}, max: {np.max(align_scores):.4f})")
#
#     def __len__(self):
#         return len(self.pairs)
#
#     def inv_transform(self, data):
#         if isinstance(data, np.ndarray):
#             return data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
#         elif isinstance(data, torch.Tensor):
#             return data * torch.from_numpy(self.std[:data.shape[-1]]).float().to(
#                 data.device
#             ) + torch.from_numpy(self.mean[:data.shape[-1]]).float().to(data.device)
#         else:
#             raise TypeError("Expected data to be either np.ndarray or torch.Tensor")
#
#     def _load_latent(self, path: str) -> torch.Tensor:
#         """加载latent文件，支持相对路径和绝对路径"""
#         if not path:
#             raise ValueError("Empty path provided")
#
#         # 已经是绝对路径
#         if Path(path).exists():
#             full_path = Path(path)
#         else:
#             # 尝试作为相对路径
#             full_path = Path(path)
#             if not full_path.exists():
#                 raise FileNotFoundError(f"Latent file not found: {path}")
#
#         try:
#             data = np.load(full_path)
#             return torch.from_numpy(data).float()
#         except Exception as e:
#             raise RuntimeError(f"Failed to load {full_path}: {e}")
#
#     def _get_latent_paths(self, pair: Dict, use_cycle: bool = False) -> tuple:
#         """
#         根据数据来源和是否循环增强，获取对应的latent路径
#         Returns: (source_path, target_path)
#         """
#         if use_cycle:
#             # 循环增强时交换源和目标
#             src_path = pair.get("edited_path", pair.get("target_latent_path", ""))
#             tgt_path = pair.get("source_path", pair.get("source_latent_path", ""))
#         else:
#             src_path = pair.get("source_path", pair.get("source_latent_path", ""))
#             tgt_path = pair.get("edited_path", pair.get("target_latent_path", ""))
#
#         return src_path, tgt_path
#
#     def __getitem__(self, idx):
#         pair = self.pairs[idx]
#         data_source = pair["_data_source"]
#
#         # 循环增强决策
#         use_cycle = (self.split == "train" and random.random() < self.cycle_aug_prob)
#
#         # 根据循环增强状态获取路径和文本
#         src_path, tgt_path = self._get_latent_paths(pair, use_cycle)
#
#         if use_cycle:
#             source_caption = pair["target_caption"]
#             target_caption = pair["source_caption"]
#             edit_text = pair.get("reverse_edit_command", "Restore the previous action")
#             is_cycle = True
#         else:
#             source_caption = pair["source_caption"]
#             target_caption = pair["target_caption"]
#             edit_text = pair["edit_command"]
#             is_cycle = False
#
#         # Caption增强：以一定概率直接使用target caption作为edit command
#         use_caption_as_edit = (self.split == "train" and
#                                random.random() < self.caption_as_edit_prob)
#         if use_caption_as_edit:
#             edit_text = target_caption
#
#         # 加载latents
#         try:
#             source_latent = self._load_latent(src_path)
#             target_latent = self._load_latent(tgt_path)
#         except (FileNotFoundError, RuntimeError) as e:
#             print(f"[Error] Failed to load pair {pair.get('original_key', idx)}: {e}")
#             # 返回一个随机的其他样本作为替代（避免训练中断）
#             return self.__getitem__((idx + 1) % len(self.pairs))
#
#         # 长度对齐
#         source_len = source_latent.shape[0]
#         target_len = target_latent.shape[0]
#
#         if source_len != target_len:
#             min_len = min(source_len, target_len, self.max_length)
#             source_latent = source_latent[:min_len]
#             target_latent = target_latent[:min_len]
#             length = min_len
#             if self.split == "train" and random.random() < 0.01:  # 减少日志频率
#                 pair_id = pair.get("original_key", idx)
#                 print(f"[Warning] Length mismatch in {pair_id} ({data_source}): "
#                       f"src={source_len}, tgt={target_len}, using {min_len}")
#         else:
#             length = min(source_len, self.max_length)
#             if length < source_len:
#                 source_latent = source_latent[:length]
#                 target_latent = target_latent[:length]
#
#         return {
#             "source": source_latent,
#             "target": target_latent,
#             "edit_text": edit_text,
#             "length": length,
#             "source_caption": source_caption,
#             "target_caption": target_caption,
#             "is_cycle": is_cycle,
#             "is_caption_edit": use_caption_as_edit,
#             "pair_id": pair.get("original_key", str(idx)),
#             "data_source": data_source,
#             # 附加信息（可用于分析）
#             # "metrics": pair.get("metrics", {}),
#             # "variation_idx": pair.get("variation_idx", 0),
#         }
#
#
# def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """自定义collate函数处理变长序列"""
#     batch = [b for b in batch if b is not None]
#     if len(batch) == 0:
#         raise RuntimeError("All samples in batch are None")
#
#     max_len = max([b["length"] for b in batch])
#     B = len(batch)
#     D = batch[0]["source"].shape[-1]
#
#     source = torch.zeros(B, max_len, D)
#     target = torch.zeros(B, max_len, D)
#     mask = torch.zeros(B, max_len, dtype=torch.bool)
#
#     for i, b in enumerate(batch):
#         L = b["length"]
#         source[i, :L] = b["source"]
#         target[i, :L] = b["target"]
#         mask[i, :L] = True
#
#     return {
#         "source": source,
#         "target": target,
#         "edit_text": [b["edit_text"] for b in batch],
#         "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
#         "mask": mask,
#         "source_caption": [b["source_caption"] for b in batch],
#         "target_caption": [b["target_caption"] for b in batch],
#         "is_cycle": [b["is_cycle"] for b in batch],
#         "is_caption_edit": [b["is_caption_edit"] for b in batch],
#         "pair_id": [b["pair_id"] for b in batch],
#         "data_source": [b["data_source"] for b in batch],
#         # "metrics": [b.get("metrics", {}) for b in batch],
#         # "variation_idx": torch.tensor([b.get("variation_idx", 0) for b in batch], dtype=torch.long),
#     }
