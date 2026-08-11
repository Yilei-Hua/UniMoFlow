#!/usr/bin/env python3
import sys as _release_sys
from pathlib import Path as _ReleasePath

_CODES_ROOT = _ReleasePath(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_CODES_ROOT))

# -*- coding: utf-8 -*-
"""Generate comprehensive, category-agnostic edit-text candidates.

Unlike ``generate_multi_attribute_edits.py``, this entry point does not
partition generation into coarse, fine, and style jobs. A single Qwen prompt
directly samples diverse edits that may combine body-part, action-type,
spatial, timing, and style changes. Each source caption yields
multiple ``edit_command``/``new_caption``/``reverse_command`` variations.

The implementation supports multi-GPU inference, per-sample fault isolation,
OOM fallback, and resuming failed source keys.
"""

import json
import os
import argparse
import re
import torch
import multiprocessing as mp
from typing import List, Dict, Optional, Set
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GPU %(process)d] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActionCaptionEditor:
    def __init__(self, model_path: str, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"

        logger.info(f"[GPU {gpu_id}] Loading model: {model_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model: BF16 + Flash Attention 2 (A100 40G optimal config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,           # BF16 for A100 optimal
            device_map=self.device,                # 直接加载到指定GPU
            attn_implementation="flash_attention_2",  # Flash Attention 2
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        logger.info(f"[GPU {gpu_id}] Model loaded successfully on {self.device}")

    def build_compact_prompt(self, original_caption: str, num_commands: int) -> str:
        """
        精简 Prompt - 强制英文输出
        """
        return f"""You are a command user, using commands to edit the virtual human's original actions. You will give {num_commands} editing variations according to the human motion description.

Source description: "{original_caption}"

For each variation, provide:
1. edit_command: Detailed editing instruction (body part, action type, style, direction, timing)
2. new_caption: Edited target description (keep source sentence structure, only modify edited parts)
3. reverse_command: Reverse instruction to recover source from target

Example:
Source: "The person stands with their legs spread wide and takes a couple of steps back."
{{
  "variations": [
    {{
      "edit_command": "At the beginning, bring legs together and step forward instead of back",
      "new_caption": "The person stands with their legs together and takes a couple of steps forward.",
      "reverse_command": "At the beginning, spread legs wide and take steps back"
    }}
  ]
}}

Requirements:
- edit_command must specify: body part + spatial direction + temporal marker; optionally: style + action type
- new_caption maintains source sentence structure, only changing semantic parts related to edit,
- reverse_command uses opposite directional operations to describe the change back to source
- The generated results exhibit diversity across various aspects, including body part, action type, style, and timing.
- Do not omit any generated content(edit_command, new_caption and reverse_command).
- **Output must be in English**

Output JSON format:
{{
  "variations": [
    {{
      "edit_command": "...",
      "new_caption": "...",
      "reverse_command": "..."
    }}
  ]
}}

JSON Output:"""

    def parse_json_response(self, text: str) -> Optional[List[Dict]]:
        """
        健壮解析，失败返回 None
        """
        if not text or not text.strip():
            return None

        # 清理 markdown 代码块
        if "```json" in text:
            matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if matches:
                text = matches[-1]
        elif "```" in text:
            matches = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if matches:
                text = matches[-1]

        try:
            data = json.loads(text.strip())
            if "variations" in data and isinstance(data["variations"], list):
                return data["variations"]
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            # 尝试正则提取
            try:
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1 and end > start:
                    data = json.loads(text[start:end + 1])
                    if isinstance(data, list):
                        return data
            except:
                pass

        return None

    def _generate_single(self, prompt: str, max_new_tokens: int = 1536, temperature: float = 0.7) -> Optional[List[Dict]]:
        """
        单样本生成（OOM 降级备用方案）
        严格隔离单个样本的异常
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            input_ids_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[:, input_ids_len:]
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            return self.parse_json_response(decoded.strip())

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"[GPU {self.gpu_id}] Single sample OOM, skipping and clearing cache")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.warning(f"[GPU {self.gpu_id}] Single sample generation error: {e}")
            return None

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 1536, temperature: float = 0.7) -> List[
        Optional[List[Dict]]]:
        """
        批量生成，单个样本失败不影响其他样本
        具备 OOM 自动恢复机制（批处理失败降级为逐个处理）
        """
        if not prompts:
            return []

        # 预初始化结果列表（全 None，成功则替换）
        results = [None] * len(prompts)

        # 首先尝试批量生成（高效模式）
        try:
            # 应用 Chat Template
            texts = []
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)

            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                padding_side="left"
            ).to(self.device)

            input_ids_len = inputs.input_ids.shape[1]

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[:, input_ids_len:]
            decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # 关键修改：逐个解析，隔离失败，避免一个样本解析失败导致整批丢弃
            for idx, text in enumerate(decoded_texts):
                try:
                    parsed = self.parse_json_response(text.strip())
                    results[idx] = parsed
                except Exception as e:
                    logger.warning(f"[GPU {self.gpu_id}] Sample {idx} parsing failed: {e}")
                    results[idx] = None

        except torch.cuda.OutOfMemoryError as e:
            # OOM 恢复：清空缓存并降级为逐个生成
            logger.warning(f"[GPU {self.gpu_id}] Batch OOM ({len(prompts)} samples), falling back to single processing: {e}")
            torch.cuda.empty_cache()

            # 逐个生成，严格隔离每个样本的异常
            for idx, prompt in enumerate(prompts):
                result = self._generate_single(prompt, max_new_tokens, temperature)
                results[idx] = result

        except Exception as e:
            # 其他批处理错误，同样降级为逐个生成
            logger.error(f"[GPU {self.gpu_id}] Batch generation error: {e}, falling back to single processing")

            for idx, prompt in enumerate(prompts):
                result = self._generate_single(prompt, max_new_tokens, temperature)
                results[idx] = result

        return results

    def clean_caption(self, text: str) -> str:
        """清理 caption"""
        if not text:
            return ""
        text = text.strip().strip('"').strip("'")
        prefixes = ["New caption:", "Modified:", "Output:", "Result:", "New action:"]
        for p in prefixes:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip().strip(":").strip()
        return text

    def process_data_slice(self, data_slice: List[tuple], num_commands: int, batch_size: int, temp_output: str, temp_failed_output: str):
        """
        处理数据切片 - 失败直接跳过，不填充
        改进：处理 batch 结果时严格按索引对应，None 表示该样本失败

        Returns:
            tuple: (temp_output, temp_failed_output, success_count, fail_count, failed_keys)
        """
        results = []
        failed_keys = []  # 记录失败的 key
        success_count = 0
        fail_count = 0

        # 断点续传检查（成功文件）
        processed_keys = set()
        if os.path.exists(temp_output):
            try:
                with open(temp_output, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    results = existing_data
                    processed_keys = {item["original_key"] for item in results}
                success_count = len(results)
                logger.info(f"[GPU {self.gpu_id}] Resumed success file: {len(processed_keys)} already processed")
            except Exception as e:
                logger.warning(f"[GPU {self.gpu_id}] Failed to load temp success file: {e}")

        # 断点续传检查（失败文件）- 加载之前记录的失败 key
        existing_failed_keys = set()
        if os.path.exists(temp_failed_output):
            try:
                with open(temp_failed_output, 'r', encoding='utf-8') as f:
                    existing_failed_keys = set(json.load(f))
                logger.info(f"[GPU {self.gpu_id}] Resumed failed file: {len(existing_failed_keys)} failed keys already recorded")
            except Exception as e:
                logger.warning(f"[GPU {self.gpu_id}] Failed to load temp failed file: {e}")

        # 合并已处理的 key（成功 + 失败）
        all_processed_keys = processed_keys.union(existing_failed_keys)

        # 过滤已处理数据
        remaining_data = [(k, c) for k, c in data_slice if k not in all_processed_keys]
        if not remaining_data:
            logger.info(f"[GPU {self.gpu_id}] All items already processed (success or failed)")
            return temp_output, temp_failed_output, success_count, fail_count, list(existing_failed_keys)

        logger.info(f"[GPU {self.gpu_id}] Processing {len(remaining_data)} new items")
        pbar = tqdm(total=len(remaining_data), desc=f"GPU{self.gpu_id}", position=self.gpu_id)

        for i in range(0, len(remaining_data), batch_size):
            batch = remaining_data[i:i + batch_size]
            batch_keys = [k for k, c in batch]
            batch_captions = [c for k, c in batch]

            # 构建 prompts
            prompts = [self.build_compact_prompt(c, num_commands) for c in batch_captions]

            # 生成（已包含内部异常隔离）
            responses = self.generate_batch(prompts, max_new_tokens=1536, temperature=0.7)

            # 处理每个结果 - 失败直接跳过并记录
            for key, orig_caption, variations in zip(batch_keys, batch_captions, responses):
                if variations is None:
                    fail_count += 1
                    failed_keys.append(key)  # 记录失败的 key
                    continue  # 跳过失败项，不保存

                # 过滤有效 variations
                valid_vars = []
                for var in variations[:num_commands]:
                    if not isinstance(var, dict):
                        continue
                    # 检查必需字段
                    if not all(k in var and var[k] for k in ["edit_command", "new_caption", "reverse_command"]):
                        continue

                    # 检查是否为空或太短
                    edit_cmd = str(var["edit_command"]).strip()
                    new_cap = str(var["new_caption"]).strip()
                    rev_cmd = str(var["reverse_command"]).strip()

                    if len(edit_cmd) < 3 or len(new_cap) < 3 or len(rev_cmd) < 3:
                        continue

                    valid_vars.append({
                        "edit_command": edit_cmd,
                        "new_caption": self.clean_caption(new_cap),
                        "reverse_edit_command": rev_cmd
                    })

                if not valid_vars:
                    fail_count += 1
                    failed_keys.append(key)  # 记录失败的 key
                    continue  # 没有有效 variation，跳过

                results.append({
                    "original_key": key,
                    "original_caption": orig_caption,
                    "variations": valid_vars
                })
                success_count += 1

            # 定期保存（包括失败记录）
            if (i // batch_size) % 5 == 0 or i + batch_size >= len(remaining_data):
                # 保存成功结果
                with open(temp_output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                # 保存失败 keys（合并之前的和新的）
                all_failed_keys = list(existing_failed_keys.union(set(failed_keys)))
                with open(temp_failed_output, 'w', encoding='utf-8') as f:
                    json.dump(all_failed_keys, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch))

        pbar.close()

        # 合并最终的失败 keys
        all_failed_keys = list(existing_failed_keys.union(set(failed_keys)))

        logger.info(f"[GPU {self.gpu_id}] Done: Success={success_count}, Failed={fail_count}, TotalFailedKeys={len(all_failed_keys)}")
        return temp_output, temp_failed_output, success_count, fail_count, all_failed_keys


def worker_task(gpu_id: int, model_path: str, data_slice: List[tuple], num_commands: int, batch_size: int,
                temp_output: str, temp_failed_output: str, result_queue):
    """
    工作进程
    """
    try:
        editor = ActionCaptionEditor(model_path, gpu_id)
        temp_file, temp_failed_file, success, fail, failed_keys = editor.process_data_slice(
            data_slice, num_commands, batch_size, temp_output, temp_failed_output
        )
        result_queue.put((gpu_id, temp_file, temp_failed_file, success, fail, failed_keys))
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Worker crashed: {e}", exc_info=True)
        # 如果崩溃，假设全部失败
        failed_keys = [k for k, c in data_slice]
        result_queue.put((gpu_id, None, None, 0, len(data_slice), failed_keys))


def merge_results(temp_files: List[str], final_output: str):
    """
    合并成功结果文件
    """
    all_results = []
    total_success = 0

    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_results.extend(data)
                    total_success += len(data)
                logger.info(f"Loaded success temp {temp_file}: {len(data)} items")
            except Exception as e:
                logger.error(f"Failed to load {temp_file}: {e}")

    # 按 key 去重（保留最后一个）
    seen = {}
    for item in reversed(all_results):
        key = item["original_key"]
        if key not in seen:
            seen[key] = item

    unique_results = list(seen.values())

    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(unique_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Final success output: {final_output}")
    logger.info(f"Total successful items: {len(unique_results)}")

    # 清理临时文件
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

    return unique_results


def merge_failed_keys(temp_failed_files: List[str], final_failed_output: str):
    """
    合并失败 key 文件
    """
    all_failed_keys = set()

    for temp_file in temp_failed_files:
        if temp_file and os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    keys = json.load(f)
                    all_failed_keys.update(keys)
                logger.info(f"Loaded failed temp {temp_file}: {len(keys)} keys")
            except Exception as e:
                logger.error(f"Failed to load failed temp {temp_file}: {e}")

    # 保存合并后的失败 keys
    if all_failed_keys:
        with open(final_failed_output, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(all_failed_keys)), f, ensure_ascii=False, indent=2)
        logger.info(f"Final failed keys output: {final_failed_output}")
        logger.info(f"Total failed keys: {len(all_failed_keys)}")
    else:
        logger.info("No failed keys to save")
        # 如果文件已存在，删除它（因为没有失败了）
        if os.path.exists(final_failed_output):
            os.remove(final_failed_output)

    # 清理临时文件
    for temp_file in temp_failed_files:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

    return list(all_failed_keys)


def load_failed_keys(failed_keys_path: str) -> Set[str]:
    """从文件加载失败的 keys"""
    if not os.path.exists(failed_keys_path):
        logger.warning(f"Failed keys file not found: {failed_keys_path}")
        return set()

    try:
        with open(failed_keys_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
            if isinstance(keys, list):
                return set(keys)
            else:
                logger.warning(f"Failed keys file format invalid, expected list")
                return set()
    except Exception as e:
        logger.error(f"Failed to load failed keys: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Action Caption Editor with Fault Isolation and Resume Support")
    parser.add_argument('--input', type=str, default="../data/SnapMoGen/all_caption_clean.json",
                        help="Input JSON path (e.g., all_caption_clean.json)")
    parser.add_argument('--output', type=str,
                        default="../outputs/omni_moedit/edit_pairs.json",
                        help="Output JSON path for successful edit pairs")
    parser.add_argument('--failed_output', type=str,
                        default="../outputs/omni_moedit/failed_keys.json",
                        help="Output JSON path for failed sample keys")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to failed keys JSON file. If provided, only retry these keys from input file")
    parser.add_argument('--model', type=str, default="../pretrained/Qwen3-8B",
                        help="Model path")
    parser.add_argument('--gpus', type=str, default="0,1",
                        help="GPU IDs to use, comma-separated (e.g., '0,1,2,3')")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size per GPU (default: 20). Auto-fallback to single on OOM")
    parser.add_argument('--num_commands', type=int, default=6,
                        help="Number of variations per sample (default: 6)")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum samples to process (for testing)")

    args = parser.parse_args()

    # GPU 设置
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    num_gpus = len(gpu_ids)
    logger.info(f"Using GPUs: {gpu_ids} ({num_gpus} total)")

    # 读取数据
    logger.info(f"Reading input: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 适配不同格式，构建 items 列表
    items = []
    for key, value in data.items():
        caption = None
        if isinstance(value, dict):
            if "manual" in value and value["manual"]:
                caption = value["manual"][0] if isinstance(value["manual"], list) else value["manual"]
            elif "caption" in value:
                caption = value["caption"]
        elif isinstance(value, str):
            caption = value

        if caption:
            items.append((key, caption))

    # 如果指定了 resume_from，只保留失败的 keys
    if args.resume_from:
        failed_keys = load_failed_keys(args.resume_from)
        if failed_keys:
            logger.info(f"Resuming from failed keys: {len(failed_keys)} keys to retry")
            items = [(k, c) for k, c in items if k in failed_keys]
            if not items:
                logger.warning("No matching keys found in input file for the failed keys provided!")
                return
        else:
            logger.warning(f"No failed keys loaded from {args.resume_from}, processing all data")

    if args.max_samples:
        items = items[:args.max_samples]

    total = len(items)
    logger.info(f"Total items to process: {total}")

    # 数据分片
    chunk_size = (total + num_gpus - 1) // num_gpus
    chunks = []
    temp_files = []
    temp_failed_files = []

    for i, gpu_id in enumerate(gpu_ids):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = items[start:end]
        chunks.append(chunk)
        temp_files.append(f"{args.output}.gpu_{gpu_id}.tmp")
        temp_failed_files.append(f"{args.failed_output}.gpu_{gpu_id}.tmp")
        logger.info(f"GPU {gpu_id}: items {start}-{end - 1} ({len(chunk)} items)")

    # 启动多进程
    result_queue = mp.Queue()
    processes = []

    for gpu_id, chunk, temp_file, temp_failed_file in zip(gpu_ids, chunks, temp_files, temp_failed_files):
        if not chunk:
            continue
        p = mp.Process(
            target=worker_task,
            args=(gpu_id, args.model, chunk, args.num_commands, args.batch_size, temp_file, temp_failed_file, result_queue)
        )
        p.start()
        processes.append(p)

    # 收集结果统计
    completed = 0
    total_success = 0
    total_fail = 0
    all_failed_keys_per_gpu = []

    while completed < len(processes):
        gpu_id, temp_file, temp_failed_file, success, fail, failed_keys = result_queue.get()
        completed += 1
        if temp_file:
            total_success += success
            total_fail += fail
            all_failed_keys_per_gpu.append(temp_failed_file)
        logger.info(f"GPU {gpu_id} finished: success={success}, fail={fail}, failed_keys={len(failed_keys)}")

    # 等待进程结束
    for p in processes:
        p.join()

    # 合并成功结果
    merge_results(temp_files, args.output)

    # 合并失败 keys
    final_failed_keys = merge_failed_keys(temp_failed_files, args.failed_output)

    logger.info(f"Processing complete:")
    logger.info(f"  - Successfully generated: {total_success}")
    logger.info(f"  - Failed/Skipped: {total_fail}")
    if total_success + total_fail > 0:
        logger.info(f"  - Success rate: {100 * total_success / (total_success + total_fail):.1f}%")
    if final_failed_keys:
        logger.info(f"  - Failed keys saved to: {args.failed_output}")
        logger.info(f"  - To retry failed samples, run with: --resume_from {args.failed_output}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
