#!/usr/bin/env python3
import sys as _release_sys
from pathlib import Path as _ReleasePath

_CODES_ROOT = _ReleasePath(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_CODES_ROOT))

# -*- coding: utf-8 -*-
"""Generate category-controlled edit-text candidates.

This complementary entry point generates three edit groups independently:
coarse action-type edits, fine-grained body-part edits, and style edits. Each
variation includes a reverse command and a localized source phase. For direct
heterogeneous or compositional synthesis, use ``generate_edit_triplets.py``.

Examples::

    python generate_multi_attribute_edits.py --types coarse,fine,style
    python generate_multi_attribute_edits.py --types coarse
    python generate_multi_attribute_edits.py --types fine,style
    python generate_multi_attribute_edits.py --types coarse \
        --resume_from ./failed_coarse_keys.json
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


class MultiAttributeEditor:
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

        # Model: BF16 + Flash Attention 2
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        logger.info(f"[GPU {gpu_id}] Model loaded successfully on {self.device}")

    def build_coarse_prompt(self, original_caption: str, num_commands: int) -> str:
        """粗粒度动作类型编辑 Prompt - 包含 reverse_command 和 locate_edit_phase"""
        return f"""You are generating editing commands for coarse-grained action type modifications.
Source description: "{original_caption}"

Generate {num_commands} editing variations that:
- Change the overall action category (e.g., greeting, boxing, basketball, rock-paper-scissors, Kung Fu, dancing, running, walking)
- May include direction changes (e.g., run backward, wave to the right, jump forward)
- Use action sequence relationships (before/after/while/simultaneously) to specify timing, NOT specific seconds

For each variation, provide:
1. edit_command: timing (using action relationships) + new action type (+ optional direction/spatial info)
2. new_caption: edited target description keeping source structure
3. reverse_command: instruction to revert from new action back to original action
4. locate_edit_phase: **Exact substring from the source description indicating which specific action phase is being edited**

Example:
Source: "The person walks forward and waves their hand."

Variation 1:
{{
  "edit_command": "After starting to walk, change to running backward instead of walking forward",
  "new_caption": "The person runs backward and waves their hand.",
  "reverse_command": "Change running backward back to walking forward",
  "locate_edit_phase": "walks forward"
}}

Variation 2:
{{
  "edit_command": "While waving hand, change the action to playing rock-paper-scissors",
  "new_caption": "The person walks forward and plays rock-paper-scissors.",
  "reverse_command": "Replace rock-paper-scissors with waving hand",
  "locate_edit_phase": "waves their hand"
}}

Requirements:
- edit_command: specify body part + spatial direction + temporal marker + action type
- new_caption: maintain source sentence structure, only modify edited semantic parts
- reverse_command: describe the inverse operation to recover the source action
- locate_edit_phase: **MUST be a direct quote or minimal paraphrase of the specific action segment in the source being edited** (for temporal localization)
- The generated results exhibit diversity across action types, directions, and timing relationships
- Do not omit any generated content (edit_command, new_caption, reverse_command, locate_edit_phase)
- **Output must be in English**

Output JSON format:
{{
  "variations": [
    {{
      "edit_command": "...",
      "new_caption": "...",
      "reverse_command": "...",
      "locate_edit_phase": "..."
    }}
  ]
}}

JSON Output:"""

    def build_fine_prompt(self, original_caption: str, num_commands: int) -> str:
        """细粒度身体部位编辑 Prompt - 包含 reverse_command 和 locate_edit_phase"""
        return f"""You are generating editing commands for fine-grained body part modifications.
Source description: "{original_caption}"

Generate {num_commands} editing variations that:
- Change specific body part actions precisely (e.g., right hand raised → left hand extended forward horizontally)
- Include direction/orientation changes (left/right/up/down/forward/backward)
- Focus on limb-level modifications: arms, hands, legs, feet, head posture
- Use action sequence relationships (when/while/before/after/simultaneously) to specify timing

For each variation, provide:
1. edit_command: timing + body part + specific action change (+ direction/orientation)
2. new_caption: edited target description
3. reverse_command: instruction to revert the body part modification back to original
4. locate_edit_phase: **Exact substring from the source describing the specific body part action being edited**

Example:
Source: "The person turns their body while raising the right hand."

Variation 1:
{{
  "edit_command": "While turning body, change raising right hand to raising left hand extended forward horizontally",
  "new_caption": "The person turns their body while raising the left hand extended forward horizontally.",
  "reverse_command": "Change raising left hand back to raising right hand",
  "locate_edit_phase": "raising the right hand"
}}

Variation 2:
{{
  "edit_command": "When raising hand, change to raising both hands above head simultaneously",
  "new_caption": "The person turns their body while raising both hands above their head.",
  "reverse_command": "Replace raising both hands with raising right hand only",
  "locate_edit_phase": "raising the right hand"
}}

Variation 3:
{{
  "edit_command": "Instead of raising right hand, extend right arm backward while turning",
  "new_caption": "The person turns their body while extending the right arm backward.",
  "reverse_command": "Change extending right arm backward back to raising right hand",
  "locate_edit_phase": "raising the right hand"
}}

Requirements:
- edit_command: specify body part + spatial direction + temporal marker + action type
- new_caption: maintain source sentence structure, only modify edited semantic parts
- reverse_command: describe the inverse operation to recover the source body part configuration
- locate_edit_phase: **MUST be a direct quote or minimal paraphrase of the specific body part action in the source** (for temporal localization)
- Be specific about body part and spatial configuration
- The generated results exhibit diversity across body parts, directions, and styles
- Do not omit any generated content (edit_command, new_caption, reverse_command, locate_edit_phase)
- **Output must be in English**

Output JSON format:
{{
  "variations": [
    {{
      "edit_command": "...",
      "new_caption": "...",
      "reverse_command": "...",
      "locate_edit_phase": "..."
    }}
  ]
}}

JSON Output:"""

    def build_style_prompt(self, original_caption: str, num_commands: int) -> str:
        """风格编辑 Prompt - 包含 reverse_command 和 locate_edit_phase"""
        return f"""You are generating editing commands for movement style modifications.
Source description: "{original_caption}"

Generate {num_commands} editing variations that:
- Add or modify movement style (e.g., like a zombie, quickly, slowly, stiffly, gracefully, energetically, sluggishly, robotically, like a ballet dancer, angrily, happily)
- May specify body part (e.g., wave hand quickly) or be global (e.g., perform all actions slowly)
- Use action sequence relationships (while/when/before/after/during) to specify timing

For each variation, provide:
1. edit_command: timing + style description (+ optional body part)
2. new_caption: edited description with style incorporated naturally
3. reverse_command: instruction to remove the style and revert to original movement quality
4. locate_edit_phase: **Exact substring from the source describing the action phase where style is applied**

Example:
Source: "The person walks forward and raises their hand."

Variation 1:
{{
  "edit_command": "While walking, move like a zombie with stiff limbs",
  "new_caption": "The person walks forward like a zombie with stiff limbs and raises their hand.",
  "reverse_command": "Remove zombie-like stiff movement and walk normally",
  "locate_edit_phase": "walks forward"
}}

Variation 2:
{{
  "edit_command": "When raising hand, do it quickly and energetically",
  "new_caption": "The person walks forward and quickly raises their hand in an energetic manner.",
  "reverse_command": "Change quick energetic hand raise back to normal hand raise",
  "locate_edit_phase": "raises their hand"
}}

Variation 3:
{{
  "edit_command": "Throughout the entire motion, perform slowly and gracefully like a ballet dancer",
  "new_caption": "The person slowly and gracefully walks forward like a ballet dancer and raises their hand.",
  "reverse_command": "Remove slow graceful ballet style and perform at normal speed",
  "locate_edit_phase": "walks forward and raises their hand"
}}

Requirements:
- edit_command: specify timing + style (+ optional body part) + manner description
- new_caption: naturally incorporate the style while maintaining source structure
- reverse_command: describe how to remove the added style and return to neutral/original quality
- locate_edit_phase: **MUST be a direct quote or minimal paraphrase of the action segment where style is applied** (for temporal localization)
- Focus on movement quality, manner, or expressive style
- The generated results exhibit diversity across styles (speed, emotion, character imitation, etc.)
- Do not omit any generated content (edit_command, new_caption, reverse_command, locate_edit_phase)
- **Output must be in English**

Output JSON format:
{{
  "variations": [
    {{
      "edit_command": "...",
      "new_caption": "...",
      "reverse_command": "...",
      "locate_edit_phase": "..."
    }}
  ]
}}

JSON Output:"""

    def parse_json_response(self, text: str) -> Optional[List[Dict]]:
        """健壮解析，失败返回 None"""
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

    def _generate_single(self, prompt: str, max_new_tokens: int = 1536, temperature: float = 0.7) -> Optional[
        List[Dict]]:
        """单样本生成（OOM 降级备用方案）"""
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
            logger.warning(f"[GPU {self.gpu_id}] Single sample OOM, skipping")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.warning(f"[GPU {self.gpu_id}] Single sample error: {e}")
            return None

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 1536, temperature: float = 0.7) -> List[
        Optional[List[Dict]]]:
        """批量生成，单个样本失败不影响其他样本"""
        if not prompts:
            return []

        results = [None] * len(prompts)

        try:
            texts = []
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                padding_side="left"
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
            decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for idx, text in enumerate(decoded_texts):
                try:
                    parsed = self.parse_json_response(text.strip())
                    results[idx] = parsed
                except Exception as e:
                    logger.warning(f"[GPU {self.gpu_id}] Sample {idx} parsing failed: {e}")
                    results[idx] = None

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"[GPU {self.gpu_id}] Batch OOM, falling back to single")
            torch.cuda.empty_cache()
            for idx, prompt in enumerate(prompts):
                results[idx] = self._generate_single(prompt, max_new_tokens, temperature)

        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Batch error: {e}, using single")
            for idx, prompt in enumerate(prompts):
                results[idx] = self._generate_single(prompt, max_new_tokens, temperature)

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

    def validate_locate_edit_phase(self, phase: str, original_caption: str) -> str:
        """验证 locate_edit_phase 是否合理"""
        if not phase:
            return ""

        phase = phase.strip().strip('"').strip("'")

        # 检查是否是源文本的子串（忽略大小写和标点）
        phase_clean = re.sub(r'[^\w\s]', '', phase.lower())
        orig_clean = re.sub(r'[^\w\s]', '', original_caption.lower())

        if phase_clean in orig_clean:
            return phase

        # 如果不是子串，返回原始 phase（可能是 LLM 轻微修改了措辞）
        return phase

    def process_type_slice(self, data_slice: List[tuple], edit_type: str, num_commands: int,
                           batch_size: int, temp_output: str, temp_failed_output: str,
                           is_resume_from: bool = False):
        """
        处理特定类型的数据切片 - 仿照 phase.py 的实现

        Args:
            data_slice: 待处理的数据 (key, caption) 列表
            edit_type: 'coarse', 'fine', 或 'style'
            num_commands: 每个样本生成的命令数
            batch_size: 批处理大小
            temp_output: 成功结果临时文件路径
            temp_failed_output: 失败 keys 临时文件路径
            is_resume_from: 是否是从失败文件恢复的模式
        """
        results = []
        failed_keys = []
        success_count = 0
        fail_count = 0

        # 选择对应的 prompt 构建函数
        prompt_builders = {
            'coarse': self.build_coarse_prompt,
            'fine': self.build_fine_prompt,
            'style': self.build_style_prompt
        }
        prompt_builder = prompt_builders[edit_type]

        # 断点续传检查（成功文件）
        processed_keys = set()
        if os.path.exists(temp_output):
            try:
                with open(temp_output, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    results = existing_data
                    processed_keys = {item["original_key"] for item in results}
                success_count = len(results)
                logger.info(f"[GPU {self.gpu_id}] [{edit_type}] Resumed success: {len(processed_keys)} processed")
            except Exception as e:
                logger.warning(f"[GPU {self.gpu_id}] Failed to resume success file: {e}")

        # 断点续传检查（失败文件）
        existing_failed_keys = set()
        if os.path.exists(temp_failed_output):
            try:
                with open(temp_failed_output, 'r', encoding='utf-8') as f:
                    existing_failed_keys = set(json.load(f))
                logger.info(
                    f"[GPU {self.gpu_id}] [{edit_type}] Resumed failed: {len(existing_failed_keys)} failed keys")
            except Exception as e:
                logger.warning(f"[GPU {self.gpu_id}] Failed to load failed keys file: {e}")

        # 关键修复：根据是否为重试模式决定是否跳过已失败的 keys
        if is_resume_from:
            all_processed_keys = processed_keys
            logger.info(
                f"[GPU {self.gpu_id}] [{edit_type}] Resume mode: will retry {len(existing_failed_keys)} failed keys")
        else:
            all_processed_keys = processed_keys.union(existing_failed_keys)

        # 过滤已处理数据
        remaining_data = [(k, c) for k, c in data_slice if k not in all_processed_keys]
        if not remaining_data:
            logger.info(f"[GPU {self.gpu_id}] [{edit_type}] All items already processed")
            all_failed_keys = list(existing_failed_keys.union(set(failed_keys)))
            return temp_output, temp_failed_output, success_count, fail_count, all_failed_keys

        logger.info(f"[GPU {self.gpu_id}] [{edit_type}] Processing {len(remaining_data)} new items")
        pbar = tqdm(total=len(remaining_data), desc=f"GPU{self.gpu_id}-{edit_type}", position=self.gpu_id)

        for i in range(0, len(remaining_data), batch_size):
            batch = remaining_data[i:i + batch_size]
            batch_keys = [k for k, c in batch]
            batch_captions = [c for k, c in batch]

            prompts = [prompt_builder(c, num_commands) for c in batch_captions]
            responses = self.generate_batch(prompts, max_new_tokens=1536, temperature=0.7)

            for key, orig_caption, variations in zip(batch_keys, batch_captions, responses):
                if variations is None:
                    fail_count += 1
                    failed_keys.append(key)
                    continue

                valid_vars = []
                for var in variations[:num_commands]:
                    if not isinstance(var, dict):
                        continue

                    # 检查必需字段（包含 reverse_command 和 locate_edit_phase）
                    required_fields = ["edit_command", "new_caption", "reverse_command", "locate_edit_phase"]
                    if not all(k in var and var[k] for k in required_fields):
                        continue

                    edit_cmd = str(var["edit_command"]).strip()
                    new_cap = str(var["new_caption"]).strip()
                    rev_cmd = str(var["reverse_command"]).strip()
                    loc_phase = str(var["locate_edit_phase"]).strip()

                    if len(edit_cmd) < 3 or len(new_cap) < 3 or len(rev_cmd) < 3 or len(loc_phase) < 3:
                        continue

                    # 验证 locate_edit_phase
                    validated_phase = self.validate_locate_edit_phase(loc_phase, orig_caption)

                    valid_vars.append({
                        "edit_command": edit_cmd,
                        "new_caption": self.clean_caption(new_cap),
                        "reverse_command": rev_cmd,
                        "locate_edit_phase": validated_phase,
                        "edit_type": edit_type
                    })

                if not valid_vars:
                    fail_count += 1
                    failed_keys.append(key)
                    continue

                results.append({
                    "original_key": key,
                    "original_caption": orig_caption,
                    "variations": valid_vars,
                    "edit_type": edit_type
                })
                success_count += 1

            # 定期保存
            if (i // batch_size) % 5 == 0 or i + batch_size >= len(remaining_data):
                with open(temp_output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                all_failed_keys = list(existing_failed_keys.union(set(failed_keys)))
                with open(temp_failed_output, 'w', encoding='utf-8') as f:
                    json.dump(all_failed_keys, f, ensure_ascii=False, indent=2)

            pbar.update(len(batch))

        pbar.close()
        all_failed_keys = list(existing_failed_keys.union(set(failed_keys)))

        logger.info(f"[GPU {self.gpu_id}] [{edit_type}] Done: Success={success_count}, Failed={fail_count}")
        return temp_output, temp_failed_output, success_count, fail_count, all_failed_keys


def worker_task(gpu_id: int, model_path: str, data_slice: List[tuple], edit_types: List[str],
                num_commands: int, batch_size: int, output_dir: str, result_queue, is_resume_from: bool = False):
    """
    工作进程：为每种类型生成编辑对
    """
    try:
        editor = MultiAttributeEditor(model_path, gpu_id)

        results = {}
        for edit_type in edit_types:
            temp_output = os.path.join(output_dir, f"{edit_type}_edit.gpu_{gpu_id}.tmp")
            temp_failed = os.path.join(output_dir, f"failed_{edit_type}.gpu_{gpu_id}.tmp")

            temp_file, temp_failed_file, success, fail, failed_keys = editor.process_type_slice(
                data_slice, edit_type, num_commands, batch_size, temp_output, temp_failed, is_resume_from
            )
            results[edit_type] = {
                'temp_file': temp_file,
                'temp_failed': temp_failed_file,
                'success': success,
                'fail': fail,
                'failed_keys': failed_keys
            }

        result_queue.put((gpu_id, results))
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Worker crashed: {e}", exc_info=True)
        result_queue.put((gpu_id, None))


def merge_results(temp_files: List[str], final_output: str, is_resume_mode: bool = False):
    """合并结果文件"""
    all_results = []

    # 如果是 resume 模式且最终输出文件已存在，先加载已有数据
    if is_resume_mode and os.path.exists(final_output):
        try:
            with open(final_output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_results.extend(existing_data)
                logger.info(f"Loaded existing final output {final_output}: {len(existing_data)} items")
        except Exception as e:
            logger.warning(f"Failed to load existing final output {final_output}: {e}")

    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_results.extend(data)
            except Exception as e:
                logger.error(f"Failed to load {temp_file}: {e}")

    # 按 key 去重（保留最后一个）
    seen = {}
    for item in reversed(all_results):
        key = item["original_key"]
        if key not in seen:
            seen[key] = item

    unique_results = list(seen.values())

    os.makedirs(os.path.dirname(final_output) if os.path.dirname(final_output) else '.', exist_ok=True)

    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(unique_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Final output: {final_output}")
    logger.info(f"Total items: {len(unique_results)}")
    if is_resume_mode:
        logger.info(f"  (Including items from previous run)")

    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

    return unique_results


def merge_failed_keys(temp_failed_files: List[str], final_failed_output: str):
    """合并失败 keys"""
    all_failed_keys = set()

    for temp_file in temp_failed_files:
        if temp_file and os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    keys = json.load(f)
                    all_failed_keys.update(keys)
            except Exception as e:
                logger.error(f"Failed to load {temp_file}: {e}")

    if all_failed_keys:
        os.makedirs(os.path.dirname(final_failed_output) if os.path.dirname(final_failed_output) else '.',
                    exist_ok=True)
        with open(final_failed_output, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(all_failed_keys)), f, ensure_ascii=False, indent=2)
        logger.info(f"Failed keys saved: {final_failed_output}")

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


def parse_types(types_str: str) -> List[str]:
    """解析并验证类型参数"""
    valid_types = ['coarse', 'fine', 'style']
    if not types_str:
        return valid_types

    types = [t.strip().lower() for t in types_str.split(',') if t.strip()]
    invalid = [t for t in types if t not in valid_types]

    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid type(s): {', '.join(invalid)}. "
            f"Valid options are: {', '.join(valid_types)}"
        )

    seen = set()
    unique_types = []
    for t in types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    return unique_types


def main():
    parser = argparse.ArgumentParser(
        description="Generate Multi-Attribute Editing Commands (Coarse/Fine/Style) with reverse_command and locate_edit_phase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 生成全部三种编辑对（默认行为）
  python %(prog)s --types coarse,fine,style

  # 只生成粗粒度动作类型编辑对
  python %(prog)s --types coarse

  # 生成细粒度和风格编辑两种
  python %(prog)s --types fine,style

  # 指定GPU和输出目录，只生成风格编辑
  python %(prog)s --types style --gpus 0,1 --output_dir ./style_only

  # 从之前失败的样本继续（仅 coarse 类型）
  python %(prog)s --types coarse --resume_from ./failed_coarse_keys.json

  # 快速测试模式
  python %(prog)s --types coarse --max_samples 10
        """
    )

    parser.add_argument('--input', type=str, default="../data/SnapMoGen/all_caption_clean.json",
                        help="Input JSON with all captions")
    parser.add_argument('--output_dir', type=str, default="../outputs/omni_moedit/text_pairs",
                        help="Output directory for JSON files")
    parser.add_argument('--model', type=str, default="../pretrained/Qwen3-8B",
                        help="Model path")
    parser.add_argument('--gpus', type=str, default="0,1",
                        help="GPU IDs to use, comma-separated")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size per GPU")
    parser.add_argument('--num_commands', type=int, default=6,
                        help="Number of variations per sample per type")
    parser.add_argument('--types', type=str, default="coarse,fine,style",
                        metavar="TYPE_LIST",
                        help="Comma-separated list of edit types to generate. "
                             "Options: coarse (action type), fine (body part), style (movement style). "
                             "Examples: 'coarse', 'coarse,fine', 'fine,style', 'coarse,fine,style' (all)")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to failed keys JSON file. If provided, only retry these keys for the specified types")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum samples to process (for testing)")

    args = parser.parse_args()

    try:
        edit_types = parse_types(args.types)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
        return

    if not edit_types:
        logger.error("No valid edit types specified!")
        return

    logger.info(f"Selected edit types: {edit_types}")
    logger.info(f"Using GPUs: {args.gpus}")

    os.makedirs(args.output_dir, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)

    # 读取数据
    logger.info(f"Reading input: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
    is_resume_mode = args.resume_from is not None
    if is_resume_mode:
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
    if total == 0:
        logger.error("No items to process!")
        return

    # 数据分片
    chunk_size = (total + num_gpus - 1) // num_gpus
    chunks = []

    for i, gpu_id in enumerate(gpu_ids):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = items[start:end]
        chunks.append(chunk)
        logger.info(f"GPU {gpu_id}: items {start}-{end - 1} ({len(chunk)} items)")

    # 关键修复：如果是重试模式，清理临时失败文件，避免历史失败记录干扰
    temp_files_per_type = {t: [] for t in edit_types}
    temp_failed_per_type = {t: [] for t in edit_types}

    if is_resume_mode:
        for edit_type in edit_types:
            for gpu_id in gpu_ids:
                temp_failed = os.path.join(args.output_dir, f"failed_{edit_type}.gpu_{gpu_id}.tmp")
                if os.path.exists(temp_failed):
                    try:
                        os.remove(temp_failed)
                        logger.info(f"Cleaned up temp failed file for retry: {temp_failed}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp failed file {temp_failed}: {e}")

    # 启动多进程
    result_queue = mp.Queue()
    processes = []

    for gpu_id, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        p = mp.Process(
            target=worker_task,
            args=(gpu_id, args.model, chunk, edit_types, args.num_commands,
                  args.batch_size, args.output_dir, result_queue, is_resume_mode)
        )
        p.start()
        processes.append(p)

    # 收集结果
    completed = 0
    all_results = {t: {'temp_files': [], 'temp_failed': [], 'success': 0, 'fail': 0, 'failed_keys': []}
                   for t in edit_types}

    while completed < len(processes):
        gpu_id, results = result_queue.get()
        completed += 1

        if results is None:
            logger.error(f"GPU {gpu_id} failed completely")
            continue

        for edit_type, res in results.items():
            all_results[edit_type]['temp_files'].append(res['temp_file'])
            all_results[edit_type]['temp_failed'].append(res['temp_failed'])
            all_results[edit_type]['success'] += res['success']
            all_results[edit_type]['fail'] += res['fail']
            all_results[edit_type]['failed_keys'].extend(res['failed_keys'])

    # 等待进程结束
    for p in processes:
        p.join()

    # 合并每种类型的结果
    final_outputs = {}
    for edit_type in edit_types:
        res = all_results[edit_type]

        final_output = os.path.join(args.output_dir, f"{edit_type}_edit_pairs.json")
        final_failed = os.path.join(args.output_dir, f"failed_{edit_type}_keys.json")

        merge_results(res['temp_files'], final_output, is_resume_mode)
        merge_failed_keys(res['temp_failed'], final_failed)

        final_outputs[edit_type] = final_output

        logger.info(f"\n[{edit_type.upper()}] Summary:")
        logger.info(f"  - Success: {res['success']}")
        logger.info(f"  - Failed: {res['fail']}")
        if res['success'] + res['fail'] > 0:
            logger.info(f"  - Rate: {100 * res['success'] / (res['success'] + res['fail']):.1f}%")
        if res['fail'] > 0:
            logger.info(f"  - To retry failed samples: --types {edit_type} --resume_from {final_failed}")

    logger.info(f"\nAll edit pairs generated in: {args.output_dir}")
    for t, path in final_outputs.items():
        logger.info(f"  - {t}: {path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
