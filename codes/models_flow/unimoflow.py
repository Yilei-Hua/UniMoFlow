import torch
import torch.nn as nn
from .tools.unimoflow_transformer import UniMoFlowTransformer
from typing import Dict, Any, Optional
from .tools.t5 import T5EncoderModel
import torch.nn.functional as F
from collections import OrderedDict


class UniMoFlow(nn.Module):
    """
    混合训练模型：支持指令式编辑（edit）和文本条件生成（gen）两种模式

    - edit 模式：使用 source + text 预测 target（与 FlowMatchingEditModel 一致）
    - gen 模式：source 置零作为占位符，仅用 text 生成 motion，attention_mask 中 target 不看 source
    """

    def __init__(
            self,
            checkpoint_path="deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path="deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl",
            input_dim=256,
            hidden_dim=1024,
            ffn_dim=4096,
            num_layers=16,
            num_heads=16,
            num_registers=0,
            text_dim=4096,
            text_len=512,
            dropout_prob=0.1,
            noise_steps=10,
            cfg_scale=7.5,
            time_scale=10.0,
            prediction_type="vel",
            use_logit_normal=False,
            logit_mean=0.0,
            logit_std=1.0,
            use_role_tags=True,
            fusion_schedule="asymmetric",
            param_dtype=torch.bfloat16,
            spatial_dim=1,
            use_dynamic_depth=False,
            gen_loss_weight=1.0,
            edit_loss_weight=1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.text_dim = text_dim
        self.text_len = text_len
        self.dropout_prob = dropout_prob
        self.prediction_type = prediction_type
        self.noise_steps = noise_steps
        self.cfg_scale = cfg_scale
        self.time_scale = time_scale
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.use_logit_normal = use_logit_normal
        self.param_dtype = param_dtype
        self.spatial_dim = spatial_dim
        self.gen_loss_weight = gen_loss_weight
        self.edit_loss_weight = edit_loss_weight

        # 1. 文本编码器初始化
        self.text_encoder = None
        if T5EncoderModel is not None:
            try:
                self.text_encoder = T5EncoderModel(
                    text_len=self.text_len,
                    dtype=param_dtype,
                    device=torch.device("cpu"),
                    checkpoint_path=checkpoint_path,
                    tokenizer_path=tokenizer_path,
                    shard_fn=None,
                )
            except Exception as e:
                print(f"Warning: Could not load T5 model: {e}")

        # 2. 缓存机制
        self.max_cache_size = 100000
        self.text_cache = OrderedDict()

        # 3. 共享 Transformer 主干
        self.model = UniMoFlowTransformer(
            in_dim=input_dim,
            out_dim=input_dim,
            dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_registers=num_registers,
            text_dim=text_dim,
            use_role_tags=use_role_tags,
            use_text_tags=True,
            dropout=0.0,
            use_dynamic_depth=use_dynamic_depth,
        )

        # 4. 编辑门控：基于物理时间步的标量门控（仅 edit 模式使用）
        self.edit_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.edit_gate[0].weight)
        nn.init.zeros_(self.edit_gate[0].bias)
        nn.init.zeros_(self.edit_gate[2].weight)
        with torch.no_grad():
            self.edit_gate[2].bias.fill_(2.0)

    def load_weights(self, path):
        """对齐 diffusion_standard.py 的权重加载，兼容旧版 4×dim adaLN"""
        sd = torch.load(path, map_location="cpu")

        dim = self.model.dim
        for key in list(sd.keys()):
            if 'modulation.1.weight' in key or 'modulation.1.bias' in key:
                old_tensor = sd[key]
                if old_tensor.shape[0] == 4 * dim:
                    if len(old_tensor.shape) == 2:
                        new_shape = (6 * dim, old_tensor.shape[1])
                    else:
                        new_shape = (6 * dim,)
                    new_tensor = torch.zeros(new_shape, dtype=old_tensor.dtype, device=old_tensor.device)
                    new_tensor[:4 * dim] = old_tensor
                    sd[key] = new_tensor
                    print(f"[Checkpoint Adapt] Expanded {key}: {old_tensor.shape} -> {new_tensor.shape}")

        for key in list(sd.keys()):
            if key.endswith("norm_final.scale"):
                sd[key.replace("norm_final", "edit_norm_final")] = sd[key]
                sd[key.replace("norm_final", "gen_norm_final")] = sd[key].clone()
                del sd[key]
            elif key.endswith("norm_final.shift"):
                sd[key.replace("norm_final", "edit_norm_final")] = sd[key]
                sd[key.replace("norm_final", "gen_norm_final")] = sd[key].clone()
                del sd[key]
            elif key.endswith("head.weight") and not key.endswith(("edit_head.weight", "gen_head.weight")):
                sd[key.replace("head.", "edit_head.")] = sd[key]
                sd[key.replace("head.", "gen_head.")] = sd[key].clone()
                del sd[key]
            elif key.endswith("head.bias") and not key.endswith(("edit_head.bias", "gen_head.bias")):
                sd[key.replace("head.", "edit_head.")] = sd[key]
                sd[key.replace("head.", "gen_head.")] = sd[key].clone()
                del sd[key]

        self.model.load_state_dict(sd, strict=False)

    def _process_spatial_input(self, tensor):
        """统一处理4D/3D输入"""
        if tensor.dim() == 4:
            B, T_time, P, C = tensor.shape
            tensor = tensor.reshape(B, T_time * P, C)
            return tensor, B, T_time, P, C, T_time * P
        elif tensor.dim() == 3:
            B, T, C = tensor.shape
            P = 1
            T_time = T
            return tensor, B, T_time, P, C, T
        else:
            raise ValueError(f"Unexpected tensor dim: {tensor.dim()}, shape: {tensor.shape}")

    def encode_text_with_cache(self, text_list, device):
        """与 diffusion_standard.py 完全一致的文本编码逻辑"""
        if self.text_encoder is None:
            return [torch.zeros(self.text_len, self.text_dim, device=device, dtype=self.param_dtype)
                    for _ in text_list]

        text_features = [None] * len(text_list)
        indices_to_encode = []
        texts_to_encode = []

        for i, text in enumerate(text_list):
            if text in self.text_cache:
                self.text_cache.move_to_end(text)
                cached = self.text_cache[text].to(device)
                if cached.dtype != self.param_dtype:
                    cached = cached.to(self.param_dtype)
                text_features[i] = cached
            else:
                indices_to_encode.append(i)
                texts_to_encode.append(text)

        if texts_to_encode:
            if hasattr(self.text_encoder, 'model'):
                self.text_encoder.model.to(device)

            try:
                encoded = self.text_encoder(texts_to_encode, device)
                if not isinstance(encoded, list):
                    encoded = [encoded[i] for i in range(len(texts_to_encode))]

                for idx, text, feature in zip(indices_to_encode, texts_to_encode, encoded):
                    if feature.dtype != self.param_dtype:
                        feature = feature.to(self.param_dtype)

                    if len(self.text_cache) >= self.max_cache_size:
                        self.text_cache.popitem(last=False)

                    feature_cpu = feature.cpu()
                    self.text_cache[text] = feature_cpu
                    text_features[idx] = feature
            except Exception as e:
                print(f"Error encoding texts: {e}")
                for idx in indices_to_encode:
                    text_features[idx] = torch.zeros(self.text_len, self.text_dim,
                                                     device=device, dtype=self.param_dtype)

        return text_features

    def get_dynamic_shift_with_jacobian(self, t, actual_lengths):
        """与 diffusion_standard.py 一致的动态时间偏移"""
        base_length = 16.0
        if not isinstance(actual_lengths, torch.Tensor):
            actual_lengths = torch.tensor(actual_lengths, device=t.device, dtype=t.dtype)

        s_i = torch.sqrt(actual_lengths.float() / base_length).clamp(min=1.0)

        if t.ndim > s_i.ndim:
            s_i = s_i.view(s_i.shape[0], *([1] * (t.ndim - s_i.ndim)))

        denominator = 1 + (s_i - 1) * t
        t_shifted = (s_i * t) / denominator
        jacobian = s_i / (denominator ** 2)

        return t_shifted, jacobian

    def forward_edit(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        编辑模式前向传播（与 FlowMatchingEditModel.forward 完全一致）

        使用 source + edit_text → 预测 target
        """
        source = x["source"]
        target = x["target"]
        edit_text = x.get("edit_text", [""] * source.shape[0])
        feature_length = x.get("length", x.get("feature_length", None))

        source, B, T_time, P, C, T = self._process_spatial_input(source)
        if target.dim() == 4:
            target = target.reshape(B, T_time * P, C)
        elif target.dim() == 3 and P > 1:
            pass

        device = source.device

        text_embeddings_list = self.encode_text_with_cache(edit_text, device)
        text_embeddings_list = [t.to(self.param_dtype) for t in text_embeddings_list]

        if self.training and self.dropout_prob > 0:
            null_emb_list = self.encode_text_with_cache([""], device)
            null_emb = null_emb_list[0].to(self.param_dtype)
            mask = torch.rand(B, device=device) < self.dropout_prob
            for i in range(B):
                if mask[i]:
                    text_embeddings_list[i] = null_emb

        if self.training and self.use_logit_normal:
            u = torch.randn(B, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(u)
        else:
            t = torch.rand(B, device=device)

        t = torch.clamp(t, min=0.0001, max=0.9999)

        if feature_length is not None:
            if not isinstance(feature_length, torch.Tensor):
                feature_length = torch.tensor(feature_length, device=device, dtype=torch.long)
            token_length = feature_length * P
            t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t, token_length)
        else:
            token_length = torch.full((B,), T, dtype=torch.long, device=device)
            t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t, token_length)

        t_expanded = t.view(B, 1, 1)
        x_1 = target
        x_0 = torch.randn_like(x_1)
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

        target_v_physical = x_1 - x_0
        jacobian_expand = jacobian.view(B, 1, 1)
        target_v_shifted = target_v_physical / (jacobian_expand + 1e-6)

        pred = self.model(
            x_target_noisy=x_t.to(self.param_dtype),
            x_source_clean=source.to(self.param_dtype),
            t=t_shifted * self.time_scale,
            context=text_embeddings_list,
            seq_lens=token_length,
            attention_mode="edit",
        )

        pred = pred.float()
        target_v_shifted = target_v_shifted.float()

        if self.prediction_type == "vel":
            loss = (pred - target_v_shifted) ** 2
        elif self.prediction_type == "x0":
            loss = (pred - x_1.float()) ** 2
        elif self.prediction_type == "noise":
            loss = (pred - x_0.float()) ** 2
        else:
            loss = (pred - target_v_shifted) ** 2

        if token_length is not None:
            mask = torch.arange(T, device=device).expand(B, T) < token_length.unsqueeze(1)
            loss = loss * mask.unsqueeze(-1)
            num_elements = mask.sum() * C
            return {"total": loss.sum() / (num_elements + 1e-6), "mse": loss.sum() / (num_elements + 1e-6), "mode": "edit"}

        return {"total": loss.mean(), "mse": loss.mean(), "mode": "edit"}

    def forward_gen(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        生成模式前向传播：标准 Flow Matching

        - source 置零作为占位符
        - attention_mode="gen"，target 不看 source
        - 与 forward_edit 逻辑对称，但 source=zeros
        """
        target = x["target"]
        text = x.get("text", [""] * target.shape[0])
        feature_length = x.get("length", x.get("feature_length", None))

        target, B, T_time, P, C, T = self._process_spatial_input(target)
        device = target.device

        # source 置零（与 target 同形状），作为占位符
        source = torch.zeros_like(target)

        text_embeddings_list = self.encode_text_with_cache(text, device)
        text_embeddings_list = [t.to(self.param_dtype) for t in text_embeddings_list]

        if self.training and self.dropout_prob > 0:
            null_emb_list = self.encode_text_with_cache([""], device)
            null_emb = null_emb_list[0].to(self.param_dtype)
            mask = torch.rand(B, device=device) < self.dropout_prob
            for i in range(B):
                if mask[i]:
                    text_embeddings_list[i] = null_emb

        if self.training and self.use_logit_normal:
            u = torch.randn(B, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(u)
        else:
            t = torch.rand(B, device=device)

        t = torch.clamp(t, min=0.0001, max=0.9999)

        if feature_length is not None:
            if not isinstance(feature_length, torch.Tensor):
                feature_length = torch.tensor(feature_length, device=device, dtype=torch.long)
            token_length = feature_length * P
            t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t, token_length)
        else:
            token_length = torch.full((B,), T, dtype=torch.long, device=device)
            t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t, token_length)

        t_expanded = t.view(B, 1, 1)
        x_1 = target
        x_0 = torch.randn_like(x_1)
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

        target_v_physical = x_1 - x_0
        jacobian_expand = jacobian.view(B, 1, 1)
        target_v_shifted = target_v_physical / (jacobian_expand + 1e-6)

        pred = self.model(
            x_target_noisy=x_t.to(self.param_dtype),
            x_source_clean=source.to(self.param_dtype),
            t=t_shifted * self.time_scale,
            context=text_embeddings_list,
            seq_lens=token_length,
            attention_mode="gen",
        )

        pred = pred.float()
        target_v_shifted = target_v_shifted.float()

        if self.prediction_type == "vel":
            loss = (pred - target_v_shifted) ** 2
        elif self.prediction_type == "x0":
            loss = (pred - x_1.float()) ** 2
        elif self.prediction_type == "noise":
            loss = (pred - x_0.float()) ** 2
        else:
            loss = (pred - target_v_shifted) ** 2

        if token_length is not None:
            mask = torch.arange(T, device=device).expand(B, T) < token_length.unsqueeze(1)
            loss = loss * mask.unsqueeze(-1)
            num_elements = mask.sum() * C
            return {"total": loss.sum() / (num_elements + 1e-6), "mse": loss.sum() / (num_elements + 1e-6), "mode": "gen"}

        return {"total": loss.mean(), "mse": loss.mean(), "mode": "gen"}

    def forward(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        统一前向传播：根据 mode 分发到 edit 或 gen

        输入 x 中应包含 "mode" 字段：
        - "edit": 使用 forward_edit（source + edit_text → target）
        - "gen":  使用 forward_gen（text → target，source 置零）
        """
        mode = x.get("mode", "edit")

        if mode == "edit":
            loss_dict = self.forward_edit(x)
            loss_dict["total"] = loss_dict["total"] * self.edit_loss_weight
            loss_dict["mse"] = loss_dict["mse"] * self.edit_loss_weight
        elif mode == "gen":
            loss_dict = self.forward_gen(x)
            loss_dict["total"] = loss_dict["total"] * self.gen_loss_weight
            loss_dict["mse"] = loss_dict["mse"] * self.gen_loss_weight
        else:
            raise ValueError(f"Unknown mode: {mode}, expected 'edit' or 'gen'")

        return loss_dict

    def forward_with_geometry(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """统一训练入口（兼容 EditDiffusionTrainer 接口）"""
        return self.forward(x)

    @torch.no_grad()
    def generate_edit(
            self,
            x: Dict[str, Any],
            num_denoise_steps: Optional[int] = None,
            cfg_scale: Optional[float] = None,
    ):
        """
        编辑模式生成：从 source + edit_text 生成编辑后的动作

        与 FlowMatchingEditModel.generate 完全一致
        """
        device = next(self.parameters()).device
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale

        source = x["source"]
        edit_text = x.get("edit_text", [""] * source.shape[0])
        feature_length = x.get("length", x.get("feature_length", None))

        source, B, T_time, P, C, T = self._process_spatial_input(source)
        input_was_4d = (x["source"].dim() == 4)

        steps = num_denoise_steps if num_denoise_steps is not None else self.noise_steps

        text_embeddings = self.encode_text_with_cache(edit_text, device)
        text_embeddings = [t.to(self.param_dtype) for t in text_embeddings]

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            null_emb = self.encode_text_with_cache([""], device)[0].to(self.param_dtype)
            combined_text = text_embeddings + [null_emb] * B
        else:
            combined_text = text_embeddings

        x_t = torch.randn(B, T, C, device=device, dtype=self.param_dtype)

        t_physical = torch.linspace(0, 1, steps + 1, device=device)
        t_physical = t_physical.unsqueeze(0).expand(B, -1)

        seq_lens_time = feature_length if feature_length is not None else torch.full((B,), T_time, device=device)
        if not isinstance(seq_lens_time, torch.Tensor):
            seq_lens_time = torch.tensor(seq_lens_time, device=device, dtype=torch.long)
        seq_lens = seq_lens_time * P

        t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t_physical, seq_lens)

        if do_cfg:
            seq_lens_cfg = torch.cat([seq_lens, seq_lens], dim=0)
            jacobian_cfg = torch.cat([jacobian, jacobian], dim=0)
        else:
            seq_lens_cfg = seq_lens
            jacobian_cfg = jacobian

        for i in range(steps):
            t_curr_physical = t_physical[:, i]
            t_next_physical = t_physical[:, i + 1]
            t_curr_shifted = t_shifted[:, i]
            t_next_shifted = t_shifted[:, min(i + 1, t_shifted.shape[1] - 1)]

            if do_cfg:
                x_in = torch.cat([x_t, x_t], dim=0)
                t_in = torch.cat([t_curr_shifted, t_curr_shifted], dim=0)
                source_in = torch.cat([source, source], dim=0).to(self.param_dtype)
            else:
                x_in = x_t
                t_in = t_curr_shifted
                source_in = source.to(self.param_dtype)

            active_layers = None
            if self.model.use_dynamic_depth:
                progress = i / steps
                if progress < 0.5:
                    active_layers = None
                elif progress < 0.75:
                    active_layers = max(1, int(len(self.model.blocks) * 0.75))
                else:
                    active_layers = max(1, int(len(self.model.blocks) * 0.5))

            pred = self.model(
                x_target_noisy=x_in,
                x_source_clean=source_in,
                t=t_in * self.time_scale,
                context=combined_text,
                seq_lens=seq_lens_cfg,
                active_layers=active_layers,
                attention_mode="edit",
            )

            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                jacobian_curr = jacobian_cfg[:, i][:B]
            else:
                jacobian_curr = jacobian[:, i]

            if self.prediction_type == "vel":
                v_shifted = pred
            elif self.prediction_type == "x0":
                t_curr_expanded = t_curr_physical.view(-1, 1, 1)
                jacobian_expanded = jacobian_curr.view(-1, 1, 1)
                v_shifted = (pred - x_t) / (1 - t_curr_expanded + 1e-6) / jacobian_expanded
            elif self.prediction_type == "noise":
                t_curr_expanded = t_curr_physical.view(-1, 1, 1)
                jacobian_expanded = jacobian_curr.view(-1, 1, 1)
                v_shifted = (x_t - pred) / (t_curr_expanded + 1e-6) / jacobian_expanded

            dt_shifted = (t_next_shifted - t_curr_shifted).view(B, 1, 1)
            x_t = x_t + v_shifted * dt_shifted

        if input_was_4d:
            x_t = x_t.reshape(B, T_time, P, C)
            source = source.reshape(B, T_time, P, C)

        return {"generated": x_t, "source": source, "edit_text": edit_text}

    @torch.no_grad()
    def generate_gen(
            self,
            x: Dict[str, Any],
            num_denoise_steps: Optional[int] = None,
            cfg_scale: Optional[float] = None,
    ):
        """
        生成模式推理：从文本生成动作

        - source 置零作为占位符
        - attention_mode="gen"，target 不看 source
        - 与 generate_edit 对称，但 source=zeros
        """
        device = next(self.parameters()).device
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale

        text = x.get("text", [""])
        feature_length = x.get("length", x.get("feature_length", None))

        B = len(text)
        if feature_length is not None:
            if isinstance(feature_length, torch.Tensor):
                N = feature_length.max().item()
            else:
                N = max(feature_length)
        else:
            N = 50

        P = self.spatial_dim
        C = self.input_dim
        T = int(N) * P
        steps = num_denoise_steps if num_denoise_steps is not None else self.noise_steps

        # source 置零
        source = torch.zeros(B, T, C, device=device, dtype=self.param_dtype)

        text_embeddings = self.encode_text_with_cache(text, device)
        text_embeddings = [t.to(self.param_dtype) for t in text_embeddings]

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            null_emb = self.encode_text_with_cache([""], device)[0].to(self.param_dtype)
            combined_text = text_embeddings + [null_emb] * B
        else:
            combined_text = text_embeddings

        x_t = torch.randn(B, T, C, device=device, dtype=self.param_dtype)

        t_physical = torch.linspace(0, 1, steps + 1, device=device)
        t_physical = t_physical.unsqueeze(0).expand(B, -1)

        seq_lens = feature_length * P if feature_length is not None else torch.full((B,), T, device=device, dtype=torch.long)
        t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t_physical, seq_lens)

        if do_cfg:
            seq_lens_cfg = torch.cat([seq_lens, seq_lens], dim=0)
            jacobian_cfg = torch.cat([jacobian, jacobian], dim=0)
        else:
            seq_lens_cfg = seq_lens
            jacobian_cfg = jacobian

        for i in range(steps):
            t_curr_physical = t_physical[:, i]
            t_next_physical = t_physical[:, i + 1]
            t_curr_shifted = t_shifted[:, i]
            t_next_shifted = t_shifted[:, min(i + 1, t_shifted.shape[1] - 1)]

            if do_cfg:
                x_in = torch.cat([x_t, x_t], dim=0)
                t_in = torch.cat([t_curr_shifted, t_curr_shifted], dim=0)
                source_in = torch.cat([source, source], dim=0)
            else:
                x_in = x_t
                t_in = t_curr_shifted
                source_in = source

            active_layers = None
            if self.model.use_dynamic_depth:
                progress = i / steps
                if progress < 0.5:
                    active_layers = None
                elif progress < 0.75:
                    active_layers = max(1, int(len(self.model.blocks) * 0.75))
                else:
                    active_layers = max(1, int(len(self.model.blocks) * 0.5))

            pred = self.model(
                x_target_noisy=x_in,
                x_source_clean=source_in,
                t=t_in * self.time_scale,
                context=combined_text,
                seq_lens=seq_lens_cfg,
                active_layers=active_layers,
                attention_mode="gen",
            )

            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                jacobian_curr = jacobian_cfg[:, i][:B]
            else:
                jacobian_curr = jacobian[:, i]

            if self.prediction_type == "vel":
                v_shifted = pred
            elif self.prediction_type == "x0":
                t_curr_expanded = t_curr_physical.view(-1, 1, 1)
                jacobian_expanded = jacobian_curr.view(-1, 1, 1)
                v_shifted = (pred - x_t) / (1 - t_curr_expanded + 1e-6) / jacobian_expanded
            elif self.prediction_type == "noise":
                t_curr_expanded = t_curr_physical.view(-1, 1, 1)
                jacobian_expanded = jacobian_curr.view(-1, 1, 1)
                v_shifted = (x_t - pred) / (t_curr_expanded + 1e-6) / jacobian_expanded

            dt_shifted = (t_next_shifted - t_curr_shifted).view(B, 1, 1)
            x_t = x_t + v_shifted * dt_shifted

        return {"generated": x_t, "text": text}

    @torch.no_grad()
    def generate(
            self,
            x: Dict[str, Any],
            num_denoise_steps: Optional[int] = None,
            cfg_scale: Optional[float] = None,
    ):
        """
        统一生成入口：根据输入自动路由到 edit 或 gen 模式

        兼容 evaluate_edit_model 和 evaluation_diffusion_model 的调用约定：
        - 如果 x 包含 "source" 和 "edit_text" → 路由到 generate_edit
        - 如果 x 包含 "text" → 路由到 generate_gen
        """
        if "source" in x and "edit_text" in x:
            return self.generate_edit(x, num_denoise_steps, cfg_scale)
        elif "text" in x:
            return self.generate_gen(x, num_denoise_steps, cfg_scale)
        else:
            raise ValueError(
                f"Cannot determine generation mode from input keys: {list(x.keys())}. "
                f"Expected 'source'+'edit_text' for edit mode or 'text' for gen mode."
            )

    @torch.no_grad()
    def flow_edit(
            self,
            x: Dict[str, Any],
            num_denoise_steps: Optional[int] = None,
            cfg_scale: Optional[float] = None,
    ):
        """
        UniMoFlow-FlowEdit 采样（编辑门控 + 动态深度）

        与 FlowMatchingEditModel.flow_edit 完全一致
        """
        device = next(self.parameters()).device
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale
        steps = num_denoise_steps if num_denoise_steps is not None else self.noise_steps

        source = x["source"]
        edit_text = x.get("edit_text", [""] * source.shape[0])
        feature_length = x.get("length", None)

        source, B, T_time, P, C, T = self._process_spatial_input(source)
        input_was_4d = (x["source"].dim() == 4)

        edit_emb = self.encode_text_with_cache(edit_text, device)
        edit_emb = [t.to(self.param_dtype) for t in edit_emb]
        null_emb = self.encode_text_with_cache([""], device)[0].to(self.param_dtype)

        epsilon = torch.randn(B, T, C, device=device, dtype=self.param_dtype)

        x_current = source.clone().to(self.param_dtype)

        t_physical = torch.linspace(0, 1, steps + 1, device=device)
        t_physical = t_physical.unsqueeze(0).expand(B, -1)

        seq_lens_time = feature_length if feature_length is not None else torch.full((B,), T_time, device=device)
        if not isinstance(seq_lens_time, torch.Tensor):
            seq_lens_time = torch.tensor(seq_lens_time, device=device, dtype=torch.long)
        seq_lens = seq_lens_time * P

        t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t_physical, seq_lens)

        for i in range(steps):
            t_curr_phys = t_physical[:, i].view(B, 1, 1)
            t_curr_shifted = t_shifted[:, i]
            t_next_shifted = t_shifted[:, i + 1]
            dt_shifted = (t_next_shifted - t_curr_shifted).view(B, 1, 1)
            jacobian_curr = jacobian[:, i].view(B, 1, 1)

            z_fe = t_curr_phys * x_current + (1 - t_curr_phys) * epsilon

            x_in = torch.cat([z_fe, z_fe], dim=0)
            t_in = torch.cat([t_curr_shifted, t_curr_shifted], dim=0)
            source_in = torch.cat([source, source], dim=0).to(self.param_dtype)
            context = edit_emb + [null_emb] * B

            active_layers = None
            if self.model.use_dynamic_depth:
                progress = i / steps
                if progress < 0.5:
                    active_layers = None
                elif progress < 0.75:
                    active_layers = max(1, int(len(self.model.blocks) * 0.75))
                else:
                    active_layers = max(1, int(len(self.model.blocks) * 0.5))

            pred = self.model(
                x_target_noisy=x_in,
                x_source_clean=source_in,
                t=t_in * self.time_scale,
                context=context,
                seq_lens=torch.cat([seq_lens, seq_lens]),
                active_layers=active_layers,
                attention_mode="edit",
            )

            pred_cond, pred_uncond = pred.chunk(2, dim=0)

            def to_shifted_velocity(pred_output):
                if self.prediction_type == "vel":
                    return pred_output
                elif self.prediction_type == "x0":
                    return (pred_output - z_fe) / (1 - t_curr_phys + 1e-6) / jacobian_curr
                elif self.prediction_type == "noise":
                    return (z_fe - pred_output) / (t_curr_phys + 1e-6) / jacobian_curr
                return pred_output

            v_cond = to_shifted_velocity(pred_cond)
            v_uncond = to_shifted_velocity(pred_uncond)

            v_edit = v_uncond + cfg_scale * (v_cond - v_uncond)
            v_recon = v_uncond

            gate_input = t_curr_phys.view(B, 1)
            edit_mag = self.edit_gate(gate_input).view(B, 1, 1)
            delta_v = edit_mag * (v_edit - v_recon)

            x_current = x_current + delta_v * dt_shifted

        if input_was_4d:
            x_current = x_current.reshape(B, T_time, P, C)
            source = source.reshape(B, T_time, P, C)

        return {
            "generated": x_current,
            "source": source,
            "edit_text": edit_text
        }
