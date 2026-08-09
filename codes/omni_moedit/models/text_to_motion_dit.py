# --- START OF FILE diffusion_standard.py ---
import torch
import torch.nn as nn
from omni_moedit.models.motion_dit_backbone import OmniMoEditDiTBackbone
from models_flow.tools.t5 import T5EncoderModel
from collections import OrderedDict
from typing import List, Dict, Any

class OmniMoEditDiT(nn.Module):
    def __init__(
            self,
            checkpoint_path="deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path="deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl",
            input_dim=256,
            hidden_dim=1024,
            ffn_dim=4096,
            num_layers=16,
            num_heads=16,
            text_dim=4096,
            text_len=512,
            dropout_prob=0.1,
            chunk_size=16,
            noise_steps=10,
            drop_out=0.1,
            cfg_scale=5.0,
            time_scale=10.0,
            prediction_type="vel",
            use_text_cond=True,
            causal=False,
            use_logit_normal=False,
            logit_mean=0.0,
            logit_std=1.0,
            spatial_dim=1,          # 默认配置，仅用于 generate/stream_generate
    ):
        super().__init__()

        self.input_dim = input_dim
        self.text_dim = text_dim
        self.text_len = text_len
        self.dropout_prob = dropout_prob
        self.prediction_type = prediction_type
        self.chunk_size = chunk_size
        self.noise_steps = noise_steps
        self.drop_out = drop_out
        self.cfg_scale = cfg_scale
        self.use_text_cond = use_text_cond
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.use_logit_normal = use_logit_normal
        self.time_scale = time_scale
        self.spatial_dim = spatial_dim   # 仅作为 generate/stream_generate 的默认值

        # 1. Initialize Text Encoder (T5)
        self.text_encoder = None
        if T5EncoderModel is not None:
            try:
                self.text_encoder = T5EncoderModel(
                    text_len=self.text_len,
                    dtype=torch.bfloat16,
                    device=torch.device("cpu"),
                    checkpoint_path=checkpoint_path,
                    tokenizer_path=tokenizer_path,
                    shard_fn=None,
                )
            except Exception as e:
                print(f"Warning: Could not load T5 model: {e}")

        # 2. Cache
        self.max_cache_size = 15000
        self.text_cache = OrderedDict()
        self.param_dtype = torch.float32

        # 3. Initialize Omni-MoEdit DiT backbone (1D Modified)
        self.model = OmniMoEditDiTBackbone(
            model_type="t2v",
            in_dim=input_dim,
            out_dim=input_dim,
            dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            text_dim=text_dim,
            text_len=text_len,
            causal=causal
        )

    def load_weights(self, path):
        sd = torch.load(path, map_location="cpu")
        self.model.load_state_dict(sd)

    def encode_text_with_cache(self, text_list, device):
        if self.text_encoder is None:
            return [torch.zeros(self.text_len, self.text_dim, device=device) for _ in text_list]

        text_features = [None] * len(text_list)
        indices_to_encode = []
        texts_to_encode = []

        for i, text in enumerate(text_list):
            if text in self.text_cache:
                self.text_cache.move_to_end(text)
                text_features[i] = self.text_cache[text].to(device)
            else:
                indices_to_encode.append(i)
                texts_to_encode.append(text)

        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)
            for idx, text, feature in zip(indices_to_encode, texts_to_encode, encoded):
                if len(self.text_cache) >= self.max_cache_size:
                    self.text_cache.popitem(last=False)
                feature_cpu = feature.cpu()
                self.text_cache[text] = feature_cpu
                text_features[idx] = feature
        return text_features

    def get_dynamic_shift_with_jacobian(self, t, actual_lengths):
        base_length = 16.0
        s_i = torch.sqrt(actual_lengths.float() / base_length).clamp(min=1.0)
        if t.ndim > s_i.ndim:
            s_i = s_i.view(s_i.shape[0], *([1] * (t.ndim - 1)))
        t_shifted = (s_i * t) / (1 + (s_i - 1) * t)
        jacobian = s_i / ((1 + (s_i - 1) * t) ** 2)
        return t_shifted, jacobian

    def forward(self, x: Dict[str, Any]):
        feature = x["feature"]  # [B, T, D] or [B, T, P, D]
        text = x.get("text", [""] * feature.shape[0])
        feature_length = x.get("feature_length", None)  # 时间长度语义，不乘 P
        device = feature.device

        # 根据当前输入形状推断局部 spatial_dim，绝不污染 self.spatial_dim
        if feature.dim() == 4:
            B, T, P, D = feature.shape
            feature = feature.reshape(B, T * P, D)
            spatial_dim = P
        elif feature.dim() == 3:
            B, T, D = feature.shape
            spatial_dim = 1
        else:
            raise ValueError(f"Unexpected feature dim: {feature.dim()}, shape: {feature.shape}")

        B, T, C = feature.shape

        # 内部 token 级别长度 = 时间步 × 部位数
        token_length = None
        if feature_length is not None:
            token_length = feature_length * spatial_dim

        # 1. Text Encoding
        if self.use_text_cond:
            text_embeddings_list = self.encode_text_with_cache(text, device)
        else:
            text_embeddings_list = self.encode_text_with_cache([""] * B, device)
        text_embeddings_list = [t.to(self.param_dtype) for t in text_embeddings_list]

        # 2. Conditional Dropout
        if self.training and self.dropout_prob > 0:
            null_emb_list = self.encode_text_with_cache([""], device)
            null_emb = null_emb_list[0].to(self.param_dtype)
            mask = torch.rand(B, device=device) < self.dropout_prob
            for i in range(B):
                if mask[i]:
                    text_embeddings_list[i] = null_emb

        # 3. Flow Matching Setup
        if self.training and self.use_logit_normal:
            u = torch.randn(B, device=device) * self.logit_std + self.logit_mean
            t_physical = torch.sigmoid(u)
        else:
            t_physical = torch.rand(B, device=device)

        # time shift 基于 token 长度（实际参与计算的 token 数）
        if token_length is not None:
            t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t_physical, token_length)
        else:
            token_length = torch.full((B,), T, dtype=t_physical.dtype, device=device)
            t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t_physical, token_length)

        t_physical_expand = t_physical.view(B, 1, 1)
        x_1 = feature
        x_0 = torch.randn_like(x_1)
        x_t = t_physical_expand * x_1 + (1 - t_physical_expand) * x_0

        target_v_physical = x_1 - x_0
        jacobian_expand = jacobian.view(B, 1, 1)
        target_v_shifted = target_v_physical / jacobian_expand

        # 模型预测：传入 token 长度（seq_lens）
        pred = self.model(x_t, t_shifted * self.time_scale, text_embeddings_list, seq_lens=token_length)

        # 损失计算
        if self.prediction_type == "vel":
            loss = (pred - target_v_shifted) ** 2
        elif self.prediction_type == "x0":
            loss = (pred - x_1) ** 2
        elif self.prediction_type == "noise":
            loss = (pred - x_0) ** 2
        else:
            loss = (pred - target_v_shifted) ** 2

        # Masking：基于 token 长度
        if token_length is not None:
            mask = torch.arange(T, device=device).expand(B, T) < token_length.unsqueeze(1)
            loss = loss * mask.unsqueeze(-1)
            num_elements = mask.sum() * C
            return {"total": loss.sum() / (num_elements + 1e-6), "mse": loss.sum() / (num_elements + 1e-6)}
        return {"total": loss.mean(), "mse": loss.mean()}

    @torch.no_grad()
    def generate(self, x: Dict[str, Any], num_denoise_steps=None):
        device = next(self.parameters()).device
        feature_length = x.get("feature_length", None)  # 时间长度语义
        text = x.get("text", [""] * (len(feature_length) if feature_length is not None else 1))
        if feature_length is None:
            B = len(text)
            N = 50.0  # 默认时间长度
        else:
            B = len(feature_length)
            N = max(feature_length).item()

        spatial_dim = self.spatial_dim  # 使用初始化时的默认配置
        C = self.input_dim
        steps = num_denoise_steps if num_denoise_steps is not None else self.noise_steps

        # 1. Encode Text
        text_embeddings = self.encode_text_with_cache(text, device)
        text_embeddings = [t.to(self.param_dtype) for t in text_embeddings]

        # 2. CFG Setup
        do_cfg = self.cfg_scale > 1.0
        if do_cfg:
            null_emb = self.encode_text_with_cache([""], device)[0].to(self.param_dtype)
            combined_text = text_embeddings + [null_emb] * B
        else:
            combined_text = text_embeddings

        # 3. Sampling：总 token 数 = 时间步 × 部位数
        total_len = int(N) * spatial_dim
        x_t = torch.randn(B, total_len, C, device=device)

        t_physical = torch.linspace(0, 1, steps + 1, device=device)
        t_physical = t_physical.unsqueeze(0).expand(B, -1)

        # token 级别长度
        seq_lens_time = feature_length if feature_length is not None else torch.full((B,), int(N), device=device)
        seq_lens = seq_lens_time * spatial_dim

        t_shifted, jacobian = self.get_dynamic_shift_with_jacobian(t_physical, seq_lens)

        if do_cfg:
            seq_lens = torch.cat([seq_lens, seq_lens])
            jacobian = torch.cat([jacobian, jacobian], dim=0)

        for i in range(steps):
            t_curr_physical = t_physical[:, i]
            t_next_physical = t_physical[:, i + 1]
            t_curr_shifted = t_shifted[:, i]
            t_next_shifted = t_shifted[:, i + 1]

            dt_shifted = (t_next_shifted - t_curr_shifted).view(-1, 1, 1)
            jacobian_curr = jacobian[:, i].view(-1, 1, 1)

            if do_cfg:
                x_in = torch.cat([x_t, x_t], dim=0)
                t_in = torch.cat([t_curr_shifted, t_curr_shifted], dim=0)
            else:
                x_in = x_t
                t_in = t_curr_shifted

            pred = self.model(x_in, t_in * self.time_scale, combined_text, seq_lens=seq_lens)

            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
                jacobian_curr = jacobian[:, i][:B]

            if self.prediction_type == "vel":
                v_shifted = pred
            elif self.prediction_type == "x0":
                t_curr_phys_expand = t_curr_physical.view(-1, 1, 1)
                jacobian_expand = jacobian_curr.view(-1, 1, 1)
                v_shifted = (pred - x_t) / (1 - t_curr_phys_expand + 1e-6) / jacobian_expand
            elif self.prediction_type == "noise":
                t_curr_phys_expand = t_curr_physical.view(-1, 1, 1)
                jacobian_expand = jacobian_curr.view(-1, 1, 1)
                v_shifted = (x_t - pred) / (t_curr_phys_expand + 1e-6) / jacobian_expand
            else:
                v_shifted = pred

            x_t = x_t + v_shifted * dt_shifted

        return {"generated": x_t, "text": text}

    @torch.no_grad()
    def flow_edit(self, x: Dict[str, Any], target_text: List[str], num_steps=None,
                  cfg_scale_tgt=5.0):
        device = next(self.parameters()).device
        feature = x["feature"]
        feature_length = x.get("feature_length", None)  # 时间长度语义

        # 局部推断 spatial_dim，不污染实例状态
        if feature.dim() == 4:
            B, T, P, C = feature.shape
            feature = feature.reshape(B, T * P, C)
            spatial_dim = P
        elif feature.dim() == 3:
            B, T, C = feature.shape
            spatial_dim = 1
        else:
            raise ValueError(f"Unexpected feature dim: {feature.dim()}, shape: {feature.shape}")

        text_source = x.get("text", [""] * B)
        B, T_total, C = feature.shape
        steps = num_steps if num_steps is not None else self.noise_steps

        # 1. Prepare Embeddings
        tgt_emb = self.encode_text_with_cache(target_text, device)
        tgt_emb = [t.to(self.param_dtype) for t in tgt_emb]
        src_emb = self.encode_text_with_cache(text_source, device)
        src_emb = [t.to(self.param_dtype) for t in src_emb]
        null_emb = self.encode_text_with_cache([""], device)[0].to(self.param_dtype)

        # 2. Shared Noise
        epsilon = torch.randn_like(feature)

        # 3. Initialize Invariant Path
        x_current = feature.clone()

        # 4. Time Schedule
        t_physical = torch.linspace(0, 1, steps + 1, device=device)
        t_physical = t_physical.unsqueeze(0).expand(B, -1)

        # token 级别长度
        seq_lens_time = feature_length if feature_length is not None else torch.full((B,), T_total, device=device)
        seq_lens = seq_lens_time * spatial_dim

        t_shifted_seq, jacobian_seq = self.get_dynamic_shift_with_jacobian(t_physical, seq_lens)

        combined_lens = torch.cat([seq_lens] * 3)

        for i in range(steps):
            t_curr_phys = t_physical[:, i].view(B, 1, 1)
            t_curr_shifted = t_shifted_seq[:, i]
            t_next_shifted = t_shifted_seq[:, i + 1]
            dt_shifted = (t_next_shifted - t_curr_shifted).view(-1, 1, 1)
            jacobian_curr = jacobian_seq[:, i].view(B, 1, 1)

            z_src_physical = t_curr_phys * feature + (1 - t_curr_phys) * epsilon
            coupling_term = z_src_physical - feature
            x_in_tgt = x_current + coupling_term
            x_in_src = z_src_physical

            x_batch = torch.cat([x_in_tgt, x_in_tgt, x_in_src], dim=0)
            t_batch = torch.cat([t_curr_shifted] * 3, dim=0)
            txt_batch = tgt_emb + [null_emb] * B + src_emb

            pred_batch = self.model(x_batch, t_batch * self.time_scale, txt_batch, seq_lens=combined_lens)
            pred_tgt, pred_uncond, pred_src = pred_batch.chunk(3)

            def get_v_shifted(pred, x_in):
                if self.prediction_type == "vel":
                    return pred
                elif self.prediction_type == "x0":
                    return (pred - x_in) / (1 - t_curr_phys + 1e-6) / jacobian_curr
                elif self.prediction_type == "noise":
                    return (x_in - pred) / (t_curr_phys + 1e-6) / jacobian_curr
                return pred

            v_tgt_shifted = get_v_shifted(pred_tgt, x_in_tgt)
            v_uncond_shifted = get_v_shifted(pred_uncond, x_in_tgt)
            v_src_shifted = get_v_shifted(pred_src, x_in_src)

            v_tgt_final = v_uncond_shifted + cfg_scale_tgt * (v_tgt_shifted - v_uncond_shifted)
            delta_v = v_tgt_final - v_src_shifted
            x_current = x_current + delta_v * dt_shifted

        return {"generated": x_current, "source": feature}

    @torch.no_grad()
    def stream_generate(self, x: Dict[str, Any], num_denoise_steps=None):
        text_list = x["text"]
        spatial_dim = self.spatial_dim  # 使用初始化默认值
        chunk_size = self.chunk_size * spatial_dim
        window_size = chunk_size * 2
        steps = num_denoise_steps if num_denoise_steps is not None else self.noise_steps
        stream_interval = max(1, steps // 2)
        cfg_scale = self.cfg_scale
        device = next(self.parameters()).device
        C = self.input_dim

        text_emb_cache = [self.encode_text_with_cache([t], device)[0].to(self.param_dtype) for t in text_list]
        null_emb = self.encode_text_with_cache([""], device)[0].to(self.param_dtype)

        active_streams = []
        chunks_finished = 0
        chunks_started = 0
        global_step = 0
        dt = 1.0 / steps
        while chunks_finished < len(text_list):
            can_start = (chunks_started < len(text_list)) and \
                            (chunks_started == 0 or (global_step % stream_interval == 0))
            if can_start:
                init_noise = torch.randn(window_size, C, device=device)
                active_streams.append({
                        'chunk_idx': chunks_started,
                        'step': 0,
                        'latent': init_noise,
                        'text_emb': text_emb_cache[chunks_started]
                })
                chunks_started += 1
            if not active_streams:
                break

            batch_latents = []
            batch_times = []
            batch_texts = []
            for stream in active_streams:
                batch_latents.append(stream['latent'])
                batch_times.append(stream['step'] * dt)
                batch_texts.append(stream['text_emb'])
            x_in = torch.stack(batch_latents)
            t_in = torch.tensor(batch_times, device=device)

            if cfg_scale > 1.0:
                x_in = torch.cat([x_in, x_in], dim=0)
                t_in = torch.cat([t_in, t_in], dim=0)
                txt_in = batch_texts + [null_emb] * len(batch_texts)
            else:
                txt_in = batch_texts

            v_pred_out = self.model(x_in, t_in, txt_in)

            if cfg_scale > 1.0:
                v_cond, v_uncond = v_pred_out.chunk(2)
                v_pred_out = v_uncond + cfg_scale * (v_cond - v_uncond)

            for i, stream in enumerate(active_streams):
                v = v_pred_out[i]
                if self.prediction_type == "x0":
                    v = (v - stream['latent']) / (1 - stream['step'] * dt + 1e-6)
                stream['latent'] = stream['latent'] + v * dt
                stream['step'] += 1

            for i in range(len(active_streams) - 1, -1, -1):
                stream = active_streams[i]
                if stream['step'] >= steps:
                    output_chunk = stream['latent'][:chunk_size]
                    yield {"generated": [output_chunk], "chunk_idx": stream['chunk_idx']}
                    chunks_finished += 1
                    active_streams.pop(i)
            global_step += 1

def run_tests():
    print("==================================================")
    print("Testing 1D OmniMoEditDiT (B, T, D)")
    print("==================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "input_dim": 16,
        "hidden_dim": 32,
        "ffn_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "text_dim": 64,
        "chunk_size": 16,
        "spatial_dim": 2,  # 测试用
    }
    model = OmniMoEditDiT(**config).to(device)
    model.text_encoder = None
    model.encode_text_with_cache = lambda txts, dev: [torch.randn(10, 64, device=dev) for _ in txts]

    B, T, D = 2, 4, 16
    x = {
        "feature": torch.randn(B, T, D, device=device),
        "text": ["A", "B"],
        "feature_length": torch.tensor([2, 3], device=device)
    }

    print("Testing Forward...")
    out = model(x)
    print(f"Loss: {out['total'].item()}")

    print("Testing Generate...")
    gen = model.generate(x, num_denoise_steps=5)
    print(f"Generated Shape: {gen['generated'].shape}")

    print("Testing Stream...")
    stream_input = {"text": ["Chunk1", "Chunk2", "Chunk3"]}
    for res in model.stream_generate(stream_input, num_denoise_steps=4):
        print(f"Stream Chunk {res['chunk_idx']} Shape: {res['generated'][0].shape}")

if __name__ == "__main__":
    run_tests()
