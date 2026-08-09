# Pretrained Models and Data Assets

Weights and datasets are deliberately excluded from Git. They will be released through external storage links and referenced from this file.

## Planned Release Links

| Asset | Status | Link |
|---|---|---|
| UniMoFlow checkpoint | TODO | coming soon |
| HRVAE / causal VAE checkpoint | TODO | coming soon |
| Text-motion evaluator checkpoint | TODO | coming soon |
| Omni-MoEdit synthetic edit data | TODO | coming soon |
| Base text-to-motion DiT for data synthesis | TODO | coming soon |
| uMT5 text encoder assets | user-provided / external | see below |
| Qwen3-7B for edit-text generation | user-provided / external | see below |

## Expected Local Layout

Create or edit the following paths before running training, inference, or data synthesis:

| Component | Expected path | Approx. size | Notes |
|---|---|---:|---|
| UniMoFlow checkpoint | `pretrained/unimoflow/net_best_fid.tar` | 546 MiB | Project-trained checkpoint |
| HRVAE / causal VAE checkpoint | `pretrained/hrvae_detail/net_best_mpjpe.tar` | 163 MiB | Encodes motions into 32-D latent tokens |
| uMT5-XXL BF16 encoder | `pretrained/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth` | 10.6 GiB | Text encoder artifact used by UniMoFlow |
| uMT5-XXL tokenizer | `pretrained/umt5-xxl/tokenizer/` | varies | `google/umt5-xxl` tokenizer files |
| Text-motion evaluator | `pretrained/evaluator/net_best_top1.tar` | 37.6 MiB | Evaluation/filtering checkpoint |
| Evaluator text encoder | `pretrained/t5-v1_1-base/` | varies | `google/t5-v1_1-base` |
| Base motion DiT | `checkpoints/snapmogen/diff/Omni-MoEdit-DiT/model/net_best_fid.tar` | 1.41 GiB | Used by Omni-MoEdit synthesis |
| Qwen3-7B | `pretrained/Qwen3-7B/` | varies | Used to generate edit commands and target captions |

## Example External Downloads

```bash
huggingface-cli download Qwen/Qwen3-7B --local-dir pretrained/Qwen3-7B
huggingface-cli download google/t5-v1_1-base --local-dir pretrained/t5-v1_1-base
huggingface-cli download google/umt5-xxl \
  --include "tokenizer*" "spiece.model" "special_tokens_map.json" \
  --local-dir pretrained/umt5-xxl/tokenizer
```

The UniMoFlow checkpoint, HRVAE checkpoint, evaluator checkpoint, base motion DiT, and Omni-MoEdit synthetic data are project artifacts. They are currently listed as TODO items and will be linked after they are uploaded to external storage.
