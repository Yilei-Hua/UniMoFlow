# Checkpoints, Pretrained Models, and Data Assets

Weights and datasets are deliberately excluded from Git. They will be released through external storage links and referenced from this file.

## Release Links

| Asset | Status | Link |
|---|---|---|
| UniMoFlow checkpoint | released | [checkpoint bundle](https://pan.baidu.com/s/1FT7uGIhXzmHsaVIkc8-TuQ) (code: `wpun`) |
| HRVAE / causal VAE checkpoint | released | [checkpoint bundle](https://pan.baidu.com/s/1FT7uGIhXzmHsaVIkc8-TuQ) (code: `wpun`) |
| Text-motion evaluator checkpoint | TODO | coming soon |
| Omni-MoEdit synthetic edit data | TODO | coming soon |
| Base text-to-motion DiT for data synthesis | released | [checkpoint bundle](https://pan.baidu.com/s/1FT7uGIhXzmHsaVIkc8-TuQ) (code: `wpun`) |
| SnapMoGen base dataset | external | [official data](https://huggingface.co/datasets/Ericguo5513/SnapMoGen) |
| uMT5 text encoder assets | external | [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) |
| Qwen3-8B for edit-text generation | external | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Evaluator T5 tokenizer/encoder | external | [google/t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base) |

## Expected Local Layout

All models trained specifically for this project live under `checkpoints/`.
The `pretrained/` directory is reserved for external model dependencies.

Create or edit the following paths before running training, inference, or data synthesis:

| Component | Expected path | Approx. size | Notes |
|---|---|---:|---|
| UniMoFlow checkpoint | `checkpoints/unimoflow/net_best_fid.tar` | 546 MiB | Project-trained checkpoint |
| HRVAE / causal VAE checkpoint | `checkpoints/vae/net_best_mpjpe.tar` | 163 MiB | Encodes motions into 32-D latent tokens |
| Base motion DiT | `checkpoints/base_dit/net_best_fid.tar` | 1.41 GiB | Project-trained model used by Omni-MoEdit synthesis |
| uMT5-XXL BF16 encoder | `pretrained/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth` | 10.6 GiB | Text encoder artifact used by UniMoFlow |
| uMT5-XXL tokenizer | `pretrained/umt5-xxl/tokenizer/` | varies | `google/umt5-xxl` tokenizer files |
| Text-motion evaluator | `pretrained/evaluator/net_best_top1.tar` | 37.6 MiB | Evaluation/filtering checkpoint |
| Evaluator text encoder | `pretrained/t5-v1_1-base/` | varies | `google/t5-v1_1-base` |
| Qwen3-8B | `pretrained/Qwen3-8B/` | varies | Used to generate edit commands, target captions, reverse instructions, and auxiliary annotations |

Download the complete project-trained checkpoint folder from [Baidu Netdisk](https://pan.baidu.com/s/1FT7uGIhXzmHsaVIkc8-TuQ) using access code `wpun`. After extraction, the repository should contain:

```text
checkpoints/
  unimoflow/net_best_fid.tar
  vae/net_best_mpjpe.tar
  base_dit/net_best_fid.tar
```

## Example External Downloads

```bash
pip install -U "huggingface_hub[cli]"

hf download Qwen/Qwen3-8B --local-dir pretrained/Qwen3-8B
hf download google/t5-v1_1-base --local-dir pretrained/t5-v1_1-base
hf download google/umt5-xxl \
  --include "tokenizer*" "spiece.model" "special_tokens_map.json" \
  --local-dir pretrained/umt5-xxl/tokenizer
hf download Ericguo5513/SnapMoGen \
  --repo-type dataset \
  --local-dir data/SnapMoGen
```

The UniMoFlow checkpoint, VAE checkpoint, and Base DiT are project-trained weights distributed in the checkpoint bundle above and belong under `checkpoints/`. The evaluator and language encoders are dependencies under `pretrained/`. Omni-MoEdit is a project dataset under `data/`; its download link will be added separately.

## SnapMoGen Base Data

Omni-MoEdit is synthesized from the SnapMoGen split structure, captions, and motions. Download SnapMoGen from its [official Hugging Face dataset](https://huggingface.co/datasets/Ericguo5513/SnapMoGen) and follow the preprocessing instructions in the [official SnapMoGen repository](https://github.com/snap-research/SnapMoGen). The project page is available at <https://snap-research.github.io/SnapMoGen/>.

The released Omni-MoEdit files do not transfer ownership of SnapMoGen. Users must obtain and use the base dataset under its own terms.
