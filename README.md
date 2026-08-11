# UniMoFlow

[![arXiv](https://img.shields.io/badge/arXiv-2608.09143-b31b1b.svg)](https://arxiv.org/abs/2608.09143)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6f5c.svg)](https://unimoflow-motion.shayla-fralix845.chatgpt.site)

Official code release for **UniMoFlow: Grounding Instruction-Driven 3D Human Motion Editing in Generation**.

UniMoFlow studies 3D human motion generation and instruction-driven motion editing in a unified latent flow-matching framework. The model treats text-to-motion generation and source-conditioned editing as mutually supportive abilities: it learns from high-quality text-to-motion data and synthetic instruction-edit triplets, then switches between generation and editing through a shared context token sequence and task-aware attention masks. The repository also includes the **Omni-MoEdit** data synthesis pipeline used to construct instruction-driven edit supervision.

> Paper: [arXiv:2608.09143](https://arxiv.org/abs/2608.09143). Project page: [UniMoFlow](https://unimoflow-motion.shayla-fralix845.chatgpt.site).

<p align="center">
  <img src="assets/teaser.webp" alt="UniMoFlow teaser" width="95%">
</p>

## Links

- Paper: [arXiv](https://arxiv.org/abs/2608.09143) | [PDF](https://arxiv.org/pdf/2608.09143)
- Project page: [UniMoFlow](https://unimoflow-motion.shayla-fralix845.chatgpt.site)
- Checkpoints: [Baidu Netdisk](https://pan.baidu.com/s/1FT7uGIhXzmHsaVIkc8-TuQ) (access code: `wpun`)
- Omni-MoEdit synthetic data: [Baidu Netdisk](https://pan.baidu.com/s/1U7aOf-z9cjHXogDayprQyA) (access code: `uupk`)

## What Is Included

```text
UniMoFlow/
  codes/
    train_unimoflow.py          # joint generation/edit training
    run_unimoflow.py            # generation and editing inference
    evaluate_unimoflow.py       # evaluation entry point
    models_flow/                # UniMoFlow and latent VAE modules
    trainers/                   # mixed generation/edit training loop
    dataset/                    # text-motion and edit-pair datasets
    model/evaluator/            # evaluation encoders
    omni_moedit/                # Omni-MoEdit synthesis and filtering pipeline
    utils/
  configs/                      # portable configuration templates
  examples/                     # minimal JSON input examples
  assets/teaser.webp            # teaser reused from the paper/project assets
  checkpoints/                  # project-trained VAE, Base DiT, and UniMoFlow weights
    vae/net_best_mpjpe.tar
    base_dit/net_best_fid.tar
    unimoflow/net_best_fid.tar
  pretrained/                   # external pretrained model dependencies only
    umt5-xxl/
    evaluator/
    t5-v1_1-base/
    Qwen3-8B/
  CONFIGURATION.md
  PRETRAINED_MODELS.md
  THIRD_PARTY_NOTICES.md
  TODO.md
  requirements.txt
```

Large model weights, datasets, logs, and generated visualizations remain outside Git. Project checkpoints are hosted separately; see `PRETRAINED_MODELS.md` for downloads and the expected local layout, and `TODO.md` for the remaining release items.

## Installation

```bash
conda create -n unimoflow python=3.10 -y
conda activate unimoflow
pip install -r requirements.txt
```

The code expects PyTorch, CUDA-enabled dependencies, text encoder assets, evaluator checkpoints, and motion data. Exact paths can be edited in `configs/*.yaml`.

## Prepare Assets

Create the expected folders:

```bash
mkdir -p checkpoints/{vae,base_dit,unimoflow} \
  pretrained/{umt5-xxl,evaluator,t5-v1_1-base,Qwen3-8B} data outputs
```

Download the project [checkpoint bundle](https://pan.baidu.com/s/1FT7uGIhXzmHsaVIkc8-TuQ) (access code: `wpun`) and place its three subfolders directly under `checkpoints/`. Then prepare the external dependencies according to `PRETRAINED_MODELS.md`. The remaining external or pending assets are:

- Evaluator checkpoint.
- uMT5 text encoder assets.
- Omni-MoEdit synthetic edit data.

### Official upstream assets

| Dependency | Role | Official source |
|---|---|---|
| SnapMoGen | Base text-motion dataset used for generation training and Omni-MoEdit synthesis | [Project](https://snap-research.github.io/SnapMoGen/) · [Code](https://github.com/snap-research/SnapMoGen) · [Data](https://huggingface.co/datasets/Ericguo5513/SnapMoGen) |
| Qwen3-8B | Generates edit commands, target captions, reverse instructions, and auxiliary annotations | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| uMT5-XXL | Frozen text encoder used by the motion models | [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) |
| T5 v1.1 Base | Text encoder used by the text-motion evaluator | [google/t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base) |
| Wan series | Upstream video-model implementation referenced when adapting the causal VAE and DiT components to 1D motion latents | [Wan2.1](https://github.com/Wan-Video/Wan2.1) · [Wan2.2](https://github.com/Wan-Video/Wan2.2) |

The Qwen and Google text-model weights are external dependencies and are not redistributed in this repository. Wan video checkpoints are not required; the Wan links above identify the upstream open-source implementations on which parts of our 1D motion architecture are based.

## Inference

Generate a motion from text:

```bash
cd codes
python run_unimoflow.py \
  --config ../configs/unimoflow.yaml \
  --which_epoch ../checkpoints/unimoflow/net_best_fid.tar \
  --mode t2m \
  --text "A person walks forward with a crossing-step gait." \
  --output_dir ../outputs/t2m
```

Edit a source latent with the native editing mode or SAFE:

```bash
cd codes
python run_unimoflow.py \
  --config ../configs/unimoflow.yaml \
  --which_epoch ../checkpoints/unimoflow/net_best_fid.tar \
  --mode edit \
  --motion_file ../data/source_latent.npy \
  --is_latent true \
  --edit_text "Raise the right hand while continuing to walk forward." \
  --self_flowedit true \
  --output_dir ../outputs/edit
```

Use `python codes/run_unimoflow.py --help` for available generation, editing, sampling, and output options.

## Training UniMoFlow

Single GPU:

```bash
cd codes
python train_unimoflow.py --config ../configs/unimoflow.yaml
```

Distributed training:

```bash
cd codes
torchrun --nproc_per_node=2 train_unimoflow.py --config ../configs/unimoflow.yaml
```

The default mixed-training configuration uses one generation batch and one editing batch at each optimizer step. Generation and editing losses are computed jointly and combined according to the weights in `configs/unimoflow.yaml`.

## Evaluation

```bash
cd codes
python evaluate_unimoflow.py \
  --config ../configs/unimoflow.yaml \
  --which_epoch ../checkpoints/unimoflow/net_best_fid.tar \
  --output_dir ../outputs/evaluation
```

The evaluator reports text-to-motion metrics and editing metrics used in the paper. Dataset paths and evaluator checkpoints are configured in `configs/unimoflow.yaml`.

## Omni-MoEdit Data Synthesis

The `codes/omni_moedit` folder contains the pipeline for constructing Omni-MoEdit. The LLM stage provides two complementary generation routes:

- **Comprehensive synthesis (direct route):** `generate_edit_triplets.py` directly samples heterogeneous, potentially compositional edits without assigning each sample to a fixed category. Its prompt jointly covers body parts, action types, spatial changes, timing, and styles.
- **Category-controlled synthesis:** `generate_multi_attribute_edits.py` separately expands coarse action-type, fine-grained body-part, and style edits. Use this route when controlled per-category coverage is needed.

Both routes emit source-caption records with multiple edit variations and feed the same motion-synthesis and filtering stages:

1. Generate edit commands, target captions, and auxiliary metadata from source captions using a pretrained LLM.
2. Encode source motions into latent tokens with the causal VAE.
3. Synthesize target motions with a pretrained text-to-motion DiT using FlowEdit-style sampling.
4. Filter candidates by target-text alignment, edit improvement over the source, and motion-structure constraints.
5. Optionally regenerate weak records and run second-stage filtering.

### Comprehensive synthesis

This is the direct, category-agnostic route used to generate diverse and compositional edit candidates:

```bash
cd codes

python omni_moedit/generate_edit_triplets.py \
  --input ../data/SnapMoGen/all_caption_clean.json \
  --output ../outputs/omni_moedit/edit_pairs.json \
  --failed_output ../outputs/omni_moedit/failed_keys.json \
  --model ../pretrained/Qwen3-8B \
  --gpus 0,1 \
  --num_commands 6
```

### Category-controlled synthesis

The specialized route generates coarse, fine, and style candidates independently:

```bash
cd codes

python omni_moedit/generate_multi_attribute_edits.py \
  --input ../data/SnapMoGen/all_caption_clean.json \
  --output_dir ../outputs/omni_moedit/text_pairs \
  --model ../pretrained/Qwen3-8B \
  --gpus 0,1 \
  --types coarse,fine,style
```

It writes `coarse_edit_pairs.json`, `fine_edit_pairs.json`, and `style_edit_pairs.json` for the selected types. Each generated JSON can be processed independently by the shared synthesis stage:

```bash
cd codes

python omni_moedit/synthesize_and_filter.py \
  --config ../configs/omni_moedit_filter.yaml \
  --input_json ../outputs/omni_moedit/edit_pairs.json \
  --output_dir ../outputs/omni_moedit/synthesized_pairs
```

For the category-controlled route, replace `--input_json` with the desired per-category file. See [`codes/omni_moedit/README.md`](codes/omni_moedit/README.md) for the record schema and the relationship between both generation routes.

## Citation

If you find this work useful, please cite:

```bibtex
@article{hua2026unimoflow,
  title   = {UniMoFlow: Grounding Instruction-Driven 3D Human Motion Editing in Generation},
  author  = {Hua, Yilei and Jing, Beibei and Zheng, Ce and Zhou, Hanyu and Luo, Yawei and Yang, Wei},
  journal = {arXiv preprint arXiv:2608.09143},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.09143}
}
```

## License

Final code, checkpoint, and dataset license terms will be added before the complete public release. Third-party models and datasets remain subject to their respective upstream licenses.

## Acknowledgements

This codebase is built on the official [SnapMoGen](https://github.com/snap-research/SnapMoGen) codebase and adapts open-source causal-VAE and DiT design/code components from the [Wan video-model series](https://github.com/Wan-Video/Wan2.2) to 1D human-motion latents. We thank the authors of these projects and the developers of Qwen, uMT5/T5, PyTorch, and Hugging Face Transformers. See `THIRD_PARTY_NOTICES.md` for source links and licensing notes.
