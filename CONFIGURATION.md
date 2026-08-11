
# Configuration checklist

## UniMoFlow (`configs/unimoflow.yaml`)

| Section | Field | Purpose |
|---|---|---|
| `exp` | `root_ckpt_dir`, `root_log_dir`, `device` | Checkpoint/log roots and training device |
| root | `vae_config`, `vae_checkpoint` | VAE architecture config and project checkpoint under `checkpoints/vae/` |
| `data` | `root_dir` | SnapMoGen dataset root |
| `data` | `edit_*_files` | Train/validation/test edit-pair JSON files |
| `data` | `gen_latent_dir` | Text-to-motion latent directory under the data root |
| `model` | `checkpoint_path`, `tokenizer_path` | uMT5-XXL encoder weights and tokenizer |
| `model` | dimensions/layers/heads | Must remain aligned with the released checkpoint |
| `model` | `noise_steps`, `cfg_scale`, `fusion_schedule` | Flow-matching sampling behavior |
| `training` | `mixing_strategy` | `all_data_joint` consumes edit and generation batches in each optimizer step |
| `training` | loss weights | Relative weighting of generation and edit losses |
| `evaluator` | `config_path` | Evaluation model configuration |

The released UniMoFlow checkpoint expects latent dimension 32, hidden dimension 1024, FFN dimension 2048, 9 layers, 8 heads, and uMT5 text dimension 4096. Do not change those fields when loading that checkpoint.

## Omni-MoEdit (`configs/omni_moedit_filter.yaml`)

| Section | Field | Purpose |
|---|---|---|
| `exp` | `diff_name` | Experiment name retained for logs and backward compatibility |
| root | `vae_config`, `vae_checkpoint` | Shared VAE config and project checkpoint under `checkpoints/vae/` |
| `diffusion` | `model_checkpoint` | Base DiT project checkpoint under `checkpoints/base_dit/` |
| `diffusion` | `text_encoder_path`, `tokenizer_path` | External uMT5-XXL assets under `pretrained/` |
| `flowedit` | steps and CFG | Motion synthesis/edit sampling controls |
| `filtering` | matching/R-precision/preservation thresholds | First-stage acceptance criteria |
| `reedit_filter` | thresholds | Regeneration acceptance criteria |
| `data` | `root_dir`, latent/meta directories | Source data and normalization files |
| `evaluator` | `config_path` | Text-motion evaluator used for filtering |
| `io` | input/output paths and batch size | Pipeline inputs and generated outputs |

## Required data layout

```text
data/SnapMoGen/
  all_caption_clean.json
  data_split_info/
  latents_hrvae_detail/{train,val,test}/
  meta_data/{mean.npy,std.npy}
  renamed_bvhs/m_ep2_00086.bvh
  renamed_feats/
```

Edit-pair JSON records should provide source/target captions, an edit command, split, and source/edited latent paths. See `examples/edit_pair.json`.
