# Omni-MoEdit Synthesis Pipeline

The text-pair stage has two complementary entry points. They share the same
Qwen3-8B dependency and produce records consumed by the same downstream motion
synthesis and filtering code.

## 1. Comprehensive synthesis

`generate_edit_triplets.py` is the direct, category-agnostic route. For each
source caption, one prompt asks Qwen3-8B for multiple heterogeneous editing
variations. A variation may combine body-part, action-type, spatial,
timing, and style changes; it is not assigned to a fixed edit category first.

```bash
python generate_edit_triplets.py \
  --input ../../data/SnapMoGen/all_caption_clean.json \
  --output ../../outputs/omni_moedit/edit_pairs.json \
  --failed_output ../../outputs/omni_moedit/failed_keys.json \
  --model ../../pretrained/Qwen3-8B \
  --gpus 0,1 \
  --num_commands 6
```

## 2. Category-controlled synthesis

`generate_multi_attribute_edits.py` generates coarse action-type,
fine-grained body-part, and style candidates in separate passes. This route is
useful for controlled coverage expansion or per-category analysis.

```bash
python generate_multi_attribute_edits.py \
  --input ../../data/SnapMoGen/all_caption_clean.json \
  --output_dir ../../outputs/omni_moedit/text_pairs \
  --model ../../pretrained/Qwen3-8B \
  --gpus 0,1 \
  --num_commands 6 \
  --types coarse,fine,style
```

The selected types produce `coarse_edit_pairs.json`, `fine_edit_pairs.json`,
and/or `style_edit_pairs.json`.

## Shared record format

Both routes group candidate variations by source-motion key. The fields used
by downstream synthesis are:

```json
[
  {
    "original_key": "source_key",
    "original_caption": "...",
    "variations": [
      {
        "edit_command": "...",
        "new_caption": "...",
        "reverse_command": "..."
      }
    ]
  }
]
```

Category-controlled records may additionally contain `locate_edit_phase` and
edit-type metadata. The comprehensive script stores the reverse field as
`reverse_edit_command`, while the category-controlled script uses
`reverse_command`; the downstream loader accepts both aliases.
`synthesize_and_filter.py` runs the
pretrained Base DiT with FlowEdit-style sampling, and filters synthesized
motions with the configured text-alignment, edit-improvement, and structure
criteria.

```bash
python synthesize_and_filter.py \
  --config ../../configs/omni_moedit_filter.yaml \
  --input_json ../../outputs/omni_moedit/edit_pairs.json \
  --output_dir ../../outputs/omni_moedit/synthesized_pairs
```

Pass a per-category JSON to `--input_json` when using the category-controlled
route. Regeneration and second-stage filtering are provided by
`regenerate_and_filter.py` and `second_stage_filter.py`.
