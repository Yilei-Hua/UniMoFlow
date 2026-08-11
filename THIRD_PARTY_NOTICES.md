
# Third-party notices

UniMoFlow is built on the official [SnapMoGen](https://github.com/snap-research/SnapMoGen) codebase. The SnapMoGen dataset, preprocessing tools, evaluation components, and any inherited source files remain subject to their original terms and notices. The base dataset is distributed separately through the [official SnapMoGen release](https://huggingface.co/datasets/Ericguo5513/SnapMoGen).

The 1D causal VAE and latent DiT implementation adapt architectural and source-code components from the open-source [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2) video-model repositories. These components are modified for temporal motion sequences; this repository does not require or redistribute Wan video-generation checkpoints. Original Alibaba Wan Team notices are retained in the corresponding source files where applicable.

External model dependencies include [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [uMT5-XXL](https://huggingface.co/google/umt5-xxl), and [T5 v1.1 Base](https://huggingface.co/google/t5-v1_1-base). Their weights are not part of this repository and must be obtained from their official releases.

This archive does not grant rights to external model weights or datasets. Review and comply with the licenses and usage terms of PyTorch, Hugging Face Transformers, Diffusers, Qwen, uMT5/T5, and any upstream implementation before redistribution or commercial use.
