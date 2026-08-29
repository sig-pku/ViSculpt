# SAM 3 Inference Service

This directory contains the local text-prompt segmentation service used by
ViSculpt. It is an inference-focused derivative of Meta's official
[facebookresearch/sam3](https://github.com/facebookresearch/sam3) repository.
The service retains the original SAM 3 image-and-text segmentation model and
adds the local Gradio runtime required by ViSculpt, with CUDA, Apple MPS, and
CPU execution support.

Model weights are not included in this repository. This project is an
unofficial derivative and is not affiliated with or endorsed by Meta.

## Source and License

The upstream SAM 3 implementation is Copyright Meta Platforms, Inc. and
affiliates. The included SAM 3 code, model weights, and derivative portions are
governed by the [SAM License](LICENSE). Redistribution and use must comply with
that license, including its attribution and use restrictions.
