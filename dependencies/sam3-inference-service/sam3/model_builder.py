# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Build the inference-only SAM 3 image model.

This module intentionally contains no SAM 3.1 or video-model builders.  The public
builder resolves a local ``checkpoint/sam3.pt`` before it ever contacts the Hub.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from sam3.device import resolve_device
from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.model_misc import (
    MLP,
    DotProductScoring,
    TransformerWrapper,
)
from sam3.model.model_misc import MultiheadAttentionWrapper as MultiheadAttention
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone

SAM3_HF_REPO_ID = "facebook/sam3"
SAM3_CHECKPOINT_FILENAME = "sam3.pt"
SAM3_CHECKPOINT_ENV = "SAM3_CHECKPOINT"


def _setup_tf32() -> None:
    """Enable TensorFloat-32 on supported NVIDIA GPUs."""
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _local_checkpoint_candidates(
    local_checkpoint_dir: str | Path | None,
) -> Iterator[Path]:
    if local_checkpoint_dir is not None:
        yield Path(local_checkpoint_dir).expanduser() / SAM3_CHECKPOINT_FILENAME
        return

    # The repository/package-local directory is the documented default.  The
    # current-working-directory candidate also supports an installed package.
    yield _project_root() / "checkpoint" / SAM3_CHECKPOINT_FILENAME
    cwd_candidate = Path.cwd() / "checkpoint" / SAM3_CHECKPOINT_FILENAME
    if cwd_candidate != _project_root() / "checkpoint" / SAM3_CHECKPOINT_FILENAME:
        yield cwd_candidate


def download_ckpt_from_hf(
    *,
    local_dir: str | Path | None = None,
) -> Path:
    """Download the original SAM 3 checkpoint (never SAM 3.1)."""
    return Path(
        hf_hub_download(
            repo_id=SAM3_HF_REPO_ID,
            filename=SAM3_CHECKPOINT_FILENAME,
            local_dir=local_dir,
        )
    )


def download_ckpt_from_cloud(
    *,
    local_dir: str | Path | None = None,
) -> Path:
    """Download from Hugging Face, falling back to the ModelScope mirror."""
    try:
        return download_ckpt_from_hf(local_dir=local_dir)
    except Exception as huggingface_error:
        try:
            from modelscope import snapshot_download

            snapshot_dir = Path(
                snapshot_download(
                    model_id=SAM3_HF_REPO_ID,
                    allow_file_pattern=[SAM3_CHECKPOINT_FILENAME],
                    local_dir=str(local_dir) if local_dir is not None else None,
                )
            )
            checkpoint_path = snapshot_dir / SAM3_CHECKPOINT_FILENAME
            if checkpoint_path.is_file():
                return checkpoint_path
            raise FileNotFoundError(checkpoint_path)
        except Exception as modelscope_error:
            raise RuntimeError(
                "Unable to download original SAM 3 from Hugging Face or ModelScope. "
                "Check network access and checkpoint-license authorization. "
                f"Hugging Face: {huggingface_error}; ModelScope: {modelscope_error}"
            ) from modelscope_error


def resolve_sam3_checkpoint(
    checkpoint_path: str | Path | None = None,
    *,
    local_checkpoint_dir: str | Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Resolve a checkpoint using explicit, environment, local, then cloud priority."""
    if checkpoint_path is not None:
        explicit_path = Path(checkpoint_path).expanduser()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint does not exist: {explicit_path}")
        return explicit_path.resolve()

    env_path = os.environ.get(SAM3_CHECKPOINT_ENV)
    if env_path:
        configured_path = Path(env_path).expanduser()
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"{SAM3_CHECKPOINT_ENV} points to a missing file: {configured_path}"
            )
        return configured_path.resolve()

    checked: list[Path] = []
    for candidate in _local_checkpoint_candidates(local_checkpoint_dir):
        checked.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    if allow_download:
        return download_ckpt_from_cloud()

    locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"No local SAM 3 checkpoint found ({locations}); cloud download is disabled."
    )


def _create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_fa3=False,
        use_rope_real=False,
    )


def _create_vision_backbone(compile_mode=None) -> Sam3DualViTDetNeck:
    return Sam3DualViTDetNeck(
        position_encoding=_create_position_encoding(precompute_resolution=1008),
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=_create_vit_backbone(compile_mode=compile_mode),
        add_sam2_neck=False,
    )


def _create_text_encoder(bpe_path: str | Path) -> VETextEncoder:
    return VETextEncoder(
        tokenizer=SimpleTokenizer(bpe_path=str(bpe_path)),
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_transformer_encoder() -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
            use_fa3=False,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
            use_fa3=False,
        ),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder() -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            use_fa3=False,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )
    return TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )


def _create_transformer() -> TransformerWrapper:
    return TransformerWrapper(
        encoder=_create_transformer_encoder(),
        decoder=_create_transformer_decoder(),
        d_model=256,
    )


def _create_dot_product_scoring() -> DotProductScoring:
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None) -> UniversalSegmentationHead:
    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=MultiheadAttention(
            num_heads=8,
            dropout=0,
            embed_dim=256,
            use_fa3=False,
        ),
        pixel_decoder=PixelDecoder(
            num_upsampling_stages=3,
            interpolation_mode="nearest",
            hidden_dim=256,
            compile_mode=compile_mode,
        ),
    )


def _create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )
    return SequenceGeometryEncoder(
        pos_enc=_create_position_encoding(),
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


def _load_checkpoint(model: Sam3Image, checkpoint_path: Path) -> None:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid SAM 3 checkpoint: {checkpoint_path}")

    detector_state = {
        key.removeprefix("detector."): value
        for key, value in state_dict.items()
        if key.startswith("detector.")
    }
    image_state = detector_state or state_dict
    if not image_state:
        raise ValueError(f"No image-model weights found in: {checkpoint_path}")

    missing_keys, unexpected_keys = model.load_state_dict(
        image_state,
        strict=False,
        assign=True,
    )
    if missing_keys:
        missing_preview = missing_keys[:8]
        raise RuntimeError(
            "Checkpoint is incompatible with the original SAM 3 image model. "
            f"missing={missing_preview}"
        )
    model.unexpected_checkpoint_keys = tuple(unexpected_keys)


def build_sam3_image_model(
    bpe_path: str | Path | None = None,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
    checkpoint_path: str | Path | None = None,
    load_from_HF: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    local_checkpoint_dir: str | Path | None = None,
) -> Sam3Image:
    """Build original SAM 3 for text-prompted image segmentation.

    Local checkpoint resolution order is: ``checkpoint_path``, the
    ``SAM3_CHECKPOINT`` environment variable, ``checkpoint/sam3.pt`` in the
    project/current directory, then ``facebook/sam3`` on Hugging Face.
    """
    if not eval_mode:
        raise ValueError("This slim project only supports inference (eval_mode=True).")
    if not enable_segmentation:
        raise ValueError(
            "Text-prompted segmentation requires enable_segmentation=True."
        )
    if enable_inst_interactivity:
        raise ValueError(
            "Interactive SAM 1 prompts are not included in this slim project."
        )

    if bpe_path is None:
        bpe_path = _project_root() / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    bpe_path = Path(bpe_path)
    if not bpe_path.is_file():
        raise FileNotFoundError(
            f"SAM 3 tokenizer vocabulary does not exist: {bpe_path}"
        )

    resolved_device = resolve_device(device)
    if compile and resolved_device.type != "cuda":
        raise ValueError("Model compilation is currently supported only on CUDA.")

    resolved_checkpoint = resolve_sam3_checkpoint(
        checkpoint_path,
        local_checkpoint_dir=local_checkpoint_dir,
        allow_download=load_from_HF,
    )
    compile_mode = "default" if compile else None
    model = Sam3Image(
        backbone=SAM3VLBackbone(
            visual=_create_vision_backbone(compile_mode=compile_mode),
            text=_create_text_encoder(bpe_path),
            scalp=1,
        ),
        transformer=_create_transformer(),
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head(compile_mode=compile_mode),
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=_create_dot_product_scoring(),
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=None,
    )
    _load_checkpoint(model, resolved_checkpoint)

    model = model.to(resolved_device).eval()
    model.checkpoint_path = str(resolved_checkpoint)
    return model


__all__ = [
    "SAM3_CHECKPOINT_ENV",
    "SAM3_CHECKPOINT_FILENAME",
    "SAM3_HF_REPO_ID",
    "build_sam3_image_model",
    "download_ckpt_from_cloud",
    "download_ckpt_from_hf",
    "resolve_sam3_checkpoint",
]
