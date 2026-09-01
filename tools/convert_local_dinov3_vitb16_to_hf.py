#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import (
    DINOv3ViTConfig,
    DINOv3ViTImageProcessorFast,
    DINOv3ViTModel,
)


# This mapping follows the official Hugging Face DINOv3 conversion script.
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"cls_token":                   r"embeddings.cls_token",
    r"mask_token":                  r"embeddings.mask_token",
    r"storage_tokens":              r"embeddings.register_tokens",
    r"patch_embed.proj":            r"embeddings.patch_embeddings",
    r"periods":                     r"inv_freq",
    r"rope_embed":                  r"rope_embeddings",
    r"blocks.(\d+).attn.proj":      r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.":          r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+).gamma":  r"layer.\1.layer_scale\2.lambda1",
    r"blocks.(\d+).mlp.fc1":        r"layer.\1.mlp.up_proj",
    r"blocks.(\d+).mlp.fc2":        r"layer.\1.mlp.down_proj",
    r"blocks.(\d+).mlp":            r"layer.\1.mlp",
    r"blocks.(\d+).norm":           r"layer.\1.norm",
    r"w1":                          r"gate_proj",
    r"w2":                          r"up_proj",
    r"w3":                          r"down_proj",
}


def rename_key(old_key: str) -> str:
    new_key = old_key
    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        new_key = re.sub(pattern, replacement, new_key)
    return new_key


def split_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split combined QKV weights and biases into Q, K, and V."""
    qkv_keys = [key for key in list(state_dict.keys()) if "qkv" in key]

    for key in qkv_keys:
        qkv = state_dict.pop(key)

        if qkv.shape[0] % 3 != 0:
            raise ValueError(
                f"Cannot split QKV tensor {key} with shape {tuple(qkv.shape)}"
            )

        q, k, v = torch.chunk(qkv, 3, dim=0)

        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v

    return state_dict


def get_vitb16_config() -> DINOv3ViTConfig:
    """Official DINOv3 ViT-B/16 LVD-1689M architecture."""
    return DINOv3ViTConfig(
        patch_size=16,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        proj_bias=True,
        num_register_tokens=4,
        use_gated_mlp=False,
        hidden_act="gelu",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an original Meta DINOv3 ViT-B/16 checkpoint "
            "to local Hugging Face Transformers format."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Original Meta DINOv3 .pth checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output Hugging Face model directory.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Loading original checkpoint")
    print("=" * 72)
    print("Checkpoint:", checkpoint_path)
    print("Output dir:", output_dir)

    original_state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )

    if not isinstance(original_state_dict, dict):
        raise TypeError(
            "Expected the checkpoint to contain a state dict, "
            f"but received {type(original_state_dict)}"
        )

    print("Original entries:", len(original_state_dict))

    original_state_dict = split_qkv(original_state_dict)

    converted_state_dict: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for old_key, weight_tensor in original_state_dict.items():
        new_key = rename_key(old_key)

        # These filters exactly follow the official conversion behavior.
        if (
            "bias_mask" in old_key
            or "attn.k_proj.bias" in old_key
            or "local_cls_norm" in old_key
        ):
            skipped_keys.append(old_key)
            continue

        if old_key.startswith("projectors."):
            skipped_keys.append(old_key)
            continue

        if "embeddings.mask_token" in new_key:
            weight_tensor = weight_tensor.unsqueeze(1)

        # RoPE frequencies are reconstructed from the Transformers config.
        if "inv_freq" in new_key:
            skipped_keys.append(old_key)
            continue

        # transformers==5.2.0 DINOv3ViTModel stores transformer
        # blocks directly under "layer.*", not "model.layer.*".

        if new_key in converted_state_dict:
            raise KeyError(f"Duplicate converted key: {new_key}")

        converted_state_dict[new_key] = weight_tensor

    print("Converted entries:", len(converted_state_dict))
    print("Skipped entries:", len(skipped_keys))

    print("\nFirst 20 converted keys:")
    for key in list(converted_state_dict.keys())[:20]:
        print(f"  {key:70s} {tuple(converted_state_dict[key].shape)}")

    print("\n" + "=" * 72)
    print("Strictly loading converted weights")
    print("=" * 72)

    config = get_vitb16_config()
    model = DINOv3ViTModel(config).eval()

    # strict=True is essential. Do not silently ignore missing weights.
    incompatible = model.load_state_dict(
        converted_state_dict,
        strict=True,
    )

    print("Missing keys:", incompatible.missing_keys)
    print("Unexpected keys:", incompatible.unexpected_keys)
    print("Strict state-dict loading: OK")

    image_processor = DINOv3ViTImageProcessorFast(
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=2,
    )

    print("\n" + "=" * 72)
    print("Saving Hugging Face model")
    print("=" * 72)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    image_processor.save_pretrained(output_dir)

    print("Saved files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name:55s} {path.stat().st_size / 1024**2:.2f} MB")

    print("\nConversion completed successfully.")


if __name__ == "__main__":
    main()
