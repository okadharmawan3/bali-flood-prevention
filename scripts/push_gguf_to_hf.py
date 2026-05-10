"""Push a fine-tuned Bali flood GGUF pair to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_CARD_TEMPLATE = """\
---
base_model: LiquidAI/LFM2.5-VL-450M
language:
- en
tags:
- gguf
- vlm
- satellite
- sentinel-2
- bali
- flood-risk
---

# LFM2.5-VL-450M Bali flood-risk GGUF

Fine-tuned from [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
on Sentinel-2 RGB and SWIR image pairs for Bali flood-prevention risk labeling.

The model predicts a strict JSON object with the Bali flood schema:

```json
{{
  "flood_risk_level": "low | medium | high",
  "water_extent_level": "small | moderate | large",
  "standing_water_present": true,
  "temporary_inundation_likely": false,
  "urban_or_infrastructure_exposure": true,
  "road_or_transport_disruption_likely": false,
  "cropland_or_settlement_exposure": true,
  "river_or_coastal_overflow_context": true,
  "low_lying_or_poor_drainage_area": true,
  "vegetation_or_soil_saturation": false,
  "permanent_water_body_present": true,
  "cloud_shadow_or_image_quality_limited": false,
  "confidence": "low | medium | high"
}}
```

## Files

| file | description |
|---|---|
| `{backbone_name}` | Quantized language-model backbone |
| `{mmproj_name}` | F16 vision tower and multimodal projector |

## llama-server

```bash
llama-server -m {backbone_name} --mmproj {mmproj_name} --jinja --port 8082
```
"""


def make_model_card(backbone_name: str, mmproj_name: str) -> str:
    return MODEL_CARD_TEMPLATE.format(
        backbone_name=backbone_name,
        mmproj_name=mmproj_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a fine-tuned Bali flood GGUF model pair to Hugging Face."
    )
    parser.add_argument("--backbone", required=True, help="Path to backbone GGUF.")
    parser.add_argument("--mmproj", required=True, help="Path to mmproj GGUF.")
    parser.add_argument("--repo", required=True, help="Hugging Face model repo id.")
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbone = Path(args.backbone)
    mmproj = Path(args.mmproj)
    for path in (backbone, mmproj):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Creating repo: {args.repo} ...")
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading backbone: {backbone.name} ...")
    api.upload_file(
        path_or_fileobj=str(backbone),
        path_in_repo=backbone.name,
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"Uploading mmproj: {mmproj.name} ...")
    api.upload_file(
        path_or_fileobj=str(mmproj),
        path_in_repo=mmproj.name,
        repo_id=args.repo,
        repo_type="model",
    )

    print("Uploading model card ...")
    api.upload_file(
        path_or_fileobj=make_model_card(backbone.name, mmproj.name).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
