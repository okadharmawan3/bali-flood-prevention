"""Convert Bali flood rows to leap-finetune VLM SFT format.

This is a preparation helper only. It writes JSONL files and image assets for
later fine-tuning, but it does not start training.

Usage:
    uv run scripts/prepare_bali_flood.py --dataset data/20260504_150038/hf_dataset
    uv run scripts/prepare_bali_flood.py --dataset USER/bali-flood-prevention --modal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from bali_flood_prevention.schema import SYSTEM_PROMPT, USER_TEXT, dumps_label, load_label_text

DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "bali-flood"
PROJECT_ROOT = Path(__file__).parent.parent

MODAL_VOLUME_NAME = "bali-flood-prevention"
MODAL_MOUNT_POINT = "/outputs"
MODAL_OUTPUT_DIR = f"{MODAL_MOUNT_POINT}/data/bali-flood"


def make_vlm_row(rgb_name: str, swir_name: str, output: str) -> dict[str, object]:
    """Build one leap-finetune VLM SFT row from image filenames and label JSON."""
    canonical_output = dumps_label(load_label_text(output))
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_name},
                    {"type": "image", "image": swir_name},
                    {"type": "text", "text": f"{SYSTEM_PROMPT.strip()}\n\n{USER_TEXT}"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": canonical_output}],
            },
        ]
    }


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in materialized)
        + ("\n" if materialized else ""),
        encoding="utf-8",
    )
    print(f"  Wrote {len(materialized)} rows to {path}")
    return len(materialized)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(row)
    return rows


def load_rows(source_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Load train/test rows from local JSONL or a downloaded HF dataset folder."""
    train_jsonl = source_dir / "train.jsonl"
    test_jsonl = source_dir / "test.jsonl"
    if train_jsonl.exists() or test_jsonl.exists():
        return {
            "train": read_jsonl(train_jsonl),
            "test": read_jsonl(test_jsonl),
        }

    from datasets import load_dataset

    dataset = load_dataset(str(source_dir))
    rows: dict[str, list[dict[str, object]]] = {}
    for split_name in ("train", "test"):
        if split_name in dataset:
            rows[split_name] = [dict(row) for row in dataset[split_name]]
    return rows


def stage_source(dataset: str, output_dir: Path) -> Path:
    """Return a local folder containing packaged rows and images."""
    source_path = Path(dataset)
    if source_path.is_dir():
        return source_path

    from huggingface_hub import snapshot_download

    print(f"Downloading snapshot of {dataset} into {output_dir} ...")
    snapshot_download(repo_id=dataset, repo_type="dataset", local_dir=str(output_dir))
    print("  Download complete.")
    return output_dir


def ensure_images(source_dir: Path, output_dir: Path) -> Path:
    source_images = source_dir / "images"
    if not source_images.is_dir():
        raise FileNotFoundError(
            f"images/ directory not found at {source_images}. "
            "Run build_hf_dataset.py or push/download the dataset with images first."
        )

    output_images = output_dir / "images"
    if source_images.resolve() != output_images.resolve():
        if output_images.exists():
            shutil.rmtree(output_images)
        print(f"Copying images to {output_images} ...")
        shutil.copytree(source_images, output_images)
    return output_images


def rows_to_vlm_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    vlm_rows: list[dict[str, object]] = []
    for row in rows:
        rgb_path = Path(str(row["rgb_path"]))
        swir_path = Path(str(row["swir_path"]))
        output = str(row["output"])
        vlm_rows.append(make_vlm_row(rgb_path.name, swir_path.name, output))
    return vlm_rows


def prepare_dataset(dataset: str, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = stage_source(dataset, output_dir)
    images_dir = ensure_images(source_dir, output_dir)
    rows_by_split = load_rows(source_dir)

    counts: dict[str, int] = {}
    for split_name in ("train", "test"):
        split_rows = rows_by_split.get(split_name, [])
        if not split_rows:
            print(f"  Split '{split_name}' not found or empty, skipping.")
            counts[split_name] = 0
            continue
        out_path = output_dir / f"bali_flood_{split_name}.jsonl"
        counts[split_name] = write_jsonl(rows_to_vlm_rows(split_rows), out_path)

    print()
    print("Done. Use these paths in leap-finetune config:")
    print(f"  train: {output_dir / 'bali_flood_train.jsonl'}")
    print(f"  test : {output_dir / 'bali_flood_test.jsonl'}")
    print(f"  images: {images_dir}")
    return counts


def run_on_modal(dataset: str) -> None:
    if Path(dataset).exists():
        raise ValueError("--modal requires --dataset to be a Hugging Face dataset repo id")

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Add it to .env or the current shell.")

    import modal

    app = modal.App("bali-flood-data-prep")
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "bali_flood_prevention"
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("datasets", "huggingface_hub")
        .add_local_file(__file__, "/app/prepare_bali_flood.py", copy=True)
        .add_local_dir(str(src_dir), "/app/bali_flood_prevention", copy=True)
    )

    @app.function(
        image=image,
        volumes={MODAL_MOUNT_POINT: volume},
        timeout=3600,
        serialized=True,
        secrets=[modal.Secret.from_local_environ(env_keys=["HF_TOKEN"])],
    )
    def prepare(remote_dataset: str, remote_output: str) -> None:
        cmd = [
            sys.executable,
            "/app/prepare_bali_flood.py",
            "--dataset",
            remote_dataset,
            "--output",
            remote_output,
        ]
        env = {**os.environ, "PYTHONPATH": "/app"}
        subprocess.run(cmd, check=True, env=env)
        volume.commit()

    print(f"Preparing Bali flood dataset on Modal volume '{MODAL_VOLUME_NAME}' ...")
    with modal.enable_output():
        with app.run():
            prepare.remote(dataset, MODAL_OUTPUT_DIR)

    print()
    print(f"Data ready in Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_OUTPUT_DIR}.")
    print("Next steps:")
    print("  LFM2.5:   uv run scripts/launch_leap_modal.py configs/bali_flood_finetune_modal.yaml")
    print("  SmolVLM2: uv run scripts/train_smolvlm_transformers_modal.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bali flood dataset rows to leap-finetune VLM SFT JSONL."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Hugging Face dataset repo id or local hf_dataset folder.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Alias for --dataset when using a local hf_dataset folder.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Directory to write leap-finetune files (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help=(
            f"Run preparation on Modal and write to volume '{MODAL_VOLUME_NAME}' "
            f"at {MODAL_OUTPUT_DIR}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or args.source_dir
    if not dataset:
        raise ValueError("Provide --dataset or --source-dir")
    if args.dataset and args.source_dir:
        raise ValueError("Use only one of --dataset or --source-dir")

    if args.modal:
        run_on_modal(str(dataset))
        return

    prepare_dataset(str(dataset), Path(args.output))


if __name__ == "__main__":
    main()
