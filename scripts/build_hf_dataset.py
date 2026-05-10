"""Build a local Hugging Face-style dataset folder from labeled samples."""

import argparse
from pathlib import Path

from bali_flood_prevention.hf_dataset import collect_rows, split_rows, write_jsonl

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Package labeled Bali flood samples.")
    parser.add_argument("--run-dir", required=True, help="Path to data/{run_id}.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to {run_dir}/hf_dataset.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "hf_dataset"
    images_dir = output_dir / "images"
    rows = collect_rows(run_dir, images_dir=images_dir)
    if not rows:
        raise RuntimeError(f"No labeled annotation.json files found under {run_dir}")

    train_rows, test_rows = split_rows(rows)
    train_count = write_jsonl(train_rows, output_dir / "train.jsonl")
    test_count = write_jsonl(test_rows, output_dir / "test.jsonl")

    print(f"Built HF-style dataset at {output_dir}")
    print(f"  train: {train_count}")
    print(f"  test: {test_count}")
    print(f"  images: {len(rows) * 2}")


if __name__ == "__main__":
    main()
