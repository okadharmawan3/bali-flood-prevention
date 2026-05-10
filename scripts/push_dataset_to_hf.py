"""Push a labeled Bali flood run to a Hugging Face dataset repo."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from bali_flood_prevention.hf_dataset import collect_rows, dataset_dict_from_rows, split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a Bali flood dataset run to Hugging Face.")
    parser.add_argument("--run-dir", required=True, help="Path to data/{run_id}.")
    parser.add_argument("--hf-dataset", required=True, help="Repo id, e.g. user/bali-flood-prevention.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    hf_dir = run_dir / "hf_dataset"
    images_dir = hf_dir / "images"
    rows = collect_rows(run_dir, images_dir=images_dir)
    if not rows:
        raise RuntimeError(f"No labeled samples found under {run_dir}")

    train_rows, test_rows = split_rows(rows)
    api = HfApi()
    api.create_repo(repo_id=args.hf_dataset, repo_type="dataset", exist_ok=True)

    print(f"Pushing parquet rows to {args.hf_dataset} (train={len(train_rows)}, test={len(test_rows)})")
    dataset_dict_from_rows(rows).push_to_hub(args.hf_dataset)

    print(f"Uploading {len(rows) * 2} images")
    api.upload_folder(
        folder_path=str(images_dir),
        path_in_repo="images",
        repo_id=args.hf_dataset,
        repo_type="dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{args.hf_dataset}")


if __name__ == "__main__":
    main()
