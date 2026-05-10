"""Delete generated annotation.json files from a dataset run.

Use this when labels need to be regenerated with a different strategy.

Usage:
    uv run scripts/delete_annotations.py --run-dir data/20260504_150038
    uv run scripts/delete_annotations.py --run-dir data/20260504_150038 --yes
"""

import argparse
from collections import Counter
from pathlib import Path


def iter_annotation_paths(run_dir: Path):
    for split in ("train", "test"):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        yield from sorted(split_dir.glob("*/*/annotation.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete annotation.json files under one Bali flood dataset run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Path to data/{run_id}.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this, only prints what would be deleted.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    annotation_paths = list(iter_annotation_paths(run_dir))
    split_counts = Counter(path.relative_to(run_dir).parts[0] for path in annotation_paths)
    action = "Deleting" if args.yes else "Would delete"
    print(f"{action} {len(annotation_paths)} annotation.json files under {run_dir}")
    for split in ("train", "test"):
        print(f"  {split}: {split_counts.get(split, 0)}")
    print()

    for path in annotation_paths:
        print(path)
        if args.yes:
            path.unlink()

    if not args.yes:
        print(f"\nDry run only. Total that would be deleted: {len(annotation_paths)}")
        print("Re-run with --yes to actually delete these files.")
    else:
        print(f"\nDeleted {len(annotation_paths)} annotation.json files.")


if __name__ == "__main__":
    main()
