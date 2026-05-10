"""Validate generated Bali flood sample runs.

Usage:
    uv run scripts/check_samples.py
    uv run scripts/check_samples.py 20260504_130000 --require-labels
"""

import argparse
import json
import sys
from pathlib import Path

from bali_flood_prevention.schema import RISK_LEVELS, load_label_text

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_METADATA = {
    "sample_id",
    "region",
    "point_id",
    "point_index",
    "spatial_index",
    "temporal_index",
    "lon",
    "lat",
    "timestamp",
    "split",
    "size_km",
}


def resolve_run_dir(run_arg: str | None) -> Path:
    if run_arg:
        candidate = Path(run_arg)
        if candidate.is_dir():
            return candidate
        run_dir = DATA_DIR / run_arg
        if run_dir.is_dir():
            return run_dir
        print(f"Run not found: {run_arg}")
        sys.exit(1)
    runs = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_dir() and len(path.name) == 15 and path.name[8] == "_"
    )
    if not runs:
        print(f"No timestamped runs found in {DATA_DIR}")
        sys.exit(1)
    return runs[-1]


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def validate_run(run_dir: Path, require_labels: bool) -> int:
    errors: list[str] = []
    labeled_rows: list[dict[str, object]] = []
    missing_label_rows: list[dict[str, object]] = []
    risk_counts = {level: 0 for level in RISK_LEVELS}
    sample_count = 0
    image_count = 0

    for split in ("train", "test"):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        for loc_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in loc_dir.iterdir() if path.is_dir()):
                sample_count += 1
                sample_id = f"{loc_dir.name}/{sample_dir.name}"
                metadata_path = sample_dir / "metadata.json"
                rgb_path = sample_dir / "rgb.png"
                swir_path = sample_dir / "swir.png"
                annotation_path = sample_dir / "annotation.json"

                metadata: dict[str, object] = {"sample_id": sample_id}
                if not metadata_path.exists():
                    errors.append(f"{sample_id}: missing metadata.json")
                else:
                    try:
                        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                        metadata = raw if isinstance(raw, dict) else metadata
                    except json.JSONDecodeError as exc:
                        errors.append(f"{sample_id}: invalid metadata.json: {exc}")
                    missing = REQUIRED_METADATA - set(metadata)
                    if missing:
                        errors.append(f"{sample_id}: metadata missing {', '.join(sorted(missing))}")

                if not rgb_path.exists():
                    errors.append(f"{sample_id}: missing rgb.png")
                else:
                    image_count += 1
                if not swir_path.exists():
                    errors.append(f"{sample_id}: missing swir.png")
                else:
                    image_count += 1

                if annotation_path.exists():
                    try:
                        label = load_label_text(annotation_path.read_text(encoding="utf-8"))
                        risk_counts[str(label["flood_risk_level"])] += 1
                        labeled_rows.append(metadata)
                    except ValueError as exc:
                        errors.append(f"{sample_id}: invalid annotation.json: {exc}")
                else:
                    missing_label_rows.append(metadata)
                    if require_labels:
                        errors.append(f"{sample_id}: missing annotation.json")

    manifests_dir = run_dir / "manifests"
    write_jsonl(labeled_rows, manifests_dir / "labeled.jsonl")
    write_jsonl(missing_label_rows, manifests_dir / "missing_labels.jsonl")

    print(f"Checking run: {run_dir.name}")
    print(f"  samples: {sample_count}")
    print(f"  images: {image_count}")
    print(f"  labeled: {len(labeled_rows)}")
    print(f"  missing labels: {len(missing_label_rows)}")
    print("  flood_risk_level:")
    for level in RISK_LEVELS:
        print(f"    {level:<6} {risk_counts[level]}")

    if errors:
        print("\nErrors:")
        for error in errors[:100]:
            print(f"  {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Bali flood sample run.")
    parser.add_argument("run", nargs="?", help="Run id or run directory. Defaults to newest run.")
    parser.add_argument("--require-labels", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    sys.exit(validate_run(run_dir, args.require_labels))


if __name__ == "__main__":
    main()
