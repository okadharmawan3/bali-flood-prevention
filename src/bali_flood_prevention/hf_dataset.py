"""Helpers for packaging labeled samples in Hugging Face dataset shape."""

import json
import shutil
from pathlib import Path
from typing import Iterable

from bali_flood_prevention.schema import dumps_label, load_label_text


HF_COLUMNS = (
    "region",
    "point_id",
    "timestamp",
    "split",
    "rgb_path",
    "swir_path",
    "output",
)


def collect_rows(run_dir: Path, images_dir: Path | None = None) -> list[dict[str, str]]:
    """Collect labeled sample folders and optionally copy images into images_dir."""
    rows: list[dict[str, str]] = []
    for ann_path in sorted(run_dir.glob("*/*/p*_s*_t*/annotation.json")):
        tile_dir = ann_path.parent
        split = tile_dir.parent.parent.name
        region = tile_dir.parent.name
        sample_key = tile_dir.name
        metadata_path = tile_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json beside {ann_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        label = load_label_text(ann_path.read_text(encoding="utf-8"))

        rgb_path = tile_dir / "rgb.png"
        swir_path = tile_dir / "swir.png"
        if not rgb_path.exists() or not swir_path.exists():
            raise FileNotFoundError(f"Missing rgb.png or swir.png beside {ann_path}")

        rgb_name = f"{region}_{sample_key}_rgb.png"
        swir_name = f"{region}_{sample_key}_swir.png"
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rgb_path, images_dir / rgb_name)
            shutil.copy2(swir_path, images_dir / swir_name)

        rows.append(
            {
                "region": region,
                "point_id": str(metadata["point_id"]),
                "timestamp": str(metadata["timestamp"]),
                "split": split,
                "rgb_path": f"images/{rgb_name}",
                "swir_path": f"images/{swir_name}",
                "output": dumps_label(label),
            }
        )
    return rows


def write_jsonl(rows: Iterable[dict[str, str]], path: Path) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in materialized)
        + ("\n" if materialized else ""),
        encoding="utf-8",
    )
    return len(materialized)


def split_rows(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    materialized = list(rows)
    train_rows = [row for row in materialized if row["split"] == "train"]
    test_rows = [row for row in materialized if row["split"] == "test"]
    return train_rows, test_rows


def dataset_features():
    """Return HF datasets Features for the packaged tabular rows."""
    from datasets import Features, Value

    return Features({name: Value("string") for name in HF_COLUMNS})


def dataset_dict_from_rows(rows: list[dict[str, str]]):
    """Create a DatasetDict from packaged rows."""
    from datasets import Dataset, DatasetDict

    features = dataset_features()
    train_rows, test_rows = split_rows(rows)
    ds_dict = {"train": Dataset.from_list(train_rows, features=features)}
    if test_rows:
        ds_dict["test"] = Dataset.from_list(test_rows, features=features)
    return DatasetDict(ds_dict)
