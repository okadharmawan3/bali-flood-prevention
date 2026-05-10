"""Audit Bali flood annotation.json files for schema and consistency issues.

This script cannot prove a label is visually perfect, but it catches the
problems that usually break dataset quality:
  - missing labels
  - invalid JSON/schema/enums/field order
  - suspicious combinations that should be manually reviewed
  - optional no-data image quality problems

Usage:
    uv run scripts/audit_labels.py 20260504_150038
    uv run scripts/audit_labels.py --run-dir data/20260504_150038 --check-image-quality
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bali_flood_prevention.quality import pair_quality
from bali_flood_prevention.schema import (
    BOOLEAN_FIELDS,
    CONFIDENCE_LEVELS,
    LABEL_FIELDS,
    RISK_LEVELS,
    WATER_EXTENT_LEVELS,
    load_label_text,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    region: str
    sample_dir: Path
    rgb_path: Path
    swir_path: Path
    metadata_path: Path
    annotation_path: Path


@dataclass
class Audit:
    sample_count: int = 0
    labeled_count: int = 0
    missing_count: int = 0
    hard_errors: list[dict[str, object]] = field(default_factory=list)
    review_items: list[dict[str, object]] = field(default_factory=list)
    risk_counts: Counter[str] = field(default_factory=Counter)
    confidence_counts: Counter[str] = field(default_factory=Counter)
    water_extent_counts: Counter[str] = field(default_factory=Counter)
    boolean_true_counts: Counter[str] = field(default_factory=Counter)
    region_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))


def resolve_run_dir(run_arg: str | None, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if run_dir.is_dir():
            return run_dir
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if run_arg:
        candidate = Path(run_arg)
        if candidate.is_dir():
            return candidate
        run_dir = DATA_DIR / run_arg
        if run_dir.is_dir():
            return run_dir
        raise FileNotFoundError(f"Run not found: {run_arg}")
    runs = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_dir() and len(path.name) == 15 and path.name[8] == "_"
    )
    if not runs:
        raise FileNotFoundError(f"No timestamped runs found in {DATA_DIR}")
    return runs[-1]


def iter_samples(run_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split in ("train", "test"):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        for loc_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in loc_dir.iterdir() if path.is_dir()):
                sample_id = f"{loc_dir.name}/{sample_dir.name}"
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        split=split,
                        region=loc_dir.name,
                        sample_dir=sample_dir,
                        rgb_path=sample_dir / "rgb.png",
                        swir_path=sample_dir / "swir.png",
                        metadata_path=sample_dir / "metadata.json",
                        annotation_path=sample_dir / "annotation.json",
                    )
                )
    return samples


def load_metadata(sample: Sample) -> dict[str, Any]:
    if not sample.metadata_path.exists():
        raise FileNotFoundError("missing metadata.json")
    metadata = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must be a JSON object")
    return metadata


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def hard_error(audit: Audit, sample: Sample, issue: str, detail: str | None = None) -> None:
    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "region": sample.region,
        "issue": issue,
        "sample_dir": str(sample.sample_dir),
    }
    if detail:
        row["detail"] = detail
    audit.hard_errors.append(row)


def review_item(
    audit: Audit,
    sample: Sample,
    issue: str,
    severity: str,
    label: dict[str, object],
    detail: str | None = None,
) -> None:
    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "region": sample.region,
        "issue": issue,
        "severity": severity,
        "annotation_path": str(sample.annotation_path),
        "rgb_path": str(sample.rgb_path),
        "swir_path": str(sample.swir_path),
        "flood_risk_level": label["flood_risk_level"],
        "water_extent_level": label["water_extent_level"],
        "confidence": label["confidence"],
    }
    if detail:
        row["detail"] = detail
    audit.review_items.append(row)


def check_file_set(audit: Audit, sample: Sample) -> None:
    for path, name in (
        (sample.rgb_path, "rgb.png"),
        (sample.swir_path, "swir.png"),
        (sample.metadata_path, "metadata.json"),
    ):
        if not path.exists():
            hard_error(audit, sample, f"missing {name}")


def check_metadata(audit: Audit, sample: Sample, metadata: dict[str, Any]) -> None:
    expected = {
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
    }
    missing = expected - set(metadata)
    if missing:
        hard_error(audit, sample, "metadata missing fields", ", ".join(sorted(missing)))
    if metadata.get("region") not in {None, sample.region}:
        hard_error(
            audit,
            sample,
            "metadata region mismatch",
            f"metadata={metadata.get('region')} folder={sample.region}",
        )
    if metadata.get("split") not in {None, sample.split}:
        hard_error(
            audit,
            sample,
            "metadata split mismatch",
            f"metadata={metadata.get('split')} folder={sample.split}",
        )


def check_canonical_json(audit: Audit, sample: Sample, label: dict[str, object]) -> None:
    raw = json.loads(sample.annotation_path.read_text(encoding="utf-8"))
    if list(raw.keys()) != list(LABEL_FIELDS):
        hard_error(
            audit,
            sample,
            "annotation field order mismatch",
            "annotation.json should use canonical schema order",
        )
    if raw != dict(label):
        hard_error(
            audit,
            sample,
            "annotation canonical value mismatch",
            "annotation.json validates but differs after canonical parsing",
        )


def check_consistency(audit: Audit, sample: Sample, label: dict[str, object]) -> None:
    risk = str(label["flood_risk_level"])
    extent = str(label["water_extent_level"])
    confidence = str(label["confidence"])
    standing = bool(label["standing_water_present"])
    temporary = bool(label["temporary_inundation_likely"])
    permanent = bool(label["permanent_water_body_present"])
    river_context = bool(label["river_or_coastal_overflow_context"])
    urban = bool(label["urban_or_infrastructure_exposure"])
    transport = bool(label["road_or_transport_disruption_likely"])
    cropland = bool(label["cropland_or_settlement_exposure"])
    saturation = bool(label["vegetation_or_soil_saturation"])
    cloud_limited = bool(label["cloud_shadow_or_image_quality_limited"])

    if temporary and not standing:
        review_item(
            audit,
            sample,
            "temporary inundation true but standing water false",
            "high",
            label,
        )
    if permanent and not standing:
        review_item(
            audit,
            sample,
            "permanent water true but standing water false",
            "high",
            label,
        )
    if extent == "large" and not standing and not saturation:
        review_item(
            audit,
            sample,
            "large water extent without standing water or saturation",
            "high",
            label,
        )
    if risk == "high" and not any([standing, temporary, saturation]):
        review_item(
            audit,
            sample,
            "high flood risk without visible water/saturation evidence",
            "high",
            label,
        )
    if temporary and risk == "low":
        review_item(
            audit,
            sample,
            "temporary inundation true but flood risk low",
            "medium",
            label,
        )
    if cloud_limited and confidence == "high":
        review_item(
            audit,
            sample,
            "cloud/image limited but confidence high",
            "medium",
            label,
        )
    if transport and not urban:
        review_item(
            audit,
            sample,
            "transport disruption true but infrastructure exposure false",
            "medium",
            label,
        )
    if river_context and not any([standing, permanent]):
        review_item(
            audit,
            sample,
            "river/coastal context true but no standing/permanent water",
            "low",
            label,
        )
    if sample.region == "denpasar_bali" and not urban:
        review_item(
            audit,
            sample,
            "Denpasar sample with urban/infrastructure exposure false",
            "low",
            label,
        )
    if not any([urban, cropland, river_context, permanent]) and confidence == "high":
        review_item(
            audit,
            sample,
            "all major exposure/context fields false with high confidence",
            "low",
            label,
        )


def check_image_quality(
    audit: Audit,
    sample: Sample,
    label: dict[str, object],
    blank_threshold: float,
) -> None:
    try:
        quality = pair_quality(sample.rgb_path, sample.swir_path)
    except Exception as exc:
        hard_error(audit, sample, "image quality check failed", str(exc))
        return
    if quality.joint_blank_fraction >= blank_threshold:
        review_item(
            audit,
            sample,
            "image pair still has high no-data blank fraction",
            "high",
            label,
            f"joint_blank_fraction={quality.joint_blank_fraction:.3f}",
        )


def update_counts(audit: Audit, sample: Sample, label: dict[str, object]) -> None:
    audit.risk_counts[str(label["flood_risk_level"])] += 1
    audit.confidence_counts[str(label["confidence"])] += 1
    audit.water_extent_counts[str(label["water_extent_level"])] += 1
    audit.region_counts[sample.region][str(label["flood_risk_level"])] += 1
    for field_name in BOOLEAN_FIELDS:
        if label[field_name]:
            audit.boolean_true_counts[field_name] += 1


def audit_run(
    run_dir: Path,
    allow_missing: bool,
    check_images: bool,
    blank_threshold: float,
) -> Audit:
    audit = Audit()
    for sample in iter_samples(run_dir):
        audit.sample_count += 1
        check_file_set(audit, sample)
        try:
            metadata = load_metadata(sample)
            check_metadata(audit, sample, metadata)
        except Exception as exc:
            hard_error(audit, sample, "invalid metadata.json", str(exc))

        if not sample.annotation_path.exists():
            audit.missing_count += 1
            if not allow_missing:
                hard_error(audit, sample, "missing annotation.json")
            continue

        try:
            label = load_label_text(sample.annotation_path.read_text(encoding="utf-8"))
            audit.labeled_count += 1
            check_canonical_json(audit, sample, label)
            check_consistency(audit, sample, label)
            if check_images:
                check_image_quality(audit, sample, label, blank_threshold)
            update_counts(audit, sample, label)
        except Exception as exc:
            hard_error(audit, sample, "invalid annotation.json", str(exc))
    return audit


def summary_dict(audit: Audit) -> dict[str, object]:
    enum_counts = {
        "flood_risk_level": {level: audit.risk_counts[level] for level in RISK_LEVELS},
        "water_extent_level": {level: audit.water_extent_counts[level] for level in WATER_EXTENT_LEVELS},
        "confidence": {level: audit.confidence_counts[level] for level in CONFIDENCE_LEVELS},
    }
    boolean_rates = {
        field_name: {
            "true": audit.boolean_true_counts[field_name],
            "true_rate": round(
                audit.boolean_true_counts[field_name] / audit.labeled_count,
                4,
            )
            if audit.labeled_count
            else 0.0,
        }
        for field_name in BOOLEAN_FIELDS
    }
    return {
        "samples": audit.sample_count,
        "labeled": audit.labeled_count,
        "missing_labels": audit.missing_count,
        "hard_errors": len(audit.hard_errors),
        "review_items": len(audit.review_items),
        "enum_counts": enum_counts,
        "boolean_rates": boolean_rates,
        "risk_by_region": {
            region: {level: counts[level] for level in RISK_LEVELS}
            for region, counts in sorted(audit.region_counts.items())
        },
    }


def print_summary(audit: Audit, manifest_dir: Path) -> None:
    print("Label audit summary:")
    print(f"  samples:        {audit.sample_count}")
    print(f"  labeled:        {audit.labeled_count}")
    print(f"  missing labels: {audit.missing_count}")
    print(f"  hard errors:    {len(audit.hard_errors)}")
    print(f"  review items:   {len(audit.review_items)}")
    print("  flood_risk_level:")
    for level in RISK_LEVELS:
        print(f"    {level:<6} {audit.risk_counts[level]}")
    print("  confidence:")
    for level in CONFIDENCE_LEVELS:
        print(f"    {level:<6} {audit.confidence_counts[level]}")
    print(f"Reports written to: {manifest_dir}")


def markdown_count_table(title: str, counts: Counter[str], levels: tuple[str, ...], total: int) -> list[str]:
    lines = [f"## {title}", "", "| Label | Count | Percent |", "|---|---:|---:|"]
    for level in levels:
        count = counts[level]
        lines.append(f"| {level} | {count} | {pct(count, total)} |")
    lines.append("")
    return lines


def write_markdown_report(path: Path, run_dir: Path, audit: Audit) -> None:
    lines: list[str] = [
        "# Bali Flood Label Audit Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Dataset Status",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Samples | {audit.sample_count} |",
        f"| Labeled | {audit.labeled_count} |",
        f"| Missing labels | {audit.missing_count} |",
        f"| Hard errors | {len(audit.hard_errors)} |",
        f"| Review items | {len(audit.review_items)} |",
        "",
    ]
    lines.extend(markdown_count_table("Flood Risk Level", audit.risk_counts, RISK_LEVELS, audit.labeled_count))
    lines.extend(markdown_count_table("Confidence", audit.confidence_counts, CONFIDENCE_LEVELS, audit.labeled_count))
    lines.extend(markdown_count_table("Water Extent Level", audit.water_extent_counts, WATER_EXTENT_LEVELS, audit.labeled_count))

    lines.extend(
        [
            "## Flood Risk By Region",
            "",
            "| Region | Low | Medium | High | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for region, counts in sorted(audit.region_counts.items()):
        total = sum(counts[level] for level in RISK_LEVELS)
        lines.append(
            f"| {region} | {counts['low']} | {counts['medium']} | {counts['high']} | {total} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Boolean Label True Counts",
            "",
            "| Field | True Count | True Percent |",
            "|---|---:|---:|",
        ]
    )
    for field_name in BOOLEAN_FIELDS:
        count = audit.boolean_true_counts[field_name]
        lines.append(f"| {field_name} | {count} | {pct(count, audit.labeled_count)} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit annotation.json files for schema validity and suspicious flood-label logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run", nargs="?", help="Run id or run directory. Defaults to newest run.")
    parser.add_argument("--run-dir", default=None, help="Explicit path to data/{run_id}.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if some annotation.json files are missing.")
    parser.add_argument("--check-image-quality", action="store_true", help="Also check RGB/SWIR no-data blank fraction.")
    parser.add_argument("--blank-threshold", type=float, default=0.05)
    parser.add_argument("--fail-on-review", action="store_true", help="Exit non-zero if suspicious review items are found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run, args.run_dir)
    audit = audit_run(
        run_dir=run_dir,
        allow_missing=args.allow_missing,
        check_images=args.check_image_quality,
        blank_threshold=args.blank_threshold,
    )

    manifest_dir = run_dir / "manifests"
    write_json(manifest_dir / "label_audit_summary.json", summary_dict(audit))
    write_jsonl(manifest_dir / "label_hard_errors.jsonl", audit.hard_errors)
    write_jsonl(manifest_dir / "label_review_queue.jsonl", audit.review_items)
    write_markdown_report(manifest_dir / "label_audit_report.md", run_dir, audit)
    print_summary(audit, manifest_dir)

    if audit.hard_errors:
        sys.exit(1)
    if args.fail_on_review and audit.review_items:
        sys.exit(2)


if __name__ == "__main__":
    main()
