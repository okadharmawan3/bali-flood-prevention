"""Detect and repair fetched samples with large near-black no-data regions.

The repair keeps the same timestamp and sample id, searches nearby tile centers,
fetches replacement RGB/SWIR images, and updates metadata.json with the new
actual tile center. Old files are backed up before replacement.

Blank/no-data detection uses pixels that are near-black in both RGB and SWIR.
This keeps valid coastal/ocean imagery, where ocean may be dark in SWIR but is
not a missing tile gap in the RGB composite.

Usage:
    uv run scripts/repair_blank_samples.py --run-dir data/20260504_150038 --scan-only
    uv run scripts/repair_blank_samples.py --run-dir data/20260504_150038 --dry-run
    uv run scripts/repair_blank_samples.py --run-dir data/20260504_150038
"""

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from tqdm import tqdm

from bali_flood_prevention.quality import PairQuality, pair_quality
from bali_flood_prevention.simsat import SIMSAT_BASE_URL, fetch_rgb, fetch_swir


@dataclass(frozen=True)
class Candidate:
    lon: float
    lat: float
    distance_km: float


@dataclass(frozen=True)
class RepairOutcome:
    sample_id: str
    status: str
    before_blank: float
    after_blank: float | None = None
    lon: float | None = None
    lat: float | None = None
    error: str | None = None
    rgb_blank: float | None = None
    swir_blank: float | None = None


def iter_sample_dirs(run_dir: Path):
    for split in ("train", "test"):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        for loc_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in loc_dir.iterdir() if path.is_dir()):
                yield sample_dir


def path_sample_id(sample_dir: Path) -> str:
    """Return region/sample_key for a sample directory."""
    return f"{sample_dir.parent.name}/{sample_dir.name}"


def candidate_grid(
    lon: float,
    lat: float,
    radius_km: float,
    step_km: float,
) -> list[Candidate]:
    """Return nearby candidate centers sorted by distance, excluding origin."""
    if radius_km <= 0 or step_km <= 0:
        raise ValueError("radius_km and step_km must be positive")

    steps = math.ceil(radius_km / step_km)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
    if abs(km_per_deg_lon) < 1e-9:
        raise ValueError("cannot build longitude offsets near the poles")

    candidates: list[Candidate] = []
    for north_step in range(-steps, steps + 1):
        for east_step in range(-steps, steps + 1):
            if north_step == 0 and east_step == 0:
                continue
            north_km = north_step * step_km
            east_km = east_step * step_km
            distance = math.hypot(east_km, north_km)
            if distance > radius_km:
                continue
            candidates.append(
                Candidate(
                    lon=round(lon + east_km / km_per_deg_lon, 6),
                    lat=round(lat + north_km / km_per_deg_lat, 6),
                    distance_km=distance,
                )
            )
    return sorted(candidates, key=lambda candidate: (candidate.distance_km, candidate.lat, candidate.lon))


def fetch_pair(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float,
    base_url: str,
) -> tuple[bytes, bytes]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        rgb_future = pool.submit(fetch_rgb, lon, lat, timestamp, size_km, base_url)
        swir_future = pool.submit(fetch_swir, lon, lat, timestamp, size_km, base_url)
        return rgb_future.result(), swir_future.result()


def quality_for_bytes(rgb_bytes: bytes, swir_bytes: bytes, pixel_threshold: int) -> PairQuality:
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rgb_path = tmp_dir / "rgb.png"
        swir_path = tmp_dir / "swir.png"
        rgb_path.write_bytes(rgb_bytes)
        swir_path.write_bytes(swir_bytes)
        return pair_quality(rgb_path, swir_path, pixel_threshold)


def backup_existing_files(sample_dir: Path) -> Path:
    backup_root = sample_dir / "_repair_backup"
    backup_dir = backup_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rgb.png", "swir.png", "metadata.json", "annotation.json"):
        src = sample_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    return backup_dir


def restore_backup_files(sample_dir: Path, backup_dir: Path) -> None:
    """Restore files from a repair backup directory."""
    for name in ("rgb.png", "swir.png", "metadata.json", "annotation.json"):
        dst = sample_dir / name
        src = backup_dir / name
        if src.exists():
            shutil.copy2(src, dst)
        elif dst.exists() and name == "annotation.json":
            dst.unlink()


def repair_sample(
    sample_dir: Path,
    *,
    blank_threshold: float,
    accept_blank_threshold: float,
    pixel_threshold: int,
    radius_km: float,
    step_km: float,
    max_candidates: int,
    base_url: str,
    dry_run: bool,
    include_labeled: bool,
    scan_only: bool,
) -> RepairOutcome:
    metadata_path = sample_dir / "metadata.json"
    rgb_path = sample_dir / "rgb.png"
    swir_path = sample_dir / "swir.png"
    annotation_path = sample_dir / "annotation.json"
    sample_id = sample_dir.name

    if not metadata_path.exists() or not rgb_path.exists() or not swir_path.exists():
        return RepairOutcome(sample_id=sample_id, status="missing_files", before_blank=1.0)
    if annotation_path.exists() and not include_labeled:
        quality = pair_quality(rgb_path, swir_path, pixel_threshold)
        return RepairOutcome(
            sample_id=sample_id,
            status="skipped_labeled",
            before_blank=quality.joint_blank_fraction,
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_id = str(metadata.get("sample_id", sample_id))
    before_quality = pair_quality(rgb_path, swir_path, pixel_threshold)
    before_blank = before_quality.joint_blank_fraction
    if before_blank < blank_threshold:
        return RepairOutcome(sample_id=sample_id, status="ok", before_blank=before_blank)
    if scan_only:
        return RepairOutcome(sample_id=sample_id, status="bad", before_blank=before_blank)

    lon = float(metadata["lon"])
    lat = float(metadata["lat"])
    timestamp = str(metadata["timestamp"])
    size_km = float(metadata["size_km"])

    best: tuple[Candidate, PairQuality] | None = None
    backup_dir: Path | None = None
    postcheck_failures = 0
    candidates = candidate_grid(lon, lat, radius_km, step_km)[:max_candidates]
    for candidate in candidates:
        try:
            rgb_bytes, swir_bytes = fetch_pair(
                candidate.lon,
                candidate.lat,
                timestamp,
                size_km,
                base_url,
            )
            quality = quality_for_bytes(rgb_bytes, swir_bytes, pixel_threshold)
        except requests.HTTPError:
            continue
        except (requests.ConnectionError, requests.Timeout):
            continue
        if best is None or quality.joint_blank_fraction < best[1].joint_blank_fraction:
            best = (candidate, quality)
        if quality.joint_blank_fraction > accept_blank_threshold:
            continue
        if quality.joint_blank_fraction >= before_blank:
            continue

        if dry_run:
            return RepairOutcome(
                sample_id=sample_id,
                status="would_repair",
                before_blank=before_blank,
                after_blank=quality.joint_blank_fraction,
                lon=candidate.lon,
                lat=candidate.lat,
                rgb_blank=quality.rgb.blank_fraction,
                swir_blank=quality.swir.blank_fraction,
            )

        if backup_dir is None:
            backup_dir = backup_existing_files(sample_dir)
        rgb_path.write_bytes(rgb_bytes)
        swir_path.write_bytes(swir_bytes)
        post_quality = pair_quality(rgb_path, swir_path, pixel_threshold)
        post_blank = post_quality.joint_blank_fraction
        if post_blank > accept_blank_threshold:
            postcheck_failures += 1
            restore_backup_files(sample_dir, backup_dir)
            continue

        repair_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "near_black_no_data_fraction",
            "blank_threshold": blank_threshold,
            "accept_blank_threshold": accept_blank_threshold,
            "pixel_threshold": pixel_threshold,
            "previous_lon": lon,
            "previous_lat": lat,
            "replacement_lon": candidate.lon,
            "replacement_lat": candidate.lat,
            "replacement_distance_km": round(candidate.distance_km, 3),
            "before_rgb_blank_fraction": before_quality.rgb.blank_fraction,
            "before_swir_blank_fraction": before_quality.swir.blank_fraction,
            "before_joint_blank_fraction": before_quality.joint_blank_fraction,
            "after_rgb_blank_fraction": post_quality.rgb.blank_fraction,
            "after_swir_blank_fraction": post_quality.swir.blank_fraction,
            "after_joint_blank_fraction": post_quality.joint_blank_fraction,
            "postcheck_failures_before_success": postcheck_failures,
            "backup_dir": str(backup_dir.relative_to(sample_dir)),
        }
        metadata["lon"] = candidate.lon
        metadata["lat"] = candidate.lat
        metadata.setdefault("repair_history", []).append(repair_record)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if annotation_path.exists():
            annotation_path.unlink()

        return RepairOutcome(
            sample_id=sample_id,
            status="repaired",
            before_blank=before_blank,
            after_blank=post_blank,
            lon=candidate.lon,
            lat=candidate.lat,
            rgb_blank=post_quality.rgb.blank_fraction,
            swir_blank=post_quality.swir.blank_fraction,
        )

    if best is None:
        return RepairOutcome(
            sample_id=sample_id,
            status="no_replacement",
            before_blank=before_blank,
            error="No candidate image could be fetched.",
        )

    if postcheck_failures:
        if backup_dir is not None:
            restore_backup_files(sample_dir, backup_dir)
        return RepairOutcome(
            sample_id=sample_id,
            status="postcheck_failed",
            before_blank=before_blank,
            after_blank=None,
            error=(
                f"{postcheck_failures} acceptable candidate(s) failed post-write "
                "quality checks; original files restored."
            ),
        )

    candidate, best_quality = best
    if best_quality.joint_blank_fraction > accept_blank_threshold:
        return RepairOutcome(
            sample_id=sample_id,
            status="no_acceptable_replacement",
            before_blank=before_blank,
            after_blank=best_quality.joint_blank_fraction,
            lon=candidate.lon,
            lat=candidate.lat,
            error=(
                f"Best candidate did not pass accept threshold "
                f"{accept_blank_threshold:.3f}."
            ),
            rgb_blank=best_quality.rgb.blank_fraction,
            swir_blank=best_quality.swir.blank_fraction,
        )

    return RepairOutcome(
        sample_id=sample_id,
        status="no_better_replacement",
        before_blank=before_blank,
        after_blank=best_quality.joint_blank_fraction,
        lon=candidate.lon,
        lat=candidate.lat,
        rgb_blank=best_quality.rgb.blank_fraction,
        swir_blank=best_quality.swir.blank_fraction,
    )


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair samples whose RGB/SWIR images share mostly near-black no-data pixels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Path to data/{run_id}.")
    parser.add_argument("--blank-threshold", type=float, default=0.05, help="Mark sample bad at or above this joint RGB+SWIR no-data fraction.")
    parser.add_argument("--accept-blank-threshold", type=float, default=0.15, help="Prefer replacements at or below this joint RGB+SWIR no-data fraction.")
    parser.add_argument("--pixel-threshold", type=int, default=3, help="Channel max at or below this value counts as blank in one image.")
    parser.add_argument("--radius-km", type=float, default=20.0, help="Nearby search radius around the current tile center.")
    parser.add_argument("--step-km", type=float, default=5.0, help="Search grid step size.")
    parser.add_argument("--max-candidates", type=int, default=80, help="Maximum nearby candidates to try per bad sample.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of sample folders to process in parallel.")
    parser.add_argument("--base-url", default=SIMSAT_BASE_URL, help="SimSat base URL.")
    parser.add_argument("--scan-only", action="store_true", help="Only detect blank samples locally; do not call SimSat or search replacements.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report replacements without modifying files.")
    parser.add_argument("--include-labeled", action="store_true", help="Allow repairing samples that already have annotation.json.")
    parser.add_argument("--region", default=None, help="Only scan one region id, e.g. bangli_bali.")
    parser.add_argument("--sample-id", action="append", default=None, help="Only scan this sample id, e.g. bangli_bali/p00_s00_t20. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sample folders scanned.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    sample_dirs = list(iter_sample_dirs(run_dir))
    if args.region is not None:
        sample_dirs = [path for path in sample_dirs if path.parent.name == args.region]
    if args.sample_id is not None:
        wanted = set(args.sample_id)
        sample_dirs = [path for path in sample_dirs if path_sample_id(path) in wanted]
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    def _process(sample_dir: Path) -> RepairOutcome:
        return repair_sample(
            sample_dir,
            blank_threshold=args.blank_threshold,
            accept_blank_threshold=args.accept_blank_threshold,
            pixel_threshold=args.pixel_threshold,
            radius_km=args.radius_km,
            step_km=args.step_km,
            max_candidates=args.max_candidates,
            base_url=args.base_url,
            dry_run=args.dry_run,
            include_labeled=args.include_labeled,
            scan_only=args.scan_only,
        )

    outcomes: list[RepairOutcome] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process, sample_dir): sample_dir for sample_dir in sample_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="samples", unit="sample"):
            sample_dir = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = RepairOutcome(
                    sample_id=path_sample_id(sample_dir),
                    status="error",
                    before_blank=1.0,
                    error=str(exc),
                )
            outcomes.append(outcome)
            if outcome.status in {
                "bad",
                "repaired",
                "would_repair",
                "no_replacement",
                "no_acceptable_replacement",
                "no_better_replacement",
                "postcheck_failed",
            }:
                after = "n/a" if outcome.after_blank is None else f"{outcome.after_blank:.3f}"
                tqdm.write(
                    f"[{outcome.status}] {outcome.sample_id} "
                    f"joint_blank {outcome.before_blank:.3f} -> {after}"
                )

    status_counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1
        rows.append(
            {
                "sample_id": outcome.sample_id,
                "status": outcome.status,
                "before_blank": outcome.before_blank,
                "after_blank": outcome.after_blank,
                "lon": outcome.lon,
                "lat": outcome.lat,
                "error": outcome.error,
                "rgb_blank": outcome.rgb_blank,
                "swir_blank": outcome.swir_blank,
            }
        )
    if args.scan_only:
        report_name = "repair_blank_scan.jsonl"
    elif args.dry_run:
        report_name = "repair_blank_dry_run.jsonl"
    else:
        report_name = "repair_blank.jsonl"
    write_jsonl(rows, run_dir / "manifests" / report_name)

    print("Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<22} {count}")
    print(f"Report: {run_dir / 'manifests' / report_name}")


if __name__ == "__main__":
    main()
