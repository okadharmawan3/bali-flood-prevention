"""Fetch Bali flood-prevention image samples without labeling them.

Each successful sample gets:
    rgb.png
    swir.png
    metadata.json

Labeling is intentionally separate: use skills/bali-flood-labeler/SKILL.md
with Codex to inspect the images and write annotation.json files.

Usage:
    uv run scripts/generate_samples.py ^
      --start-date 2024-01-01 --end-date 2025-12-31 ^
      --points-per-location 10 --n-temporal-tiles 48 --n-spatial-tiles 4 ^
      --test-ratio 0.2 --concurrency 3
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from bali_flood_prevention.locations import LOCATIONS_BY_ID
from bali_flood_prevention.points import SamplePoint, generate_points, write_points_manifest
from bali_flood_prevention.simsat import SIMSAT_BASE_URL, fetch_rgb, fetch_swir
from bali_flood_prevention.tiles import TileCoord, spatial_grid, temporal_timestamps, train_test_cutoff

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class TileTask:
    point: SamplePoint
    spatial: TileCoord
    timestamp: str
    split: str
    spatial_index: int
    temporal_index: int

    @property
    def sample_key(self) -> str:
        return f"{self.point.point_id}_s{self.spatial_index:02d}_t{self.temporal_index:02d}"

    @property
    def sample_id(self) -> str:
        return f"{self.point.location_id}/{self.sample_key}"

    def metadata(self, run_dir: Path, size_km: float) -> dict[str, object]:
        rel_dir = Path(self.split) / self.point.location_id / self.sample_key
        return {
            "sample_id": self.sample_id,
            "region": self.point.location_id,
            "location_name": self.point.location_name,
            "point_id": self.point.point_id,
            "point_index": self.point.point_index,
            "point_lon": self.point.lon,
            "point_lat": self.point.lat,
            "point_source": self.point.source,
            "spatial_index": self.spatial_index,
            "temporal_index": self.temporal_index,
            "lon": self.spatial.lon,
            "lat": self.spatial.lat,
            "timestamp": self.timestamp,
            "split": self.split,
            "size_km": size_km,
            "rgb_path": (rel_dir / "rgb.png").as_posix(),
            "swir_path": (rel_dir / "swir.png").as_posix(),
            "metadata_path": (rel_dir / "metadata.json").as_posix(),
            "annotation_path": (rel_dir / "annotation.json").as_posix(),
        }


@dataclass(frozen=True)
class FetchResult:
    status: str
    metadata: dict[str, object]
    error: str | None = None


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def process_tile(
    task: TileTask,
    run_dir: Path,
    size_km: float,
    base_url: str,
    dry_run: bool,
) -> FetchResult:
    metadata = task.metadata(run_dir, size_km)
    sample_dir = run_dir / task.split / task.point.location_id / task.sample_key

    if dry_run:
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return FetchResult(status="dry_run", metadata=metadata)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rgb_future = pool.submit(
                fetch_rgb, task.spatial.lon, task.spatial.lat, task.timestamp, size_km, base_url
            )
            swir_future = pool.submit(
                fetch_swir, task.spatial.lon, task.spatial.lat, task.timestamp, size_km, base_url
            )
            rgb_bytes = rgb_future.result()
            swir_bytes = swir_future.result()
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        return FetchResult(status="skipped", metadata=metadata, error=f"SimSat HTTP {status}")
    except (requests.ConnectionError, requests.Timeout) as exc:
        return FetchResult(status="skipped", metadata=metadata, error=f"SimSat unavailable: {exc}")

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "rgb.png").write_bytes(rgb_bytes)
    (sample_dir / "swir.png").write_bytes(swir_bytes)
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return FetchResult(status="unlabeled", metadata=metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Bali flood image samples from SimSat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="Start of the sampling window.")
    parser.add_argument("--end-date", required=True, help="End of the sampling window.")
    parser.add_argument("--points-per-location", type=int, default=10, help="Random parent points per Bali region.")
    parser.add_argument("--n-temporal-tiles", type=int, default=48, help="Evenly spaced timestamps in the date window.")
    parser.add_argument("--n-spatial-tiles", type=int, default=4, help="Spatial tiles around each parent point.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Latest fraction of the time window reserved for test.")
    parser.add_argument("--size-km", type=float, default=5.0, help="Tile edge length in kilometers.")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of sample fetches running in parallel.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic random point generation.")
    parser.add_argument("--location", choices=list(LOCATIONS_BY_ID), default=None)
    parser.add_argument("--base-url", default=SIMSAT_BASE_URL, help="SimSat base URL.")
    parser.add_argument("--refresh-boundaries", action="store_true", help="Redownload cached Bali boundary data.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata only; do not call SimSat.")
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks for a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        start_dt = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        print(f"Invalid date: {exc}")
        sys.exit(1)
    if end_dt <= start_dt:
        print("--end-date must be after --start-date")
        sys.exit(1)
    if args.test_ratio < 0.0 or args.test_ratio >= 1.0:
        print("--test-ratio must be in [0.0, 1.0)")
        sys.exit(1)

    all_points = generate_points(
        points_per_location=args.points_per_location,
        seed=args.seed,
        cache_dir=DATA_DIR / "boundaries",
        refresh_boundaries=args.refresh_boundaries,
    )
    points = [
        point
        for point in all_points
        if args.location is None or point.location_id == args.location
    ]

    timestamps = temporal_timestamps(start_dt, end_dt, args.n_temporal_tiles)
    cutoff = train_test_cutoff(start_dt, end_dt, args.test_ratio) if args.test_ratio > 0 else None

    tasks: list[TileTask] = []
    for point in points:
        spatial_tiles = spatial_grid(point.lon, point.lat, args.n_spatial_tiles, args.size_km)
        for ti, ts in enumerate(timestamps):
            ts_dt = datetime.fromisoformat(ts)
            split = "test" if cutoff is not None and ts_dt >= cutoff else "train"
            for si, spatial in enumerate(spatial_tiles):
                tasks.append(
                    TileTask(
                        point=point,
                        spatial=spatial,
                        timestamp=ts,
                        split=split,
                        spatial_index=si,
                        temporal_index=ti,
                    )
                )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_DIR / run_id
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "run_id": run_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "points_per_location": args.points_per_location,
        "n_temporal_tiles": args.n_temporal_tiles,
        "n_spatial_tiles": args.n_spatial_tiles,
        "test_ratio": args.test_ratio,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "size_km": args.size_km,
        "seed": args.seed,
        "location": args.location,
        "base_url": args.base_url,
        "dry_run": args.dry_run,
        "task_slots": len(tasks),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_points_manifest(points, manifests_dir / "points.jsonl")
    write_jsonl([task.metadata(run_dir, args.size_km) for task in tasks], manifests_dir / "tasks.jsonl")

    print(
        f"Run: {run_id} | points: {len(points)} | temporal: {args.n_temporal_tiles} "
        f"| spatial: {args.n_spatial_tiles} | task slots: {len(tasks)} | dry_run: {args.dry_run}"
    )

    results: list[FetchResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_tile, task, run_dir, args.size_km, args.base_url, args.dry_run): task
            for task in tasks
        }
        with tqdm(total=len(futures), desc="tiles", unit="tile") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = FetchResult(
                        status="error",
                        metadata=task.metadata(run_dir, args.size_km),
                        error=str(exc),
                    )
                results.append(result)
                pbar.set_postfix(status=result.status)
                pbar.update(1)

    results.sort(key=lambda item: str(item.metadata["sample_id"]))
    unlabeled = [r.metadata for r in results if r.status == "unlabeled"]
    dry_run = [r.metadata for r in results if r.status == "dry_run"]
    skipped = [
        {**r.metadata, "error": r.error or "unknown"}
        for r in results
        if r.status in {"skipped", "error"}
    ]
    write_jsonl(unlabeled, manifests_dir / "unlabeled.jsonl")
    write_jsonl(dry_run, manifests_dir / "dry_run.jsonl")
    write_jsonl(skipped, manifests_dir / "skipped.jsonl")

    print(
        f"Done. unlabeled={len(unlabeled)} dry_run={len(dry_run)} skipped={len(skipped)} "
        f"run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
