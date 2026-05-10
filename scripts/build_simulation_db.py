"""Build the precomputed Bali flood simulation database.

This script fetches January Sentinel-2 image observations from SimSat and runs
the fine-tuned local GGUF models through llama-server. The dashboard replays the
stored database, so no model inference is needed during playback.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bali_flood_prevention.evaluator import (  # noqa: E402
    EvalSample,
    make_llama_backend,
    start_llama_server,
    stop_server,
    wait_for_server,
)
from bali_flood_prevention.points import generate_points  # noqa: E402
from bali_flood_prevention.schema import empty_label, validate_label  # noqa: E402
from bali_flood_prevention.simsat import SIMSAT_BASE_URL, fetch_rgb, fetch_swir  # noqa: E402
from bali_flood_prevention.simulation import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MODELS,
    ModelSpec,
    connect_db,
    create_run,
    insert_checkpoint,
    insert_observation,
    insert_passes,
    insert_prediction,
    refresh_aggregates,
)


@dataclass(frozen=True)
class ObservationWork:
    observation_id: int
    sample_id: str
    split: str
    region: str
    timestamp: str
    rgb_path: Path
    swir_path: Path
    metadata: dict[str, object]


def parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_metadata_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_models(value: str) -> list[ModelSpec]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--models must contain at least one model id")
    unknown = [model_id for model_id in ids if model_id not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown model id(s): {', '.join(unknown)}")
    return [DEFAULT_MODELS[model_id] for model_id in ids]


def sentinel_metadata_from_response(response: requests.Response) -> dict[str, object]:
    raw = response.headers.get("sentinel_metadata")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fetch_rgb_with_metadata(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float,
    base_url: str,
) -> tuple[bytes, dict[str, object]]:
    params: list[tuple[str, object]] = [
        ("lon", lon),
        ("lat", lat),
        ("timestamp", timestamp),
        ("size_km", size_km),
        ("return_type", "png"),
        ("spectral_bands", "red"),
        ("spectral_bands", "green"),
        ("spectral_bands", "blue"),
    ]
    response = requests.get(f"{base_url}/data/image/sentinel", params=params, timeout=60)
    response.raise_for_status()
    return response.content, sentinel_metadata_from_response(response)


def discover_pass_timestamps(
    *,
    start_date: datetime,
    end_date: datetime,
    max_timesteps: int,
    lon: float,
    lat: float,
    size_km: float,
    base_url: str,
) -> list[str]:
    """Probe daily timestamps and keep unique SimSat acquisition datetimes."""
    if max_timesteps < 1:
        raise ValueError("max_timesteps must be >= 1")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    discovered: list[str] = []
    seen: set[str] = set()
    current = start_date.replace(hour=12, minute=0, second=0, microsecond=0)
    final = end_date.replace(hour=12, minute=0, second=0, microsecond=0)
    while current <= final and len(discovered) < max_timesteps:
        probe_ts = format_utc(current)
        try:
            image_bytes, metadata = fetch_rgb_with_metadata(lon, lat, probe_ts, size_km, base_url)
        except requests.RequestException:
            current += timedelta(days=1)
            continue
        if not image_bytes:
            current += timedelta(days=1)
            continue
        candidate = str(metadata.get("datetime") or probe_ts)
        candidate_dt = parse_metadata_datetime(candidate)
        if candidate_dt is not None and not (start_date <= candidate_dt <= end_date):
            current += timedelta(days=1)
            continue
        if candidate and candidate not in seen:
            seen.add(candidate)
            discovered.append(candidate)
        current += timedelta(days=1)
    return discovered


def observation_dir(
    images_dir: Path,
    run_id: int,
    area_id: str,
    point_id: str,
    timestep_index: int,
) -> Path:
    return images_dir / f"run_{run_id}" / area_id / point_id / f"t{timestep_index:02d}"


def write_observation_images(
    *,
    images_dir: Path,
    run_id: int,
    area_id: str,
    point_id: str,
    timestep_index: int,
    rgb_bytes: bytes,
    swir_bytes: bytes,
) -> tuple[Path, Path]:
    out_dir = observation_dir(images_dir, run_id, area_id, point_id, timestep_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / "rgb.png"
    swir_path = out_dir / "swir.png"
    rgb_path.write_bytes(rgb_bytes)
    swir_path.write_bytes(swir_bytes)
    return rgb_path, swir_path


def build_observations(
    *,
    conn,
    run_id: int,
    pass_ids: list[int],
    timestamps: list[str],
    points_per_location: int,
    seed: int,
    size_km: float,
    images_dir: Path,
    base_url: str,
) -> list[ObservationWork]:
    points = generate_points(
        points_per_location=points_per_location,
        seed=seed,
        cache_dir=ROOT / "data" / "boundaries",
        refresh_boundaries=False,
    )
    checkpoint_ids: dict[tuple[str, str], int] = {}
    for point in points:
        checkpoint_ids[(point.location_id, point.point_id)] = insert_checkpoint(
            conn,
            run_id=run_id,
            area_id=point.location_id,
            area_name=point.location_name,
            point_id=point.point_id,
            point_index=point.point_index,
            lon=point.lon,
            lat=point.lat,
            source=point.source,
        )
    conn.commit()

    works: list[ObservationWork] = []
    total = len(points) * len(timestamps)
    completed = 0
    for pass_index, (pass_id, timestamp) in enumerate(zip(pass_ids, timestamps)):
        for point in points:
            completed += 1
            sample_id = f"{point.location_id}/{point.point_id}_t{pass_index:02d}"
            print(f"[fetch {completed}/{total}] {sample_id} {timestamp}", flush=True)
            try:
                rgb_bytes, metadata = fetch_rgb_with_metadata(
                    point.lon, point.lat, timestamp, size_km, base_url
                )
                swir_bytes = fetch_swir(point.lon, point.lat, timestamp, size_km, base_url)
                if not rgb_bytes or not swir_bytes:
                    raise RuntimeError("SimSat returned an empty RGB or SWIR image")
                rgb_path, swir_path = write_observation_images(
                    images_dir=images_dir,
                    run_id=run_id,
                    area_id=point.location_id,
                    point_id=point.point_id,
                    timestep_index=pass_index,
                    rgb_bytes=rgb_bytes,
                    swir_bytes=swir_bytes,
                )
                status = "ready"
                error = None
            except (requests.RequestException, RuntimeError) as exc:
                metadata = {}
                rgb_path = None
                swir_path = None
                status = "skipped"
                error = str(exc)

            observation_id = insert_observation(
                conn,
                run_id=run_id,
                pass_id=pass_id,
                checkpoint_id=checkpoint_ids[(point.location_id, point.point_id)],
                area_id=point.location_id,
                point_id=point.point_id,
                timestamp=timestamp,
                lon=point.lon,
                lat=point.lat,
                size_km=size_km,
                rgb_path=str(rgb_path) if rgb_path else None,
                swir_path=str(swir_path) if swir_path else None,
                sentinel_metadata=metadata,
                status=status,
                error=error,
            )
            conn.commit()

            if rgb_path is not None and swir_path is not None:
                works.append(
                    ObservationWork(
                        observation_id=observation_id,
                        sample_id=sample_id,
                        split="simulation",
                        region=point.location_id,
                        timestamp=timestamp,
                        rgb_path=rgb_path,
                        swir_path=swir_path,
                        metadata={
                            "sample_id": sample_id,
                            "region": point.location_id,
                            "point_id": point.point_id,
                            "timestamp": timestamp,
                            "lon": point.lon,
                            "lat": point.lat,
                            "size_km": size_km,
                            "sentinel": metadata,
                        },
                    )
                )
    return works


def make_eval_sample(work: ObservationWork) -> EvalSample:
    return EvalSample(
        id=work.sample_id,
        split=work.split,
        region=work.region,
        timestamp=work.timestamp,
        rgb_path=work.rgb_path,
        swir_path=work.swir_path,
        metadata_path=work.rgb_path.parent / "metadata.json",
        annotation_path=work.rgb_path.parent / "annotation.json",
        rgb_bytes=work.rgb_path.read_bytes(),
        swir_bytes=work.swir_path.read_bytes(),
        metadata=work.metadata,
        ground_truth=empty_label(),
    )


def ensure_model_files(models: Iterable[ModelSpec]) -> None:
    missing: list[Path] = []
    for spec in models:
        for path in (spec.model_path, spec.mmproj_path, spec.chat_template_file):
            if path is not None and not path.is_file():
                missing.append(path)
    if missing:
        paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model artifact(s):\n{paths}")


def run_model_predictions(
    *,
    conn,
    run_id: int,
    works: list[ObservationWork],
    model: ModelSpec,
    port: int,
    verbose_server: bool,
) -> None:
    print(f"Starting llama-server for {model.id} on port {port} ...", flush=True)
    server = start_llama_server(
        str(model.model_path),
        port=port,
        verbose=verbose_server,
        mmproj=str(model.mmproj_path),
        chat_template_file=str(model.chat_template_file) if model.chat_template_file else None,
    )
    try:
        wait_for_server(port=port)
        print(f"llama-server ready for {model.id}.", flush=True)
        predict = make_llama_backend(str(model.model_path), port)
        for index, work in enumerate(works, start=1):
            sample = make_eval_sample(work)
            t0 = perf_counter()
            try:
                raw_prediction = predict(sample)
                prediction = dict(validate_label(raw_prediction))
                valid = True
                error = None
            except Exception as exc:
                prediction = None
                valid = False
                error = str(exc)
            latency_s = perf_counter() - t0
            insert_prediction(
                conn,
                run_id=run_id,
                observation_id=work.observation_id,
                model_id=model.id,
                model_label=model.label,
                prediction=prediction,
                valid_json=valid,
                error=error,
                latency_s=latency_s,
            )
            conn.commit()
            status = "ok" if valid else "error"
            print(
                f"[{model.id} {index}/{len(works)}] {work.sample_id} {status} "
                f"latency={latency_s:.2f}s",
                flush=True,
            )
    finally:
        stop_server(server)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the precomputed Bali flood simulation SQLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-01-31")
    parser.add_argument("--max-timesteps", type=int, default=6)
    parser.add_argument("--points-per-location", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-km", type=float, default=5.0)
    parser.add_argument("--models", default="lfm2,smolvlm2")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--base-url", default=SIMSAT_BASE_URL)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--verbose-server", action="store_true")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Fetch observations and build DB rows without calling llama-server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if "T" not in args.end_date:
        end_date = end_date + timedelta(days=1) - timedelta(seconds=1)
    models = parse_models(args.models)
    if not args.skip_inference:
        if not shutil.which("llama-server"):
            raise RuntimeError("llama-server not found on PATH")
        ensure_model_files(models)

    conn = connect_db(Path(args.db))
    probe_lon = 115.2167
    probe_lat = -8.65
    print("Discovering Sentinel timesteps for January 2026 ...", flush=True)
    timestamps = discover_pass_timestamps(
        start_date=start_date,
        end_date=end_date,
        max_timesteps=args.max_timesteps,
        lon=probe_lon,
        lat=probe_lat,
        size_km=args.size_km,
        base_url=args.base_url,
    )
    if not timestamps:
        raise RuntimeError("No Sentinel timesteps discovered. Is SimSat running?")
    print(f"Discovered {len(timestamps)} timestep(s):")
    for timestamp in timestamps:
        print(f"  {timestamp}")

    run_id = create_run(
        conn,
        name=f"bali-flood-sim-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_date=args.start_date,
        end_date=args.end_date,
        max_timesteps=args.max_timesteps,
        points_per_location=args.points_per_location,
        seed=args.seed,
        size_km=args.size_km,
        metadata={"models": [model.id for model in models]},
    )
    pass_ids = insert_passes(conn, run_id, timestamps, source="simsat_discovery")
    works = build_observations(
        conn=conn,
        run_id=run_id,
        pass_ids=pass_ids,
        timestamps=timestamps,
        points_per_location=args.points_per_location,
        seed=args.seed,
        size_km=args.size_km,
        images_dir=Path(args.images_dir),
        base_url=args.base_url,
    )
    print(f"Ready observations: {len(works)}", flush=True)

    if not args.skip_inference:
        for offset, model in enumerate(models):
            run_model_predictions(
                conn=conn,
                run_id=run_id,
                works=works,
                model=model,
                port=args.port + offset,
                verbose_server=args.verbose_server,
            )
        refresh_aggregates(conn, run_id)
        print("Aggregates refreshed.", flush=True)

    print(f"Simulation DB ready: {Path(args.db)}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
