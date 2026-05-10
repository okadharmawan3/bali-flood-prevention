"""SQLite persistence and aggregation helpers for the Bali flood simulation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bali_flood_prevention.locations import LOCATIONS
from bali_flood_prevention.schema import (
    BOOLEAN_FIELDS,
    CONFIDENCE_LEVELS,
    LABEL_FIELDS,
    RISK_LEVELS,
    WATER_EXTENT_LEVELS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "bali_simulation.db"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "simulation_images"

SEVERITY_FIELDS: tuple[str, ...] = (
    "standing_water_present",
    "temporary_inundation_likely",
    "urban_or_infrastructure_exposure",
    "road_or_transport_disruption_likely",
    "cropland_or_settlement_exposure",
    "river_or_coastal_overflow_context",
    "low_lying_or_poor_drainage_area",
    "vegetation_or_soil_saturation",
    "permanent_water_body_present",
    "cloud_shadow_or_image_quality_limited",
)

LEVEL_SCORES = {"low": 1.0, "medium": 2.0, "high": 3.0}
WATER_SCORES = {"small": 1.0, "moderate": 2.0, "large": 3.0}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    model_path: Path
    mmproj_path: Path
    chat_template_file: Path | None = None


DEFAULT_MODELS: dict[str, ModelSpec] = {
    "lfm2": ModelSpec(
        id="lfm2",
        label="LFM2.5-VL Bali Flood",
        model_path=PROJECT_ROOT / "outputs" / "lfm2.5-vl-bali-flood-Q8_0.gguf",
        mmproj_path=PROJECT_ROOT / "outputs" / "mmproj-lfm2.5-vl-bali-flood-Q8_0.gguf",
    ),
    "smolvlm2": ModelSpec(
        id="smolvlm2",
        label="SmolVLM2 Bali Flood",
        model_path=PROJECT_ROOT / "outputs" / "smolvlm2-transformers-bali-flood-Q8_0.gguf",
        mmproj_path=PROJECT_ROOT / "outputs" / "mmproj-smolvlm2-transformers-bali-flood-Q8_0.gguf",
        chat_template_file=PROJECT_ROOT / "templates" / "smolvlm-image-url.jinja",
    ),
}


def connect_db(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            max_timesteps INTEGER NOT NULL,
            points_per_location INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            size_km REAL NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS satellite_passes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            timestep_index INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(run_id, timestep_index),
            UNIQUE(run_id, timestamp)
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            area_id TEXT NOT NULL,
            area_name TEXT NOT NULL,
            point_id TEXT NOT NULL,
            point_index INTEGER NOT NULL,
            lon REAL NOT NULL,
            lat REAL NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(run_id, area_id, point_id)
        );

        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            pass_id INTEGER NOT NULL REFERENCES satellite_passes(id) ON DELETE CASCADE,
            checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
            area_id TEXT NOT NULL,
            point_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            lon REAL NOT NULL,
            lat REAL NOT NULL,
            size_km REAL NOT NULL,
            rgb_path TEXT,
            swir_path TEXT,
            sentinel_metadata_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, pass_id, checkpoint_id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            model_label TEXT NOT NULL,
            raw_json TEXT,
            valid_json INTEGER NOT NULL,
            error TEXT,
            latency_s REAL NOT NULL DEFAULT 0,
            flood_risk_level TEXT,
            water_extent_level TEXT,
            standing_water_present INTEGER,
            temporary_inundation_likely INTEGER,
            urban_or_infrastructure_exposure INTEGER,
            road_or_transport_disruption_likely INTEGER,
            cropland_or_settlement_exposure INTEGER,
            river_or_coastal_overflow_context INTEGER,
            low_lying_or_poor_drainage_area INTEGER,
            vegetation_or_soil_saturation INTEGER,
            permanent_water_body_present INTEGER,
            cloud_shadow_or_image_quality_limited INTEGER,
            confidence TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(observation_id, model_id)
        );

        CREATE TABLE IF NOT EXISTS area_timestep_aggregates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            pass_id INTEGER NOT NULL REFERENCES satellite_passes(id) ON DELETE CASCADE,
            area_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            observation_count INTEGER NOT NULL,
            valid_prediction_count INTEGER NOT NULL,
            flood_risk_score REAL,
            flood_risk_level TEXT,
            water_extent_score REAL,
            water_extent_level TEXT,
            severity_score REAL,
            severity_level TEXT,
            confidence_score REAL,
            confidence_level TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(run_id, pass_id, area_id, model_id)
        );

        CREATE INDEX IF NOT EXISTS idx_observations_area_time
            ON observations(run_id, area_id, pass_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_model
            ON predictions(run_id, model_id);
        CREATE INDEX IF NOT EXISTS idx_aggregates_lookup
            ON area_timestep_aggregates(run_id, model_id, pass_id, area_id);
        """
    )
    conn.commit()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run(
    conn: sqlite3.Connection,
    *,
    name: str,
    start_date: str,
    end_date: str,
    max_timesteps: int,
    points_per_location: int,
    seed: int,
    size_km: float,
    metadata: dict[str, object] | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO simulation_runs (
            name, start_date, end_date, max_timesteps, points_per_location,
            seed, size_km, created_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            start_date,
            end_date,
            max_timesteps,
            points_per_location,
            seed,
            size_km,
            now_utc(),
            json.dumps(metadata or {}, sort_keys=True),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_passes(
    conn: sqlite3.Connection,
    run_id: int,
    timestamps: Iterable[str],
    *,
    source: str,
) -> list[int]:
    pass_ids: list[int] = []
    for index, timestamp in enumerate(timestamps):
        cursor = conn.execute(
            """
            INSERT INTO satellite_passes (run_id, timestep_index, timestamp, source)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, index, timestamp, source),
        )
        pass_ids.append(int(cursor.lastrowid))
    conn.commit()
    return pass_ids


def insert_checkpoint(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    area_id: str,
    area_name: str,
    point_id: str,
    point_index: int,
    lon: float,
    lat: float,
    source: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO checkpoints (
            run_id, area_id, area_name, point_id, point_index, lon, lat, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, area_id, area_name, point_id, point_index, lon, lat, source),
    )
    return int(cursor.lastrowid)


def insert_observation(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    pass_id: int,
    checkpoint_id: int,
    area_id: str,
    point_id: str,
    timestamp: str,
    lon: float,
    lat: float,
    size_km: float,
    rgb_path: str | None,
    swir_path: str | None,
    sentinel_metadata: dict[str, object] | None,
    status: str,
    error: str | None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO observations (
            run_id, pass_id, checkpoint_id, area_id, point_id, timestamp,
            lon, lat, size_km, rgb_path, swir_path, sentinel_metadata_json,
            status, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            pass_id,
            checkpoint_id,
            area_id,
            point_id,
            timestamp,
            lon,
            lat,
            size_km,
            rgb_path,
            swir_path,
            json.dumps(sentinel_metadata or {}, sort_keys=True),
            status,
            error,
            now_utc(),
        ),
    )
    return int(cursor.lastrowid)


def insert_prediction(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    observation_id: int,
    model_id: str,
    model_label: str,
    prediction: dict[str, object] | None,
    valid_json: bool,
    error: str | None,
    latency_s: float,
) -> int:
    prediction = prediction or {}
    values: dict[str, object | None] = {
        field: prediction.get(field)
        for field in LABEL_FIELDS
    }
    for field in BOOLEAN_FIELDS:
        if values[field] is not None:
            values[field] = int(bool(values[field]))

    cursor = conn.execute(
        """
        INSERT INTO predictions (
            run_id, observation_id, model_id, model_label, raw_json, valid_json,
            error, latency_s, flood_risk_level, water_extent_level,
            standing_water_present, temporary_inundation_likely,
            urban_or_infrastructure_exposure, road_or_transport_disruption_likely,
            cropland_or_settlement_exposure, river_or_coastal_overflow_context,
            low_lying_or_poor_drainage_area, vegetation_or_soil_saturation,
            permanent_water_body_present, cloud_shadow_or_image_quality_limited,
            confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            observation_id,
            model_id,
            model_label,
            json.dumps(prediction, sort_keys=True) if prediction else None,
            int(valid_json),
            error,
            latency_s,
            values["flood_risk_level"],
            values["water_extent_level"],
            values["standing_water_present"],
            values["temporary_inundation_likely"],
            values["urban_or_infrastructure_exposure"],
            values["road_or_transport_disruption_likely"],
            values["cropland_or_settlement_exposure"],
            values["river_or_coastal_overflow_context"],
            values["low_lying_or_poor_drainage_area"],
            values["vegetation_or_soil_saturation"],
            values["permanent_water_body_present"],
            values["cloud_shadow_or_image_quality_limited"],
            values["confidence"],
            now_utc(),
        ),
    )
    return int(cursor.lastrowid)


def average_level(score: float | None, labels: tuple[str, ...]) -> str | None:
    if score is None:
        return None
    if len(labels) != 3:
        index = min(len(labels) - 1, max(0, round(score) - 1))
        return labels[index]
    if score < 1.5:
        return labels[0]
    if score < 2.5:
        return labels[1]
    return labels[2]


def severity_level(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def aggregate_prediction_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if row.get("valid_json")]
    risk_scores = [
        LEVEL_SCORES[str(row["flood_risk_level"])]
        for row in valid
        if row.get("flood_risk_level") in LEVEL_SCORES
    ]
    water_scores = [
        WATER_SCORES[str(row["water_extent_level"])]
        for row in valid
        if row.get("water_extent_level") in WATER_SCORES
    ]
    confidence_scores = [
        LEVEL_SCORES[str(row["confidence"])]
        for row in valid
        if row.get("confidence") in LEVEL_SCORES
    ]

    severity_votes: list[float] = []
    for row in valid:
        votes = [row.get(field) for field in SEVERITY_FIELDS if row.get(field) is not None]
        if votes:
            severity_votes.append(sum(1 for vote in votes if bool(vote)) / len(votes))

    risk_score = _mean(risk_scores)
    water_score = _mean(water_scores)
    confidence_score = _mean(confidence_scores)
    severity_score = _mean(severity_votes)

    return {
        "observation_count": len(rows),
        "valid_prediction_count": len(valid),
        "flood_risk_score": risk_score,
        "flood_risk_level": average_level(risk_score, RISK_LEVELS),
        "water_extent_score": water_score,
        "water_extent_level": average_level(water_score, WATER_EXTENT_LEVELS),
        "severity_score": severity_score,
        "severity_level": severity_level(severity_score),
        "confidence_score": confidence_score,
        "confidence_level": average_level(confidence_score, CONFIDENCE_LEVELS),
    }


def refresh_aggregates(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute("DELETE FROM area_timestep_aggregates WHERE run_id=?", (run_id,))
    groups = conn.execute(
        """
        SELECT DISTINCT o.pass_id, o.area_id, p.model_id
        FROM observations o
        JOIN predictions p ON p.observation_id = o.id
        WHERE o.run_id=?
        ORDER BY o.pass_id, o.area_id, p.model_id
        """,
        (run_id,),
    ).fetchall()
    for group in groups:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT p.*
                FROM predictions p
                JOIN observations o ON o.id = p.observation_id
                WHERE o.run_id=? AND o.pass_id=? AND o.area_id=? AND p.model_id=?
                """,
                (run_id, group["pass_id"], group["area_id"], group["model_id"]),
            ).fetchall()
        ]
        agg = aggregate_prediction_rows(rows)
        conn.execute(
            """
            INSERT INTO area_timestep_aggregates (
                run_id, pass_id, area_id, model_id, observation_count,
                valid_prediction_count, flood_risk_score, flood_risk_level,
                water_extent_score, water_extent_level, severity_score,
                severity_level, confidence_score, confidence_level, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                group["pass_id"],
                group["area_id"],
                group["model_id"],
                agg["observation_count"],
                agg["valid_prediction_count"],
                agg["flood_risk_score"],
                agg["flood_risk_level"],
                agg["water_extent_score"],
                agg["water_extent_level"],
                agg["severity_score"],
                agg["severity_level"],
                agg["confidence_score"],
                agg["confidence_level"],
                now_utc(),
            ),
        )
    conn.commit()


def latest_run_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT id FROM simulation_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return int(row["id"]) if row else None


def list_runs(conn: sqlite3.Connection) -> list[dict[str, object]]:
    return [dict(row) for row in conn.execute("SELECT * FROM simulation_runs ORDER BY created_at DESC")]


def dashboard_state(conn: sqlite3.Connection, run_id: int) -> dict[str, object]:
    run = conn.execute("SELECT * FROM simulation_runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(f"run not found: {run_id}")

    passes = [
        dict(row)
        for row in conn.execute(
            "SELECT id, timestep_index, timestamp, source FROM satellite_passes WHERE run_id=? ORDER BY timestep_index",
            (run_id,),
        )
    ]
    models = [
        dict(row)
        for row in conn.execute(
            """
            SELECT DISTINCT model_id, model_label
            FROM predictions
            WHERE run_id=?
            ORDER BY model_id
            """,
            (run_id,),
        )
    ]
    if not models:
        models = [
            {"model_id": spec.id, "model_label": spec.label}
            for spec in DEFAULT_MODELS.values()
        ]

    areas = [
        {
            "area_id": loc.id,
            "area_name": loc.name,
            "fallback_lon": loc.fallback_lon,
            "fallback_lat": loc.fallback_lat,
        }
        for loc in LOCATIONS
    ]
    aggregates = [
        dict(row)
        for row in conn.execute(
            """
            SELECT a.*, p.timestep_index, p.timestamp
            FROM area_timestep_aggregates a
            JOIN satellite_passes p ON p.id = a.pass_id
            WHERE a.run_id=?
            ORDER BY p.timestep_index, a.area_id, a.model_id
            """,
            (run_id,),
        )
    ]
    return {
        "run": dict(run),
        "passes": passes,
        "models": models,
        "areas": areas,
        "aggregates": aggregates,
        "overlays": [
            {"id": "flood_risk", "label": "Flood risk"},
            {"id": "water_extent", "label": "Water extent"},
            {"id": "severity", "label": "Severity risk"},
            {"id": "confidence", "label": "Inspection confidence"},
        ],
    }


def area_observations(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    area_id: str,
    pass_id: int,
) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
            o.id,
            o.point_id,
            o.timestamp,
            o.lon,
            o.lat,
            o.rgb_path,
            o.swir_path,
            o.status,
            o.error,
            c.point_index
        FROM observations o
        JOIN checkpoints c ON c.id = o.checkpoint_id
        WHERE o.run_id=? AND o.area_id=? AND o.pass_id=?
        ORDER BY c.point_index
        """,
        (run_id, area_id, pass_id),
    ).fetchall()
    return [dict(row) for row in rows]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
