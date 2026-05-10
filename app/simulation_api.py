"""API for the Bali flood realtime-style simulation dashboard.

Run from the project root:
    uv run uvicorn app.simulation_api:app --reload --port 8010
"""

from __future__ import annotations

import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bali_flood_prevention.locations import LOCATIONS, Location  # noqa: E402
from bali_flood_prevention.points import find_feature_for_location  # noqa: E402
from bali_flood_prevention.simulation import (  # noqa: E402
    DEFAULT_DB_PATH,
    area_observations,
    connect_db,
    dashboard_state,
    latest_run_id,
    list_runs,
)

SIMSAT_DASHBOARD_URL = "http://localhost:8000"
SIMSAT_API_URL = "http://localhost:9005"
BUNDLED_BOUNDARY_PATH = ROOT / "assets" / "bali_adm2.geojson"
CACHED_BOUNDARY_PATH = ROOT / "data" / "boundaries" / "geoboundaries_idn_adm2.geojson"

app = FastAPI(title="Bali Flood Simulation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def db_path() -> Path:
    return Path(DEFAULT_DB_PATH)


def resolve_run_id(run_id: int | None) -> int:
    conn = connect_db(db_path())
    try:
        resolved = run_id or latest_run_id(conn)
    finally:
        conn.close()
    if resolved is None:
        raise HTTPException(status_code=404, detail="No simulation runs found")
    return resolved


def read_boundaries() -> dict[str, Any]:
    boundary_path = (
        BUNDLED_BOUNDARY_PATH
        if BUNDLED_BOUNDARY_PATH.is_file()
        else CACHED_BOUNDARY_PATH
    )
    if not boundary_path.is_file():
        raise HTTPException(status_code=404, detail=f"Boundary file not found: {boundary_path}")
    data = json.loads(boundary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Boundary file is not a GeoJSON object")
    return data


def feature_for_location(geojson: dict[str, Any], loc: Location) -> dict[str, Any] | None:
    feature = find_feature_for_location(geojson, loc)
    if feature is None:
        return None
    copied = json.loads(json.dumps(feature))
    props = copied.setdefault("properties", {})
    props["area_id"] = loc.id
    props["area_name"] = loc.name
    return copied


@app.get("/api/runs")
def runs() -> dict[str, object]:
    conn = connect_db(db_path())
    try:
        return {"runs": list_runs(conn)}
    finally:
        conn.close()


@app.get("/api/state")
def state(run_id: int | None = Query(default=None)) -> dict[str, object]:
    resolved = resolve_run_id(run_id)
    conn = connect_db(db_path())
    try:
        return dashboard_state(conn, resolved)
    finally:
        conn.close()


@app.get("/api/geojson")
def geojson() -> dict[str, object]:
    source = read_boundaries()
    features = [
        feature
        for loc in LOCATIONS
        if (feature := feature_for_location(source, loc)) is not None
    ]
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/areas/{area_id}/observations")
def observations(
    area_id: str,
    pass_id: int,
    run_id: int | None = Query(default=None),
) -> dict[str, object]:
    resolved = resolve_run_id(run_id)
    conn = connect_db(db_path())
    try:
        rows = area_observations(conn, run_id=resolved, area_id=area_id, pass_id=pass_id)
    finally:
        conn.close()
    for row in rows:
        row["rgb_url"] = f"/api/observations/{row['id']}/image/rgb"
        row["swir_url"] = f"/api/observations/{row['id']}/image/swir"
    return {"observations": rows}


@app.get("/api/observations/{observation_id}/image/{kind}")
def observation_image(observation_id: int, kind: str) -> FileResponse:
    if kind not in {"rgb", "swir"}:
        raise HTTPException(status_code=404, detail="Image kind must be 'rgb' or 'swir'")

    column = "rgb_path" if kind == "rgb" else "swir_path"
    conn = connect_db(db_path())
    try:
        row = conn.execute(
            f"SELECT {column} AS image_path FROM observations WHERE id=?",
            (observation_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row["image_path"]:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    image_path = image_path.resolve()
    if ROOT.resolve() not in image_path.parents and image_path != ROOT.resolve():
        raise HTTPException(status_code=403, detail="Image path is outside project root")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image file missing on disk")

    media_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    return FileResponse(str(image_path), media_type=media_type)


@app.get("/api/simsat/telemetry")
def simsat_telemetry() -> dict[str, object]:
    telemetry: list[dict[str, object]] = []
    position: dict[str, object] | None = None
    errors: list[str] = []

    try:
        response = requests.get(f"{SIMSAT_DASHBOARD_URL}/api/telemetry/recent/", timeout=5)
        response.raise_for_status()
        payload = response.json()
        telemetry = payload.get("telemetry") or []
    except requests.RequestException as exc:
        errors.append(f"dashboard telemetry: {exc}")

    try:
        response = requests.get(f"{SIMSAT_API_URL}/data/current/position", timeout=5)
        response.raise_for_status()
        position = response.json()
    except requests.RequestException as exc:
        errors.append(f"sim api position: {exc}")

    return {"telemetry": telemetry, "position": position, "errors": errors}


@app.post("/api/simsat/commands")
def simsat_command(payload: dict[str, Any]) -> JSONResponse:
    command = payload.get("command")
    if command not in {"start", "pause", "stop", "set_start_time", "set_step_size", "set_replay_speed"}:
        raise HTTPException(status_code=400, detail="Invalid SimSat command")
    try:
        response = requests.post(
            f"{SIMSAT_DASHBOARD_URL}/api/commands/",
            json=payload,
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach SimSat dashboard: {exc}") from exc

    if response.status_code >= 400:
        return JSONResponse(status_code=response.status_code, content=response.json())
    return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "dashboard": "Bali flood simulation API"}
