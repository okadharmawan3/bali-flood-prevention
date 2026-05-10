"""Deterministic point generation for Bali administrative locations."""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from bali_flood_prevention.locations import LOCATIONS, Location

GEOBOUNDARIES_API_URL = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/"


@dataclass(frozen=True)
class SamplePoint:
    location_id: str
    location_name: str
    point_index: int
    point_id: str
    lon: float
    lat: float
    source: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def generate_points(
    points_per_location: int,
    seed: int,
    cache_dir: Path,
    *,
    refresh_boundaries: bool = False,
    allow_bbox_fallback: bool = True,
) -> list[SamplePoint]:
    """Generate deterministic random points for every Bali location."""
    if points_per_location < 1:
        raise ValueError("points_per_location must be >= 1")

    geojson = load_boundary_geojson(cache_dir, refresh=refresh_boundaries)
    points: list[SamplePoint] = []
    for loc in LOCATIONS:
        feature = find_feature_for_location(geojson, loc) if geojson else None
        if feature is not None:
            polygons = geometry_polygons(feature.get("geometry"))
            loc_points = _sample_polygon_points(loc, polygons, points_per_location, seed)
        elif allow_bbox_fallback:
            loc_points = _sample_bbox_points(loc, points_per_location, seed)
        else:
            raise RuntimeError(f"No boundary feature found for {loc.id}")
        points.extend(loc_points)
    return points


def write_points_manifest(points: Iterable[SamplePoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(point.to_json(), sort_keys=True) for point in points]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def read_points_manifest(path: Path) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        points.append(
            SamplePoint(
                location_id=str(data["location_id"]),
                location_name=str(data["location_name"]),
                point_index=int(data["point_index"]),
                point_id=str(data["point_id"]),
                lon=float(data["lon"]),
                lat=float(data["lat"]),
                source=str(data["source"]),
            )
        )
    return points


def load_boundary_geojson(cache_dir: Path, *, refresh: bool = False) -> dict[str, Any] | None:
    """Load cached geoBoundaries ADM2 GeoJSON, downloading it when needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = cache_dir / "geoboundaries_idn_adm2.geojson"
    if boundary_path.exists() and not refresh:
        return json.loads(boundary_path.read_text(encoding="utf-8"))

    try:
        meta_resp = requests.get(GEOBOUNDARIES_API_URL, timeout=30)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        download_url = meta["gjDownloadURL"]
        geo_resp = requests.get(download_url, timeout=120)
        geo_resp.raise_for_status()
    except Exception:
        if boundary_path.exists():
            return json.loads(boundary_path.read_text(encoding="utf-8"))
        return None

    boundary_path.write_text(geo_resp.text, encoding="utf-8")
    return geo_resp.json()


def find_feature_for_location(geojson: dict[str, Any], loc: Location) -> dict[str, Any] | None:
    aliases = {_normalize_name(alias) for alias in loc.boundary_aliases}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        values = {
            _normalize_name(str(value))
            for value in props.values()
            if isinstance(value, str) and value.strip()
        }
        if aliases & values:
            return feature
        if any(alias in value or value in alias for alias in aliases for value in values):
            return feature
    return None


def geometry_polygons(geometry: Any) -> list[list[list[tuple[float, float]]]]:
    """Return a list of polygons, each as a list of rings of (lon, lat)."""
    if not isinstance(geometry, dict):
        return []
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Polygon":
        return [_rings(coords)]
    if geom_type == "MultiPolygon":
        return [_rings(poly) for poly in coords]
    return []


def contains_point(polygons: list[list[list[tuple[float, float]]]], lon: float, lat: float) -> bool:
    for polygon in polygons:
        if not polygon:
            continue
        outer = polygon[0]
        holes = polygon[1:]
        if _point_in_ring(lon, lat, outer) and not any(
            _point_in_ring(lon, lat, hole) for hole in holes
        ):
            return True
    return False


def _sample_polygon_points(
    loc: Location,
    polygons: list[list[list[tuple[float, float]]]],
    count: int,
    seed: int,
) -> list[SamplePoint]:
    if not polygons:
        return _sample_bbox_points(loc, count, seed)
    min_lon, min_lat, max_lon, max_lat = _polygons_bbox(polygons)
    rng = random.Random(f"{seed}:{loc.id}:boundary")
    points: list[SamplePoint] = []
    attempts = 0
    while len(points) < count and attempts < 100000:
        attempts += 1
        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)
        if contains_point(polygons, lon, lat):
            idx = len(points)
            points.append(
                SamplePoint(
                    location_id=loc.id,
                    location_name=loc.name,
                    point_index=idx,
                    point_id=f"p{idx:02d}",
                    lon=round(lon, 6),
                    lat=round(lat, 6),
                    source="geoboundaries_adm2",
                )
            )
    if len(points) != count:
        raise RuntimeError(f"Could only generate {len(points)} points for {loc.id}")
    return points


def _sample_bbox_points(loc: Location, count: int, seed: int) -> list[SamplePoint]:
    min_lon, min_lat, max_lon, max_lat = loc.fallback_bbox
    rng = random.Random(f"{seed}:{loc.id}:bbox")
    points: list[SamplePoint] = []
    for idx in range(count):
        points.append(
            SamplePoint(
                location_id=loc.id,
                location_name=loc.name,
                point_index=idx,
                point_id=f"p{idx:02d}",
                lon=round(rng.uniform(min_lon, max_lon), 6),
                lat=round(rng.uniform(min_lat, max_lat), 6),
                source="fallback_bbox",
            )
        )
    return points


def _rings(coords: Any) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    for ring in coords or []:
        rings.append([(float(x), float(y)) for x, y, *_ in ring])
    return rings


def _point_in_ring(lon: float, lat: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        intersects = (yi > lat) != (yj > lat) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _polygons_bbox(polygons: list[list[list[tuple[float, float]]]]) -> tuple[float, float, float, float]:
    coords = [point for polygon in polygons for ring in polygon for point in ring]
    min_lon = min(lon for lon, _ in coords)
    max_lon = max(lon for lon, _ in coords)
    min_lat = min(lat for _, lat in coords)
    max_lat = max(lat for _, lat in coords)
    return min_lon, min_lat, max_lon, max_lat


def _normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(kabupaten|kota|regency|city|province|provinsi)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())
