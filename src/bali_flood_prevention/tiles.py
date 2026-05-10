"""Deterministic spatial and temporal tile generation."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TileCoord:
    index: int
    lon: float
    lat: float


def spatial_grid(
    center_lon: float,
    center_lat: float,
    n_tiles: int,
    size_km: float,
) -> list[TileCoord]:
    """Return tile centers arranged in a centered square grid."""
    if n_tiles < 1:
        raise ValueError(f"n_tiles must be >= 1, got {n_tiles}")

    grid_size = math.ceil(math.sqrt(n_tiles))
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))
    if abs(km_per_deg_lon) < 1e-9:
        raise ValueError("cannot build longitude offsets near the poles")

    coords: list[TileCoord] = []
    for i in range(n_tiles):
        row = i // grid_size
        col = i % grid_size
        row_offset = row - (grid_size - 1) / 2.0
        col_offset = col - (grid_size - 1) / 2.0
        coords.append(
            TileCoord(
                index=i,
                lon=round(center_lon + col_offset * size_km / km_per_deg_lon, 6),
                lat=round(center_lat + row_offset * size_km / km_per_deg_lat, 6),
            )
        )
    return coords


def temporal_timestamps(start_date: datetime, end_date: datetime, n: int) -> list[str]:
    """Return n evenly spaced ISO timestamps inside [start_date, end_date]."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    duration = end_date - start_date
    bin_width = duration / n
    timestamps: list[str] = []
    for i in range(n):
        ts = start_date + bin_width * i + bin_width / 2
        timestamps.append(_format_iso(ts))
    return timestamps


def train_test_cutoff(start_date: datetime, end_date: datetime, test_ratio: float) -> datetime:
    """Return the timestamp cutoff for a temporal train/test split."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    return start_date + (end_date - start_date) * (1.0 - test_ratio)


def _format_iso(dt: datetime) -> str:
    truncated = dt - timedelta(microseconds=dt.microsecond)
    return truncated.isoformat()
