"""Bali administrative locations used for dataset generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    boundary_aliases: tuple[str, ...]
    fallback_lon: float
    fallback_lat: float
    fallback_bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat


LOCATIONS: tuple[Location, ...] = (
    Location(
        "denpasar_bali",
        "Denpasar, Indonesia",
        ("Denpasar", "Kota Denpasar"),
        115.2167,
        -8.6500,
        (115.1700, -8.7300, 115.2900, -8.5700),
    ),
    Location(
        "badung_bali",
        "Badung, Indonesia",
        ("Badung", "Kabupaten Badung"),
        115.1771,
        -8.5819,
        (115.0400, -8.8500, 115.3000, -8.2700),
    ),
    Location(
        "bangli_bali",
        "Bangli, Indonesia",
        ("Bangli", "Kabupaten Bangli"),
        115.3549,
        -8.4542,
        (115.2500, -8.5700, 115.4800, -8.1200),
    ),
    Location(
        "buleleng_bali",
        "Buleleng, Indonesia",
        ("Buleleng", "Kabupaten Buleleng"),
        114.9517,
        -8.1152,
        (114.4300, -8.3600, 115.4500, -8.0300),
    ),
    Location(
        "gianyar_bali",
        "Gianyar, Indonesia",
        ("Gianyar", "Kabupaten Gianyar"),
        115.3250,
        -8.5442,
        (115.2100, -8.6700, 115.4100, -8.3800),
    ),
    Location(
        "jembrana_bali",
        "Jembrana, Indonesia",
        ("Jembrana", "Kabupaten Jembrana"),
        114.6668,
        -8.3561,
        (114.4200, -8.5600, 114.9600, -8.1300),
    ),
    Location(
        "karangasem_bali",
        "Karangasem, Indonesia",
        ("Karangasem", "Karang Asem", "Kabupaten Karangasem", "Kabupaten Karang Asem"),
        115.6170,
        -8.3891,
        (115.4000, -8.6000, 115.7300, -8.1000),
    ),
    Location(
        "klungkung_bali",
        "Klungkung, Indonesia",
        ("Klungkung", "Kabupaten Klungkung"),
        115.4045,
        -8.5389,
        (115.3300, -8.8300, 115.7500, -8.4300),
    ),
    Location(
        "tabanan_bali",
        "Tabanan, Indonesia",
        ("Tabanan", "Kabupaten Tabanan"),
        115.1252,
        -8.5445,
        (114.8200, -8.7200, 115.2600, -8.2200),
    ),
)

LOCATIONS_BY_ID = {loc.id: loc for loc in LOCATIONS}
