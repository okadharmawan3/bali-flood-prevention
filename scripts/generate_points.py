"""Generate deterministic random Bali sample points.

Usage:
    uv run scripts/generate_points.py --points-per-location 10 --seed 42
"""

import argparse
from pathlib import Path

from bali_flood_prevention.points import generate_points, write_points_manifest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bali sample point manifest.")
    parser.add_argument("--points-per-location", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-boundaries", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to data/points/bali_points_seed{seed}.jsonl.",
    )
    args = parser.parse_args()

    output = (
        Path(args.output)
        if args.output
        else DATA_DIR / "points" / f"bali_points_seed{args.seed}.jsonl"
    )
    points = generate_points(
        points_per_location=args.points_per_location,
        seed=args.seed,
        cache_dir=DATA_DIR / "boundaries",
        refresh_boundaries=args.refresh_boundaries,
    )
    write_points_manifest(points, output)

    print(f"Wrote {len(points)} points to {output}")
    by_source: dict[str, int] = {}
    for point in points:
        by_source[point.source] = by_source.get(point.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
