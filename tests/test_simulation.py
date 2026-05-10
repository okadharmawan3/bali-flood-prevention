import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bali_flood_prevention.locations import LOCATIONS  # noqa: E402
from bali_flood_prevention.points import find_feature_for_location, generate_points  # noqa: E402
from bali_flood_prevention.schema import empty_label  # noqa: E402
from bali_flood_prevention.simulation import (  # noqa: E402
    SEVERITY_FIELDS,
    aggregate_prediction_rows,
    area_observations,
    connect_db,
    create_run,
    dashboard_state,
    insert_checkpoint,
    insert_observation,
    insert_passes,
    insert_prediction,
    refresh_aggregates,
)
from scripts import build_simulation_db  # noqa: E402


def _boundary_feature(name: str, bbox: tuple[float, float, float, float]) -> dict[str, object]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "Feature",
        "properties": {"shapeName": name},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        },
    }


def _write_test_boundaries(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = [
        _boundary_feature(loc.boundary_aliases[0], loc.fallback_bbox)
        for loc in LOCATIONS
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


class SimulationTests(unittest.TestCase):
    def test_point_generation_uses_nine_areas_and_ten_points(self) -> None:
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _write_test_boundaries(cache / "geoboundaries_idn_adm2.geojson")

            points_a = generate_points(10, 42, cache)
            points_b = generate_points(10, 42, cache)

            self.assertEqual(len(points_a), 9 * 10)
            self.assertEqual(points_a, points_b)
            self.assertEqual({point.location_id for point in points_a}, {loc.id for loc in LOCATIONS})

    def test_karangasem_boundary_alias_matches_geoboundaries_name(self) -> None:
        geojson = {
            "type": "FeatureCollection",
            "features": [_boundary_feature("Karang Asem", (115.4, -8.6, 115.73, -8.1))],
        }
        loc = next(item for item in LOCATIONS if item.id == "karangasem_bali")

        feature = find_feature_for_location(geojson, loc)

        self.assertIsNotNone(feature)

    def test_pass_discovery_deduplicates_and_caps(self) -> None:
        calls: list[str] = []
        values = [
            "2026-01-01T03:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-06T03:00:00Z",
            "2026-01-11T03:00:00Z",
            "2026-01-16T03:00:00Z",
        ]

        original = build_simulation_db.fetch_rgb_with_metadata

        def fake_fetch(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(str(args[2]))
            return b"png", {"datetime": values[min(len(calls) - 1, len(values) - 1)]}

        build_simulation_db.fetch_rgb_with_metadata = fake_fetch
        try:
            timestamps = build_simulation_db.discover_pass_timestamps(
                start_date=build_simulation_db.parse_date("2026-01-01"),
                end_date=build_simulation_db.parse_date("2026-01-31"),
                max_timesteps=3,
                lon=115.2,
                lat=-8.6,
                size_km=5.0,
                base_url="http://example.test",
            )
        finally:
            build_simulation_db.fetch_rgb_with_metadata = original

        self.assertEqual(
            timestamps,
            [
                "2026-01-01T03:00:00Z",
                "2026-01-06T03:00:00Z",
                "2026-01-11T03:00:00Z",
            ],
        )

    def test_aggregate_prediction_rows(self) -> None:
        high = empty_label()
        high["flood_risk_level"] = "high"
        high["water_extent_level"] = "large"
        high["confidence"] = "high"
        for field in SEVERITY_FIELDS:
            high[field] = True

        low = empty_label()
        low["flood_risk_level"] = "low"
        low["water_extent_level"] = "small"
        low["confidence"] = "low"
        for field in SEVERITY_FIELDS:
            low[field] = False

        rows = [
            {"valid_json": 1, **high},
            {"valid_json": 1, **low},
        ]
        aggregate = aggregate_prediction_rows(rows)

        self.assertEqual(aggregate["flood_risk_level"], "medium")
        self.assertEqual(aggregate["water_extent_level"], "moderate")
        self.assertEqual(aggregate["confidence_level"], "medium")
        self.assertEqual(aggregate["severity_level"], "medium")
        self.assertAlmostEqual(float(aggregate["severity_score"]), 0.5)

    def test_db_insert_fetch_and_aggregate(self) -> None:
        with TemporaryDirectory() as tmp:
            conn = connect_db(Path(tmp) / "sim.db")
            run_id, pass_id = _seed_tiny_db(conn, Path(tmp))

            refresh_aggregates(conn, run_id)
            state = dashboard_state(conn, run_id)
            rows = area_observations(conn, run_id=run_id, area_id="denpasar_bali", pass_id=pass_id)
            conn.close()

            self.assertEqual(state["run"]["id"], run_id)
            self.assertEqual(len(state["aggregates"]), 1)
            self.assertEqual(state["aggregates"][0]["flood_risk_level"], "medium")
            self.assertEqual(len(rows), 2)

    def test_api_smoke_paths(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from app import simulation_api
        except Exception as exc:  # pragma: no cover - makes missing optional deps obvious.
            self.fail(f"Cannot import simulation API: {exc}")

        api_image_root = ROOT / "simulation_images"
        api_image_root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=api_image_root) as tmp:
            tmp_path = Path(tmp)
            db = tmp_path / "api.db"
            conn = connect_db(db)
            run_id, pass_id = _seed_tiny_db(conn, tmp_path)
            refresh_aggregates(conn, run_id)
            conn.close()

            old_db = simulation_api.DEFAULT_DB_PATH
            simulation_api.DEFAULT_DB_PATH = db
            try:
                client = TestClient(simulation_api.app)
                state_response = client.get("/api/state")
                detail_response = client.get(
                    "/api/areas/denpasar_bali/observations",
                    params={"run_id": run_id, "pass_id": pass_id},
                )
                image_response = client.get("/api/observations/1/image/rgb")
            finally:
                simulation_api.DEFAULT_DB_PATH = old_db

            self.assertEqual(state_response.status_code, 200)
            self.assertEqual(detail_response.status_code, 200)
            self.assertEqual(len(detail_response.json()["observations"]), 2)
            self.assertEqual(image_response.status_code, 200)
            self.assertEqual(image_response.headers["content-type"], "image/png")


def _seed_tiny_db(conn, tmp_path: Path) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    run_id = create_run(
        conn,
        name="test-run",
        start_date="2026-01-01",
        end_date="2026-01-31",
        max_timesteps=1,
        points_per_location=2,
        seed=42,
        size_km=5.0,
    )
    pass_id = insert_passes(conn, run_id, ["2026-01-01T03:00:00Z"], source="test")[0]
    for index in range(2):
        sample_dir = tmp_path / f"sample_{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgb = sample_dir / "rgb.png"
        swir = sample_dir / "swir.png"
        Image.new("RGB", (4, 4), (10 + index, 20, 30)).save(rgb)
        Image.new("RGB", (4, 4), (30, 20, 10 + index)).save(swir)
        checkpoint_id = insert_checkpoint(
            conn,
            run_id=run_id,
            area_id="denpasar_bali",
            area_name="Denpasar, Indonesia",
            point_id=f"p{index:02d}",
            point_index=index,
            lon=115.2 + index / 100,
            lat=-8.6,
            source="test",
        )
        observation_id = insert_observation(
            conn,
            run_id=run_id,
            pass_id=pass_id,
            checkpoint_id=checkpoint_id,
            area_id="denpasar_bali",
            point_id=f"p{index:02d}",
            timestamp="2026-01-01T03:00:00Z",
            lon=115.2 + index / 100,
            lat=-8.6,
            size_km=5.0,
            rgb_path=str(rgb),
            swir_path=str(swir),
            sentinel_metadata={},
            status="ready",
            error=None,
        )
        label = empty_label()
        label["flood_risk_level"] = "high" if index == 0 else "low"
        label["water_extent_level"] = "large" if index == 0 else "small"
        label["confidence"] = "high" if index == 0 else "low"
        for field in SEVERITY_FIELDS:
            label[field] = index == 0
        insert_prediction(
            conn,
            run_id=run_id,
            observation_id=observation_id,
            model_id="lfm2",
            model_label="LFM2.5-VL Bali Flood",
            prediction=label,
            valid_json=True,
            error=None,
            latency_s=0.1,
        )
    conn.commit()
    return run_id, pass_id


if __name__ == "__main__":
    unittest.main()
