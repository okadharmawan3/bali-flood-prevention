import json
import sys
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bali_flood_prevention.locations import LOCATIONS
from bali_flood_prevention.quality import pair_quality
from bali_flood_prevention.schema import empty_label, validate_label
from bali_flood_prevention.tiles import spatial_grid, temporal_timestamps, train_test_cutoff


class CoreTests(unittest.TestCase):
    def test_full_study_default_task_counts(self) -> None:
        start = datetime.fromisoformat("2024-01-01").replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat("2025-12-31").replace(tzinfo=timezone.utc)
        timestamps = temporal_timestamps(start, end, 48)
        cutoff = train_test_cutoff(start, end, 0.2)
        train_temporal = [ts for ts in timestamps if datetime.fromisoformat(ts) < cutoff]
        test_temporal = [ts for ts in timestamps if datetime.fromisoformat(ts) >= cutoff]

        points = len(LOCATIONS) * 10
        spatial = 4
        self.assertEqual(points * len(timestamps) * spatial, 17280)
        self.assertEqual(points * len(train_temporal) * spatial, 13680)
        self.assertEqual(points * len(test_temporal) * spatial, 3600)

    def test_spatial_grid_count(self) -> None:
        tiles = spatial_grid(115.2167, -8.65, 4, 5.0)
        self.assertEqual(len(tiles), 4)
        self.assertEqual([tile.index for tile in tiles], [0, 1, 2, 3])

    def test_schema_validates_exact_clean_json(self) -> None:
        label = empty_label()
        validated = validate_label(label)
        self.assertEqual(list(validated), list(label))
        encoded = json.dumps(validated)
        self.assertEqual(validate_label(json.loads(encoded)), validated)

    def test_schema_rejects_extra_fields(self) -> None:
        label = empty_label()
        label["caption"] = "not allowed"
        with self.assertRaises(ValueError):
            validate_label(label)

    def test_blank_image_quality_detection(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rgb_path = tmp_dir / "rgb.png"
            swir_path = tmp_dir / "swir.png"
            Image.new("RGB", (10, 10), (0, 0, 0)).save(rgb_path)
            Image.new("RGB", (10, 10), (0, 255, 0)).save(swir_path)
            quality = pair_quality(rgb_path, swir_path)
            self.assertEqual(quality.rgb.blank_fraction, 1.0)
            self.assertEqual(quality.swir.blank_fraction, 0.0)
            self.assertEqual(quality.joint_blank_fraction, 0.0)
            self.assertFalse(quality.is_bad(0.5))

    def test_joint_blank_image_quality_detection(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rgb_path = tmp_dir / "rgb.png"
            swir_path = tmp_dir / "swir.png"
            Image.new("RGB", (10, 10), (0, 0, 0)).save(rgb_path)
            Image.new("RGB", (10, 10), (0, 0, 0)).save(swir_path)
            quality = pair_quality(rgb_path, swir_path)
            self.assertEqual(quality.joint_blank_fraction, 1.0)
            self.assertTrue(quality.is_bad(0.5))


if __name__ == "__main__":
    unittest.main()
