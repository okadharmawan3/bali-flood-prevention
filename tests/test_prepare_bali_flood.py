import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bali_flood_prevention.schema import SYSTEM_PROMPT, USER_TEXT, empty_label, validate_label
from scripts.prepare_bali_flood import make_vlm_row, prepare_dataset


class PrepareBaliFloodTests(unittest.TestCase):
    def test_make_vlm_row_uses_two_images_and_bali_prompt(self) -> None:
        output = json.dumps(empty_label())

        row = make_vlm_row("sample_rgb.png", "sample_swir.png", output)

        messages = row["messages"]
        self.assertEqual(len(messages), 2)
        user_content = messages[0]["content"]
        self.assertEqual(user_content[0], {"type": "image", "image": "sample_rgb.png"})
        self.assertEqual(user_content[1], {"type": "image", "image": "sample_swir.png"})
        self.assertEqual(user_content[2]["type"], "text")
        self.assertIn(SYSTEM_PROMPT.strip().splitlines()[0], user_content[2]["text"])
        self.assertIn(USER_TEXT, user_content[2]["text"])

        assistant_text = messages[1]["content"][0]["text"]
        validate_label(json.loads(assistant_text))

    def test_prepare_dataset_writes_leap_jsonl_and_copies_images(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "source_hf_dataset"
            output = tmp_dir / "prepared"
            images = source / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (4, 4), (10, 20, 30)).save(images / "a_rgb.png")
            Image.new("RGB", (4, 4), (30, 20, 10)).save(images / "a_swir.png")
            row = {
                "region": "denpasar_bali",
                "point_id": "p00",
                "timestamp": "2025-08-16T03:00:00+00:00",
                "split": "train",
                "rgb_path": "images/a_rgb.png",
                "swir_path": "images/a_swir.png",
                "output": json.dumps(empty_label()),
            }
            (source / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (source / "test.jsonl").write_text(json.dumps({**row, "split": "test"}) + "\n", encoding="utf-8")

            counts = prepare_dataset(str(source), output)

            self.assertEqual(counts, {"train": 1, "test": 1})
            self.assertTrue((output / "images" / "a_rgb.png").is_file())
            train_rows = [
                json.loads(line)
                for line in (output / "bali_flood_train.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(train_rows), 1)
            user_content = train_rows[0]["messages"][0]["content"]
            self.assertEqual(user_content[0]["image"], "a_rgb.png")
            self.assertEqual(user_content[1]["image"], "a_swir.png")


if __name__ == "__main__":
    unittest.main()
