import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bali_flood_prevention.evaluator import (  # noqa: E402
    EVAL_FIELDS,
    EvalSample,
    EvalSummary,
    SampleResult,
    evaluate_sample,
    load_local_samples,
)
from bali_flood_prevention.schema import empty_label  # noqa: E402


def _write_sample(root: Path, split: str, region: str, key: str, label: dict[str, object]) -> None:
    sample_dir = root / split / region / key
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(sample_dir / "rgb.png")
    Image.new("RGB", (4, 4), (30, 20, 10)).save(sample_dir / "swir.png")
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": f"{region}/{key}",
                "region": region,
                "timestamp": "2025-08-16T03:00:00+00:00",
                "point_id": "p00",
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "annotation.json").write_text(json.dumps(label), encoding="utf-8")


class EvaluatorTests(unittest.TestCase):
    def test_load_local_samples_uses_requested_test_split(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            label = empty_label()
            _write_sample(run_dir, "train", "denpasar_bali", "p00_s00_t00", label)
            _write_sample(run_dir, "test", "denpasar_bali", "p00_s00_t19", label)

            samples = load_local_samples(run_dir, "test")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].id, "denpasar_bali/p00_s00_t19")
            self.assertEqual(samples[0].split, "test")

    def test_evaluate_sample_matches_all_schema_fields(self) -> None:
        sample = self._sample(empty_label())

        result = evaluate_sample(sample, lambda _sample: dict(sample.ground_truth))

        self.assertTrue(result.valid_json)
        self.assertTrue(result.fields_present)
        self.assertEqual(set(result.field_matches), set(EVAL_FIELDS))
        self.assertTrue(all(result.field_matches.values()))

    def test_evaluate_sample_marks_missing_fields_without_exception(self) -> None:
        sample = self._sample(empty_label())

        result = evaluate_sample(sample, lambda _sample: {"flood_risk_level": "low"})

        self.assertTrue(result.valid_json)
        self.assertFalse(result.fields_present)
        self.assertFalse(any(result.field_matches.values()))

    def test_evaluate_sample_captures_backend_errors(self) -> None:
        sample = self._sample(empty_label())

        def fail(_sample: EvalSample) -> dict[str, object]:
            raise ValueError("bad json")

        result = evaluate_sample(sample, fail)

        self.assertFalse(result.valid_json)
        self.assertFalse(result.fields_present)
        self.assertEqual(result.error, "bad json")

    def test_risk_metrics_are_macro_balanced_for_imbalanced_data(self) -> None:
        low_truth = empty_label()
        low_truth["flood_risk_level"] = "low"
        medium_truth = empty_label()
        medium_truth["flood_risk_level"] = "medium"
        high_truth = empty_label()
        high_truth["flood_risk_level"] = "high"

        results = [
            self._result(low_truth, {"flood_risk_level": "low"}),
            self._result(medium_truth, {"flood_risk_level": "low"}),
            self._result(high_truth, None),
        ]
        risk = EvalSummary(results).risk_metrics()

        self.assertEqual(risk.confusion["low"]["low"], 1)
        self.assertEqual(risk.confusion["medium"]["low"], 1)
        self.assertEqual(risk.invalid_or_missing, 1)
        self.assertAlmostEqual(risk.macro_precision, 1 / 6)
        self.assertAlmostEqual(risk.macro_recall, 1 / 3)
        self.assertAlmostEqual(risk.balanced_accuracy, 1 / 3)
        self.assertAlmostEqual(risk.macro_f1, 2 / 9)

    def _sample(self, label: dict[str, object]) -> EvalSample:
        return EvalSample(
            id="denpasar_bali/p00_s00_t19",
            split="test",
            region="denpasar_bali",
            timestamp="2025-08-16T03:00:00+00:00",
            rgb_path=Path("rgb.png"),
            swir_path=Path("swir.png"),
            metadata_path=Path("metadata.json"),
            annotation_path=Path("annotation.json"),
            rgb_bytes=b"rgb",
            swir_bytes=b"swir",
            metadata={},
            ground_truth=label,
        )

    def _result(
        self,
        truth: dict[str, object],
        prediction: dict[str, object] | None,
    ) -> SampleResult:
        return SampleResult(
            id="sample",
            region="region",
            timestamp="timestamp",
            valid_json=prediction is not None,
            fields_present=prediction is not None,
            field_matches={field: False for field in EVAL_FIELDS},
            latency_s=0.0,
            prediction=prediction,
            ground_truth=truth,
        )


if __name__ == "__main__":
    unittest.main()
