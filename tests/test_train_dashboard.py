import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = ROOT / "app" / "train_dashboard.py"

spec = importlib.util.spec_from_file_location("train_dashboard", DASHBOARD_PATH)
assert spec is not None
train_dashboard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(train_dashboard)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TrainDashboardTests(unittest.TestCase):
    def test_discovers_lfm_checkpoint_with_trainer_state(self) -> None:
        with TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            run = (
                outputs
                / "modal-checkpoints"
                / "lfm2.5-VL-450M-vlm_sft-bali_flood-all-lr2em05-w0p0-no_lora-e2s256-20260505_105831"
            )
            state = {
                "global_step": 256,
                "max_steps": 256,
                "epoch": 2.98,
                "log_history": [
                    {
                        "step": 1,
                        "epoch": 0.01,
                        "loss": 2.2,
                        "grad_norm": 0.3,
                        "learning_rate": 0.0,
                        "lr/language_model": 2.5e-6,
                    },
                    {
                        "step": 256,
                        "epoch": 2.98,
                        "eval_loss": 0.01,
                        "eval_runtime": 17.4,
                    },
                ],
            }
            write_json(run / "trainer_state.json", state)
            (run / "model.safetensors").write_bytes(b"weights")

            runs = train_dashboard.discover_runs(outputs)

            self.assertEqual(len(runs), 1)
            overview = runs[0]["overview"]
            self.assertEqual(overview["model"], "LiquidAI/lfm2.5-VL-450M")
            self.assertEqual(overview["source_type"], "leap-finetune")
            self.assertEqual(overview["status"], "completed")
            self.assertEqual(overview["checkpoint_count"], 1)
            self.assertEqual(list(runs[0]["history"]["step"]), [1, 256])

    def test_discovers_smolvlm_transformers_run_with_final_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            run = outputs / "SmolVLM2-500M-Video-Instruct-transformers-bali_flood-20260506_115928"
            write_json(
                run / "train_meta.json",
                {
                    "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    "training_config": {"max_steps": 321},
                },
            )
            write_json(
                run / "checkpoint-25" / "trainer_state.json",
                {
                    "global_step": 25,
                    "max_steps": 321,
                    "epoch": 0.23,
                    "log_history": [{"step": 25, "loss": 0.08}],
                },
            )
            final = run / "final-global_step321"
            final.mkdir(parents=True)
            (final / "model.safetensors").write_bytes(b"weights")

            runs = train_dashboard.discover_runs(outputs)

            self.assertEqual(len(runs), 1)
            overview = runs[0]["overview"]
            self.assertEqual(overview["model"], "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            self.assertEqual(overview["source_type"], "hf-transformers")
            self.assertEqual(overview["status"], "completed")
            self.assertEqual(overview["final_step"], 321)
            self.assertEqual(overview["checkpoint_count"], 2)


if __name__ == "__main__":
    unittest.main()
