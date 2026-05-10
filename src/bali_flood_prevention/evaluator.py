"""Evaluation backends and metrics for Bali flood-risk predictions."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from openai import OpenAI

from bali_flood_prevention.schema import (
    FLOOD_LABEL_JSON_SCHEMA,
    LABEL_FIELDS,
    OPENAI_LABELING_INSTRUCTIONS,
    RISK_LEVELS,
    validate_label,
)

EVAL_FIELDS: tuple[str, ...] = LABEL_FIELDS
LLAMA_MEDIA_MARKER = "<__media__>"


class PredictFn(Protocol):
    def __call__(self, sample: "EvalSample") -> dict[str, object]: ...


@dataclass(frozen=True)
class EvalSample:
    id: str
    split: str
    region: str
    timestamp: str
    rgb_path: Path
    swir_path: Path
    metadata_path: Path
    annotation_path: Path
    rgb_bytes: bytes
    swir_bytes: bytes
    metadata: dict[str, object]
    ground_truth: dict[str, object]


@dataclass
class SampleResult:
    id: str
    region: str
    timestamp: str
    valid_json: bool
    fields_present: bool
    field_matches: dict[str, bool]
    latency_s: float
    prediction: dict[str, object] | None
    ground_truth: dict[str, object]
    error: str | None = None
    attempts: int = 1
    rgb_path: str | None = None
    swir_path: str | None = None

    @property
    def all_fields_match(self) -> bool:
        return all(self.field_matches.values())


@dataclass(frozen=True)
class RiskMetrics:
    confusion: dict[str, dict[str, int]]
    invalid_or_missing: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    balanced_accuracy: float


def path_for_json(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_local_samples(dataset_dir: Path, split: str, limit: int | None = None) -> list[EvalSample]:
    """Load samples from data/{run_id}/{split}/{region}/{sample_key}."""
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split '{split}' not found in {dataset_dir}")

    samples: list[EvalSample] = []
    for region_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for sample_dir in sorted(path for path in region_dir.iterdir() if path.is_dir()):
            sample_id = f"{region_dir.name}/{sample_dir.name}"
            rgb_path = sample_dir / "rgb.png"
            swir_path = sample_dir / "swir.png"
            metadata_path = sample_dir / "metadata.json"
            annotation_path = sample_dir / "annotation.json"
            missing = [
                path.name
                for path in (rgb_path, swir_path, metadata_path, annotation_path)
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(f"{sample_id} missing: {', '.join(missing)}")

            metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata_raw, dict):
                raise ValueError(f"{metadata_path} must contain a JSON object")
            metadata = dict(metadata_raw)
            label_raw = json.loads(annotation_path.read_text(encoding="utf-8"))
            ground_truth = validate_label(label_raw)

            samples.append(
                EvalSample(
                    id=sample_id,
                    split=split,
                    region=region_dir.name,
                    timestamp=str(metadata.get("timestamp", "")),
                    rgb_path=rgb_path,
                    swir_path=swir_path,
                    metadata_path=metadata_path,
                    annotation_path=annotation_path,
                    rgb_bytes=rgb_path.read_bytes(),
                    swir_bytes=swir_path.read_bytes(),
                    metadata=metadata,
                    ground_truth=ground_truth,
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def encode_data_url(image_bytes: bytes) -> str:
    encoded = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_eval_user_text(sample: EvalSample) -> str:
    context = {
        "sample_id": sample.id,
        "split": sample.split,
        "region": sample.region,
        "metadata": sample.metadata,
    }
    return (
        "Evaluate this single Bali Sentinel-2 sample. Use the RGB image, SWIR "
        "image, and metadata together. Return only the Bali flood-risk JSON "
        "object matching the schema.\n\n"
        f"Sample context JSON:\n{json.dumps(context, indent=2, ensure_ascii=True)}"
    )


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def make_openai_backend(
    model: str,
    image_detail: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> PredictFn:
    client = OpenAI()

    def predict(sample: EvalSample) -> dict[str, object]:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": OPENAI_LABELING_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_eval_user_text(sample)},
                        {
                            "type": "input_image",
                            "image_url": encode_data_url(sample.rgb_bytes),
                            "detail": image_detail,
                        },
                        {
                            "type": "input_image",
                            "image_url": encode_data_url(sample.swir_bytes),
                            "detail": image_detail,
                        },
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "bali_flood_label",
                    "schema": FLOOD_LABEL_JSON_SCHEMA,
                    "strict": True,
                }
            },
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
        )
        raw = extract_output_text(response)
        if not raw:
            raise ValueError("OpenAI returned empty output text")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI output was valid JSON but not an object")
        return parsed

    return predict


def llama_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "BaliFloodRisk",
            "schema": FLOOD_LABEL_JSON_SCHEMA,
            "strict": True,
        },
    }


def make_llama_backend(model: str, port: int = 8080) -> PredictFn:
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")

    def predict(sample: EvalSample) -> dict[str, object]:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format=llama_response_format(),
            messages=[
                {"role": "system", "content": OPENAI_LABELING_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_data_url(sample.rgb_bytes)},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_data_url(sample.swir_bytes)},
                        },
                        {"type": "text", "text": build_eval_user_text(sample)},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("llama-server output was valid JSON but not an object")
        return parsed

    return predict


def start_llama_server(
    model: str,
    quant: str | None = None,
    port: int = 8080,
    verbose: bool = False,
    mmproj: str | None = None,
    chat_template: str | None = None,
    chat_template_file: str | None = None,
    skip_chat_parsing: bool = False,
) -> subprocess.Popen[bytes]:
    local_path = Path(model)
    if local_path.is_file():
        cmd = ["llama-server", "-m", str(local_path), "--jinja", "--port", str(port)]
    else:
        hf_repo = f"{model}:{quant}" if quant else model
        cmd = ["llama-server", "-hf", hf_repo, "--jinja", "--port", str(port)]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    if chat_template:
        cmd += ["--chat-template", chat_template]
    if chat_template_file:
        cmd += ["--chat-template-file", chat_template_file]
    if skip_chat_parsing:
        cmd += ["--skip-chat-parsing"]

    env = os.environ.copy()
    env.setdefault("LLAMA_MEDIA_MARKER", LLAMA_MEDIA_MARKER)

    kwargs: dict[str, object] = {"env": env}
    if not verbose:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.Popen(cmd, **kwargs)  # type: ignore[call-overload]


def wait_for_server(port: int = 8080, timeout: int = 180) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def evaluate_sample(sample: EvalSample, predict: PredictFn, attempts: int = 1) -> SampleResult:
    t0 = perf_counter()
    try:
        prediction = predict(sample)
    except Exception as exc:
        return SampleResult(
            id=sample.id,
            region=sample.region,
            timestamp=sample.timestamp,
            valid_json=False,
            fields_present=False,
            field_matches={field: False for field in EVAL_FIELDS},
            latency_s=perf_counter() - t0,
            prediction=None,
            ground_truth=sample.ground_truth,
            error=str(exc),
            attempts=attempts,
            rgb_path=path_for_json(sample.rgb_path, Path.cwd()),
            swir_path=path_for_json(sample.swir_path, Path.cwd()),
        )

    latency_s = perf_counter() - t0
    valid_json = isinstance(prediction, dict)
    fields_present = valid_json and all(field in prediction for field in EVAL_FIELDS)
    field_matches = {
        field: bool(fields_present and prediction.get(field) == sample.ground_truth.get(field))
        for field in EVAL_FIELDS
    }
    return SampleResult(
        id=sample.id,
        region=sample.region,
        timestamp=sample.timestamp,
        valid_json=valid_json,
        fields_present=fields_present,
        field_matches=field_matches,
        latency_s=latency_s,
        prediction=prediction if valid_json else None,
        ground_truth=sample.ground_truth,
        attempts=attempts,
        rgb_path=path_for_json(sample.rgb_path, Path.cwd()),
        swir_path=path_for_json(sample.swir_path, Path.cwd()),
    )


def evaluate_sample_with_retries(
    sample: EvalSample,
    predict: PredictFn,
    retries: int,
    retry_delay: float,
) -> SampleResult:
    last_result: SampleResult | None = None
    for attempt in range(1, retries + 2):
        result = evaluate_sample(sample, predict, attempts=attempt)
        if result.error is None or attempt == retries + 1:
            return result
        last_result = result
        time.sleep(retry_delay * attempt)
    assert last_result is not None
    return last_result


@dataclass
class EvalSummary:
    results: list[SampleResult]

    def valid_json_accuracy(self) -> float:
        return _mean(result.valid_json for result in self.results)

    def fields_present_accuracy(self) -> float:
        return _mean(result.fields_present for result in self.results)

    def field_accuracy(self, field: str) -> float:
        return _mean(result.field_matches.get(field, False) for result in self.results)

    def overall_accuracy(self) -> float:
        matches = [
            result.field_matches.get(field, False)
            for result in self.results
            for field in EVAL_FIELDS
        ]
        return _mean(matches)

    def avg_latency_s(self) -> float:
        return (
            sum(result.latency_s for result in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    def risk_metrics(self) -> RiskMetrics:
        return compute_risk_metrics(self.results)


def _mean(values: Any) -> float:
    materialized = [bool(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def compute_risk_metrics(results: list[SampleResult]) -> RiskMetrics:
    labels = list(RISK_LEVELS)
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    invalid_or_missing = 0

    for result in results:
        true_value = str(result.ground_truth.get("flood_risk_level", ""))
        pred_value = None
        if result.prediction and "flood_risk_level" in result.prediction:
            pred_value = str(result.prediction.get("flood_risk_level"))
        if true_value in labels and pred_value in labels:
            confusion[true_value][pred_value] += 1
        else:
            invalid_or_missing += 1

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        pred_total = sum(confusion[true][label] for true in labels)
        true_total = sum(confusion[label][pred] for pred in labels)
        true_total += sum(
            1
            for result in results
            if result.ground_truth.get("flood_risk_level") == label
            and not (
                result.prediction
                and result.prediction.get("flood_risk_level") in labels
            )
        )

        precision = tp / pred_total if pred_total else 0.0
        recall = tp / true_total if true_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return RiskMetrics(
        confusion=confusion,
        invalid_or_missing=invalid_or_missing,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        balanced_accuracy=macro_recall,
    )


def render_report(
    summary: EvalSummary,
    dataset: str,
    backend: str,
    model: str,
    split: str,
    eval_run_id: str,
) -> str:
    risk = summary.risk_metrics()
    lines: list[str] = []
    lines.append(f"# Bali Flood Risk Eval - {eval_run_id}")
    lines.append("")
    lines.append(f"**Dataset:** {dataset}  ")
    lines.append(f"**Split:** {split}  ")
    lines.append(f"**Backend:** {backend}  ")
    lines.append(f"**Model:** {model}")
    lines.append("")
    lines.append("## Accuracy summary")
    lines.append("")
    lines.append("| field | accuracy |")
    lines.append("|---|---:|")
    lines.append(f"| valid_json | {summary.valid_json_accuracy():.2f} |")
    lines.append(f"| fields_present | {summary.fields_present_accuracy():.2f} |")
    for field in EVAL_FIELDS:
        lines.append(f"| {field} | {summary.field_accuracy(field):.2f} |")
    lines.append(f"| **overall** | **{summary.overall_accuracy():.2f}** |")
    lines.append(f"| **avg latency (s)** | **{summary.avg_latency_s():.2f}** |")
    lines.append("")
    lines.append("## Flood risk macro metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| macro_precision | {risk.macro_precision:.2f} |")
    lines.append(f"| macro_recall | {risk.macro_recall:.2f} |")
    lines.append(f"| macro_f1 | {risk.macro_f1:.2f} |")
    lines.append(f"| balanced_accuracy | {risk.balanced_accuracy:.2f} |")
    lines.append(f"| invalid_or_missing_risk | {risk.invalid_or_missing} |")
    lines.append("")
    lines.append("## Flood risk confusion matrix")
    lines.append("")
    lines.append("| true \\ predicted | low | medium | high |")
    lines.append("|---|---:|---:|---:|")
    for true_label in RISK_LEVELS:
        row = risk.confusion[true_label]
        lines.append(
            f"| {true_label} | {row['low']} | {row['medium']} | {row['high']} |"
        )
    lines.append("")
    lines.append("## Per-sample results")
    lines.append("")
    lines.append(
        "| id | region | latency (s) | valid_json | fields_present | "
        "flood_risk_level | water_extent_level | confidence | overall_match |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|---|")
    for result in summary.results:
        fm = result.field_matches
        lines.append(
            f"| {result.id} | {result.region} | {result.latency_s:.2f} | "
            f"{_yes_no(result.valid_json)} | {_yes_no(result.fields_present)} | "
            f"{_yes_no(fm.get('flood_risk_level', False))} | "
            f"{_yes_no(fm.get('water_extent_level', False))} | "
            f"{_yes_no(fm.get('confidence', False))} | "
            f"{_yes_no(result.all_fields_match)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "Y" if value else "N"


def model_name(backend: str, model: str, quant: str | None = None) -> str:
    if backend == "local" and quant and not Path(model).is_file():
        return f"{model}:{quant}"
    return model


def save_results(
    eval_dir: Path,
    summary: EvalSummary,
    dataset: str,
    backend: str,
    model: str,
    split: str,
    eval_run_id: str,
    *,
    quant: str | None = None,
    reasoning_effort: str | None = None,
    image_detail: str | None = None,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    risk = summary.risk_metrics()
    meta = {
        "eval_run_id": eval_run_id,
        "dataset": dataset,
        "backend": backend,
        "model": model,
        "quant": quant,
        "split": split,
        "reasoning_effort": reasoning_effort,
        "image_detail": image_detail,
        "schema_fields": list(EVAL_FIELDS),
        "summary": {
            "valid_json": summary.valid_json_accuracy(),
            "fields_present": summary.fields_present_accuracy(),
            "overall": summary.overall_accuracy(),
            "avg_latency_s": summary.avg_latency_s(),
            "risk_macro_precision": risk.macro_precision,
            "risk_macro_recall": risk.macro_recall,
            "risk_macro_f1": risk.macro_f1,
            "risk_balanced_accuracy": risk.balanced_accuracy,
            "invalid_or_missing_risk": risk.invalid_or_missing,
        },
        "risk_confusion": risk.confusion,
    }
    (eval_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    records = [
        {
            "id": result.id,
            "region": result.region,
            "timestamp": result.timestamp,
            "valid_json": result.valid_json,
            "fields_present": result.fields_present,
            "field_matches": result.field_matches,
            "latency_s": result.latency_s,
            "prediction": result.prediction,
            "ground_truth": result.ground_truth,
            "error": result.error,
            "attempts": result.attempts,
            "rgb_path": result.rgb_path,
            "swir_path": result.swir_path,
        }
        for result in summary.results
    ]
    (eval_dir / "results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
