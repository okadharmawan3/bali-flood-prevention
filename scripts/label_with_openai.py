"""Label Bali flood samples one-by-one with the OpenAI Responses API.

The script sends each sample folder's rgb.png, swir.png, and metadata.json to
OpenAI, validates the strict structured response, and writes a clean
annotation.json beside the images. API response metadata is logged separately
under manifests/openai_label_log.jsonl.

Usage:
    uv run scripts/label_with_openai.py --run-dir data/20260504_150038 --limit 3 --dry-run
    uv run scripts/label_with_openai.py --run-dir data/20260504_150038 --region denpasar_bali --limit 20
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from bali_flood_prevention.schema import (
    FLOOD_LABEL_JSON_SCHEMA,
    OPENAI_LABELING_INSTRUCTIONS,
    dumps_label,
    validate_label,
)

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    region: str
    sample_dir: Path
    rgb_path: Path
    swir_path: Path
    metadata_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class LabelResult:
    sample_id: str
    status: str
    annotation_path: str | None = None
    response_id: str | None = None
    model: str | None = None
    error: str | None = None
    raw_output: str | None = None
    attempts: int = 0


def iter_samples(run_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split in ("train", "test"):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        for loc_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in loc_dir.iterdir() if path.is_dir()):
                sample_id = f"{loc_dir.name}/{sample_dir.name}"
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        split=split,
                        region=loc_dir.name,
                        sample_dir=sample_dir,
                        rgb_path=sample_dir / "rgb.png",
                        swir_path=sample_dir / "swir.png",
                        metadata_path=sample_dir / "metadata.json",
                        annotation_path=sample_dir / "annotation.json",
                    )
                )
    return samples


def encode_data_url(path: Path) -> str:
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_user_text(sample: Sample, metadata: dict[str, Any]) -> str:
    context = {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "region": sample.region,
        "metadata": metadata,
    }
    return (
        "Label this single Bali Sentinel-2 sample. Use the two images and "
        "metadata together. Remember: exposure/context fields can be true even "
        "without active flooding; do not mark dense urban or cropland context "
        "false simply because no floodwater is visible.\n\n"
        f"Sample context JSON:\n{json.dumps(context, indent=2, ensure_ascii=True)}"
    )


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    # Defensive fallback for SDK/API shape differences.
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def response_debug_text(response: Any) -> str:
    """Return compact debug text for a response without image/request payloads."""
    try:
        return json.dumps(response.model_dump(mode="json"), ensure_ascii=True)[:4000]
    except Exception:
        return repr(response)[:4000]


def response_token_error_text(response: Any) -> str | None:
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None) if incomplete is not None else None
    if reason != "max_output_tokens":
        return None
    usage = getattr(response, "usage", None)
    reasoning_tokens = None
    output_tokens = None
    if usage is not None:
        output_tokens = getattr(usage, "output_tokens", None)
        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is not None:
            reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
    return (
        "OpenAI used the full max_output_tokens budget before producing JSON "
        f"(response_id={getattr(response, 'id', '')}, "
        f"output_tokens={output_tokens}, reasoning_tokens={reasoning_tokens}). "
        "Lower --reasoning-effort or raise --max-output-tokens."
    )


def is_risky_mini_xhigh(model: str, reasoning_effort: str) -> bool:
    normalized = model.lower()
    return "mini" in normalized and reasoning_effort == "xhigh"


def select_samples(
    run_dir: Path,
    region: str | None,
    split: str | None,
    sample_ids: list[str] | None,
    overwrite: bool,
    limit: int | None,
) -> list[Sample]:
    samples = iter_samples(run_dir)
    if region:
        samples = [sample for sample in samples if sample.region == region]
    if split:
        samples = [sample for sample in samples if sample.split == split]
    if sample_ids:
        wanted = set(sample_ids)
        samples = [sample for sample in samples if sample.sample_id in wanted]
    if not overwrite:
        samples = [sample for sample in samples if not sample.annotation_path.exists()]
    if limit is not None:
        samples = samples[:limit]
    return samples


def call_openai_label(
    client: OpenAI,
    sample: Sample,
    model: str,
    image_detail: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> tuple[dict[str, object], str, str]:
    metadata = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
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
                    {"type": "input_text", "text": build_user_text(sample, metadata)},
                    {
                        "type": "input_image",
                        "image_url": encode_data_url(sample.rgb_path),
                        "detail": image_detail,
                    },
                    {
                        "type": "input_image",
                        "image_url": encode_data_url(sample.swir_path),
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
        token_error = response_token_error_text(response)
        if token_error:
            raise ValueError(token_error)
        raise ValueError(f"OpenAI returned empty output text. Response: {response_debug_text(response)}")
    label = validate_label(json.loads(raw))
    return label, str(getattr(response, "id", "")), raw


def label_sample(
    sample: Sample,
    client: OpenAI | None,
    model: str,
    image_detail: str,
    max_output_tokens: int,
    reasoning_effort: str,
    overwrite: bool,
    retries: int,
    retry_delay: float,
    dry_run: bool,
) -> LabelResult:
    if sample.annotation_path.exists() and not overwrite:
        return LabelResult(sample_id=sample.sample_id, status="skipped_existing")
    missing = [
        path.name
        for path in (sample.rgb_path, sample.swir_path, sample.metadata_path)
        if not path.exists()
    ]
    if missing:
        return LabelResult(
            sample_id=sample.sample_id,
            status="missing_files",
            error=", ".join(missing),
        )
    if dry_run:
        return LabelResult(sample_id=sample.sample_id, status="would_label")
    if client is None:
        return LabelResult(
            sample_id=sample.sample_id,
            status="error",
            model=model,
            error="OpenAI client is not configured.",
        )

    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            label, response_id, raw = call_openai_label(
                client=client,
                sample=sample,
                model=model,
                image_detail=image_detail,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
            sample.annotation_path.write_text(dumps_label(label, indent=2), encoding="utf-8")
            return LabelResult(
                sample_id=sample.sample_id,
                status="labeled",
                annotation_path=str(sample.annotation_path),
                response_id=response_id,
                model=model,
                raw_output=raw,
                attempts=attempt + 1,
            )
        except Exception as exc:
            last_error = str(exc)
            if "empty output text" in last_error.lower():
                break
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))

    return LabelResult(
        sample_id=sample.sample_id,
        status="error",
        model=model,
        error=last_error,
        attempts=attempt + 1,
    )


def write_log_row(path: Path, result: LabelResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": result.sample_id,
        "status": result.status,
        "annotation_path": result.annotation_path,
        "response_id": result.response_id,
        "model": result.model,
        "error": result.error,
        "raw_output": result.raw_output,
        "attempts": result.attempts,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def is_error_result(result: LabelResult) -> bool:
    return result.status in {"error", "missing_files"}


def submit_label_job(
    pool: ThreadPoolExecutor,
    sample: Sample,
    client: OpenAI | None,
    args: argparse.Namespace,
) -> Future[LabelResult]:
    return pool.submit(
        label_sample,
        sample,
        client,
        args.model,
        args.image_detail,
        args.max_output_tokens,
        args.reasoning_effort,
        args.overwrite,
        args.retries,
        args.retry_delay,
        args.dry_run,
    )


def run_label_jobs(
    samples: list[Sample],
    client: OpenAI | None,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, int]:
    status_counts: dict[str, int] = {}
    error_count = 0
    sample_iter = iter(samples)
    pending: dict[Future[LabelResult], Sample] = {}
    stopped_reason: str | None = None

    def fill_pending(pool: ThreadPoolExecutor) -> None:
        nonlocal stopped_reason
        while len(pending) < args.concurrency and stopped_reason is None:
            try:
                sample = next(sample_iter)
            except StopIteration:
                break
            future = submit_label_job(pool, sample, client, args)
            pending[future] = sample

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        fill_pending(pool)
        with tqdm(total=len(samples), desc="labels", unit="sample") as progress:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result = future.result()
                    status_counts[result.status] = status_counts.get(result.status, 0) + 1
                    write_log_row(log_path, result)
                    progress.update(1)

                    if result.status in {"labeled", "error", "missing_files"}:
                        suffix = f" response={result.response_id}" if result.response_id else ""
                        error = f" error={result.error}" if result.error else ""
                        tqdm.write(f"[{result.status}] {result.sample_id}{suffix}{error}")

                    if is_error_result(result):
                        error_count += 1
                        if args.max_errors and error_count >= args.max_errors:
                            stopped_reason = (
                                f"Stopped after {error_count} errors because "
                                f"--max-errors={args.max_errors}."
                            )
                fill_pending(pool)

    if stopped_reason:
        status_counts["stopped_early"] = 1
        print(stopped_reason)
    return status_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label Bali flood samples with OpenAI, one folder at a time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Path to data/{run_id}.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--region", default=None, help="Only label one region id.")
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sample-id", action="append", default=None, help="Only label this sample id. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel API calls. Keep low for cost/rate-limit control.")
    parser.add_argument("--max-errors", type=int, default=10, help="Stop submitting new samples after this many errors. Use 0 for no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing annotation.json files.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be labeled without calling the API.")
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="low")
    parser.add_argument("--allow-xhigh-mini", action="store_true", help="Allow gpt mini models with xhigh reasoning despite frequent token-budget exhaustion.")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if is_risky_mini_xhigh(args.model, args.reasoning_effort) and not args.allow_xhigh_mini:
        print("Blocked risky configuration: mini model + xhigh reasoning.")
        print("This combination can spend the entire output budget on hidden reasoning and return no JSON.")
        print("Use --reasoning-effort medium or high, or add --allow-xhigh-mini if you deliberately want to risk it.")
        sys.exit(2)
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print(f"OPENAI_API_KEY is not set. Expected local env file: {env_path}")
        print("Create it from .env.example or set OPENAI_API_KEY in your shell environment.")
        sys.exit(1)

    samples = select_samples(
        run_dir=run_dir,
        region=args.region,
        split=args.split,
        sample_ids=args.sample_id,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    log_path = run_dir / "manifests" / "openai_label_log.jsonl"
    print(
        f"Labeling {len(samples)} samples | model={args.model} | "
        f"detail={args.image_detail} | concurrency={args.concurrency} | dry_run={args.dry_run}"
    )

    client = OpenAI() if not args.dry_run else None
    status_counts = run_label_jobs(samples, client, args, log_path)

    print("Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<18} {count}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
