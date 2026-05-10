"""Create, monitor, and collect OpenAI Batch API jobs for Bali flood labels.

This is the cost-efficient path for labeling many samples. It builds JSONL
request files for the Responses API, uploads them with purpose=batch, creates
one or more 24h batch jobs, then later collects successful outputs into each
sample folder's annotation.json.

Examples:
    uv run scripts/label_with_openai_batch.py create --run-dir data/20260504_150038 --region denpasar_bali --limit 5 --dry-run
    uv run scripts/label_with_openai_batch.py create --run-dir data/20260504_150038 --reasoning-effort xhigh
    uv run scripts/label_with_openai_batch.py status --batch-manifest data/20260504_150038/manifests/openai_batches/20260505_104500/group_manifest.json
    uv run scripts/label_with_openai_batch.py collect --batch-manifest data/20260504_150038/manifests/openai_batches/20260505_104500/group_manifest.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
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
ENDPOINT = "/v1/responses"


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def is_risky_mini_xhigh(model: str, reasoning_effort: str) -> bool:
    normalized = model.lower()
    return "mini" in normalized and reasoning_effort == "xhigh"


def path_for_json(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_manifest_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


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


def select_samples(args: argparse.Namespace) -> list[Sample]:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    samples = iter_samples(run_dir)
    if args.region:
        samples = [sample for sample in samples if sample.region == args.region]
    if args.split:
        samples = [sample for sample in samples if sample.split == args.split]
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [sample for sample in samples if sample.sample_id in wanted]
    if not args.overwrite:
        samples = [sample for sample in samples if not sample.annotation_path.exists()]
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples


def ensure_sample_files(sample: Sample) -> None:
    missing = [
        path.name
        for path in (sample.rgb_path, sample.swir_path, sample.metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{sample.sample_id} missing: {', '.join(missing)}")


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


def build_request_body(
    sample: Sample,
    model: str,
    image_detail: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    metadata = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
    return {
        "model": model,
        "input": [
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
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bali_flood_label",
                "schema": FLOOD_LABEL_JSON_SCHEMA,
                "strict": True,
            }
        },
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
    }


def build_request_line(
    sample: Sample,
    model: str,
    image_detail: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> tuple[str, str]:
    custom_id = safe_id(sample.sample_id)
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": build_request_body(
            sample=sample,
            model=model,
            image_detail=image_detail,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        ),
    }
    line = json.dumps(request, ensure_ascii=True, separators=(",", ":")) + "\n"
    return custom_id, line


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def batch_status_to_dict(batch: Any) -> dict[str, Any]:
    if hasattr(batch, "model_dump"):
        return batch.model_dump(mode="json")
    if isinstance(batch, dict):
        return batch
    return json.loads(json.dumps(batch, default=str))


def create_batch_for_file(
    client: OpenAI,
    input_path: Path,
    description: str,
) -> tuple[str, str, dict[str, Any]]:
    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    input_file_id = str(getattr(uploaded, "id"))
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={"description": description[:512]},
    )
    batch_data = batch_status_to_dict(batch)
    return input_file_id, str(batch_data["id"]), batch_data


def create_batches(args: argparse.Namespace) -> None:
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    if is_risky_mini_xhigh(args.model, args.reasoning_effort) and not args.allow_xhigh_mini:
        print("Blocked risky configuration: mini model + xhigh reasoning.")
        print("This combination can spend the entire output budget on hidden reasoning and return no JSON.")
        print("Use --reasoning-effort medium or high, or add --allow-xhigh-mini if you deliberately want to risk it.")
        sys.exit(2)
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print(f"OPENAI_API_KEY is not set. Expected local env file: {env_path}")
        print("Create it from .env.example or set OPENAI_API_KEY in your shell environment.")
        sys.exit(1)

    run_dir = Path(args.run_dir)
    samples = select_samples(args)
    if not samples:
        print("No samples selected.")
        return

    local_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = run_dir / "manifests" / "openai_batches" / local_id
    group_dir.mkdir(parents=True, exist_ok=True)
    max_file_bytes = int(args.max_batch_file_mb * 1024 * 1024)
    max_requests = args.max_requests_per_batch

    print(
        f"Preparing {len(samples)} samples | model={args.model} | "
        f"reasoning={args.reasoning_effort} | detail={args.image_detail} | dry_run={args.dry_run}"
    )
    print(
        f"Chunk limits: {max_requests} requests or {args.max_batch_file_mb:.1f} MB per JSONL"
    )

    chunks: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_records: list[dict[str, Any]] = []
    current_bytes = 0

    def flush_chunk() -> None:
        nonlocal current_lines, current_records, current_bytes
        if not current_lines:
            return
        index = len(chunks)
        batch_dir = group_dir / f"batch_{index:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        input_path = batch_dir / "input.jsonl"
        input_path.write_text("".join(current_lines), encoding="utf-8")
        manifest_path = batch_dir / "manifest.json"
        manifest = {
            "kind": "openai_batch_manifest",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir.resolve()),
            "endpoint": ENDPOINT,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "image_detail": args.image_detail,
            "max_output_tokens": args.max_output_tokens,
            "dry_run": args.dry_run,
            "status": "prepared",
            "batch_id": None,
            "input_file_id": None,
            "output_file_id": None,
            "error_file_id": None,
            "input_path": path_for_json(input_path, group_dir),
            "sample_count": len(current_records),
            "input_file_bytes": input_path.stat().st_size,
            "records": current_records,
        }
        write_json(manifest_path, manifest)
        chunks.append(
            {
                "manifest_path": path_for_json(manifest_path, group_dir),
                "sample_count": len(current_records),
                "input_file_bytes": input_path.stat().st_size,
                "batch_id": None,
                "status": "prepared",
            }
        )
        current_lines = []
        current_records = []
        current_bytes = 0

    for sample in tqdm(samples, desc="build-jsonl", unit="sample"):
        ensure_sample_files(sample)
        custom_id, line = build_request_line(
            sample=sample,
            model=args.model,
            image_detail=args.image_detail,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > max_file_bytes:
            raise ValueError(
                f"{sample.sample_id} request is {line_bytes / 1024 / 1024:.1f} MB, "
                f"larger than --max-batch-file-mb={args.max_batch_file_mb}."
            )
        if current_lines and (
            len(current_lines) >= max_requests
            or current_bytes + line_bytes > max_file_bytes
        ):
            flush_chunk()
        current_lines.append(line)
        current_bytes += line_bytes
        current_records.append(
            {
                "custom_id": custom_id,
                "sample_id": sample.sample_id,
                "split": sample.split,
                "region": sample.region,
                "sample_dir": path_for_json(sample.sample_dir, run_dir),
                "annotation_path": path_for_json(sample.annotation_path, run_dir),
                "rgb_path": path_for_json(sample.rgb_path, run_dir),
                "swir_path": path_for_json(sample.swir_path, run_dir),
                "metadata_path": path_for_json(sample.metadata_path, run_dir),
            }
        )
    flush_chunk()

    client = None if args.dry_run else OpenAI()
    for index, chunk in enumerate(chunks):
        manifest_path = resolve_manifest_path(chunk["manifest_path"], group_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_path = resolve_manifest_path(manifest["input_path"], group_dir)
        if args.dry_run:
            print(
                f"[prepared] batch_{index:03d}: {manifest['sample_count']} samples, "
                f"{manifest['input_file_bytes'] / 1024 / 1024:.1f} MB"
            )
            continue

        assert client is not None
        input_file_id, batch_id, batch_data = create_batch_for_file(
            client,
            input_path,
            description=f"Bali flood labels {local_id} batch_{index:03d}",
        )
        manifest.update(
            {
                "status": batch_data.get("status", "submitted"),
                "batch_id": batch_id,
                "input_file_id": input_file_id,
                "output_file_id": batch_data.get("output_file_id"),
                "error_file_id": batch_data.get("error_file_id"),
                "last_batch_object": batch_data,
                "submitted_at_utc": utc_now(),
            }
        )
        write_json(manifest_path, manifest)
        chunk.update({"batch_id": batch_id, "status": manifest["status"]})
        print(
            f"[submitted] batch_{index:03d}: {batch_id} | "
            f"{manifest['sample_count']} samples | {manifest['input_file_bytes'] / 1024 / 1024:.1f} MB"
        )

    group_manifest = {
        "kind": "openai_batch_group_manifest",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir.resolve()),
        "group_dir": str(group_dir.resolve()),
        "endpoint": ENDPOINT,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "image_detail": args.image_detail,
        "max_output_tokens": args.max_output_tokens,
        "dry_run": args.dry_run,
        "sample_count": len(samples),
        "batch_count": len(chunks),
        "batches": chunks,
    }
    group_path = group_dir / "group_manifest.json"
    write_json(group_path, group_manifest)
    print(f"Group manifest: {group_path}")


def load_group_or_single_manifest(path: Path) -> tuple[Path, list[Path]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") == "openai_batch_group_manifest":
        group_dir = Path(data.get("group_dir", path.parent))
        manifests = [
            resolve_manifest_path(item["manifest_path"], group_dir)
            for item in data.get("batches", [])
        ]
        return group_dir, manifests
    if data.get("kind") == "openai_batch_manifest":
        return path.parent.parent, [path]
    raise ValueError(f"Unrecognized batch manifest type: {path}")


def status_batches(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print(f"OPENAI_API_KEY is not set. Expected local env file: {PROJECT_ROOT / '.env'}")
        sys.exit(1)

    _, manifest_paths = load_group_or_single_manifest(Path(args.batch_manifest))
    client = OpenAI()
    counts: dict[str, int] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_id = manifest.get("batch_id")
        if not batch_id:
            status = "prepared_only"
            counts[status] = counts.get(status, 0) + 1
            print(f"[{status}] {manifest_path}")
            continue
        batch_data = batch_status_to_dict(client.batches.retrieve(batch_id))
        status = str(batch_data.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        manifest.update(
            {
                "status": status,
                "output_file_id": batch_data.get("output_file_id"),
                "error_file_id": batch_data.get("error_file_id"),
                "last_batch_object": batch_data,
                "last_status_checked_at_utc": utc_now(),
            }
        )
        write_json(manifest_path, manifest)
        request_counts = batch_data.get("request_counts") or {}
        print(
            f"[{status}] {batch_id} | samples={manifest.get('sample_count')} | "
            f"requests={request_counts}"
        )
    print("Summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status:<18} {count}")


def download_file(client: OpenAI, file_id: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    if hasattr(content, "write_to_file"):
        content.write_to_file(str(target_path))
        return
    if hasattr(content, "read"):
        data = content.read()
    elif hasattr(content, "content"):
        data = content.content
    else:
        data = bytes(content)
    if isinstance(data, str):
        target_path.write_text(data, encoding="utf-8")
    else:
        target_path.write_bytes(data)


def extract_output_text_from_body(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def collect_output_file(
    output_path: Path,
    manifest: dict[str, Any],
    run_dir: Path,
    overwrite: bool,
    log_path: Path,
) -> dict[str, int]:
    records_by_id = {record["custom_id"]: record for record in manifest.get("records", [])}
    counts: dict[str, int] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            record = records_by_id.get(custom_id)
            if not record:
                counts["unknown_custom_id"] = counts.get("unknown_custom_id", 0) + 1
                append_jsonl(
                    log_path,
                    {
                        "timestamp_utc": utc_now(),
                        "status": "unknown_custom_id",
                        "custom_id": custom_id,
                        "error": "Output custom_id was not found in local manifest.",
                    },
                )
                continue

            annotation_path = run_dir / record["annotation_path"]
            if annotation_path.exists() and not overwrite:
                counts["skipped_existing"] = counts.get("skipped_existing", 0) + 1
                continue

            error = row.get("error")
            response = row.get("response") or {}
            status_code = response.get("status_code")
            if error or status_code != 200:
                counts["api_error"] = counts.get("api_error", 0) + 1
                append_jsonl(
                    log_path,
                    {
                        "timestamp_utc": utc_now(),
                        "status": "api_error",
                        "custom_id": custom_id,
                        "sample_id": record["sample_id"],
                        "status_code": status_code,
                        "error": error,
                        "response": response,
                    },
                )
                continue

            body = response.get("body") or {}
            try:
                raw_output = extract_output_text_from_body(body)
                if not raw_output:
                    raise ValueError(f"empty output text: {json.dumps(body, ensure_ascii=True)[:4000]}")
                label = validate_label(json.loads(raw_output))
                annotation_path.write_text(dumps_label(label, indent=2), encoding="utf-8")
                counts["labeled"] = counts.get("labeled", 0) + 1
                append_jsonl(
                    log_path,
                    {
                        "timestamp_utc": utc_now(),
                        "status": "labeled",
                        "custom_id": custom_id,
                        "sample_id": record["sample_id"],
                        "annotation_path": str(annotation_path),
                        "response_id": body.get("id"),
                        "model": body.get("model"),
                        "usage": body.get("usage"),
                    },
                )
            except Exception as exc:
                counts["parse_error"] = counts.get("parse_error", 0) + 1
                append_jsonl(
                    log_path,
                    {
                        "timestamp_utc": utc_now(),
                        "status": "parse_error",
                        "custom_id": custom_id,
                        "sample_id": record["sample_id"],
                        "error": str(exc),
                        "body": body,
                    },
                )
    return counts


def collect_batches(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print(f"OPENAI_API_KEY is not set. Expected local env file: {PROJECT_ROOT / '.env'}")
        sys.exit(1)

    _, manifest_paths = load_group_or_single_manifest(Path(args.batch_manifest))
    client = OpenAI()
    total_counts: dict[str, int] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_id = manifest.get("batch_id")
        run_dir = Path(manifest["run_dir"])
        batch_dir = manifest_path.parent
        if not batch_id:
            total_counts["prepared_only"] = total_counts.get("prepared_only", 0) + 1
            print(f"[prepared_only] {manifest_path}")
            continue

        batch_data = batch_status_to_dict(client.batches.retrieve(batch_id))
        status = str(batch_data.get("status", "unknown"))
        manifest.update(
            {
                "status": status,
                "output_file_id": batch_data.get("output_file_id"),
                "error_file_id": batch_data.get("error_file_id"),
                "last_batch_object": batch_data,
                "last_status_checked_at_utc": utc_now(),
            }
        )
        write_json(manifest_path, manifest)
        if status not in {"completed", "cancelled", "expired"}:
            total_counts[f"not_ready_{status}"] = total_counts.get(f"not_ready_{status}", 0) + 1
            print(f"[not_ready] {batch_id} status={status}")
            continue

        output_file_id = manifest.get("output_file_id")
        error_file_id = manifest.get("error_file_id")
        if error_file_id:
            error_path = batch_dir / "error.jsonl"
            download_file(client, error_file_id, error_path)
            print(f"[downloaded errors] {batch_id}: {error_path}")
        if not output_file_id:
            total_counts["no_output_file"] = total_counts.get("no_output_file", 0) + 1
            print(f"[no_output_file] {batch_id}")
            continue

        output_path = batch_dir / "output.jsonl"
        download_file(client, output_file_id, output_path)
        counts = collect_output_file(
            output_path=output_path,
            manifest=manifest,
            run_dir=run_dir,
            overwrite=args.overwrite,
            log_path=run_dir / "manifests" / "openai_batch_collect_log.jsonl",
        )
        for key, value in counts.items():
            total_counts[key] = total_counts.get(key, 0) + value
        print(f"[collected] {batch_id}: {counts}")

    print("Summary:")
    for status, count in sorted(total_counts.items()):
        print(f"  {status:<22} {count}")


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Path to data/{run_id}.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--region", default=None, help="Only label one region id.")
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sample-id", action="append", default=None, help="Only label this sample id. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Include samples that already have annotation.json.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API workflow for Bali flood sample labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Build JSONL, upload it, and create batch job(s).")
    add_selection_args(create)
    create.add_argument("--dry-run", action="store_true", help="Build local JSONL/manifest without uploading or creating API batches.")
    create.add_argument("--image-detail", choices=["low", "high", "auto"], default="low")
    create.add_argument("--max-output-tokens", type=int, default=8192)
    create.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="xhigh")
    create.add_argument("--allow-xhigh-mini", action="store_true", help="Allow gpt mini models with xhigh reasoning despite frequent token-budget exhaustion.")
    create.add_argument("--max-requests-per-batch", type=int, default=50, help="Split JSONL files after this many requests.")
    create.add_argument("--max-batch-file-mb", type=float, default=180.0, help="Split JSONL files before this file size.")
    create.set_defaults(func=create_batches)

    status = subparsers.add_parser("status", help="Refresh status for one batch manifest or a group manifest.")
    status.add_argument("--batch-manifest", required=True, help="Path to manifest.json or group_manifest.json.")
    status.set_defaults(func=status_batches)

    collect = subparsers.add_parser("collect", help="Download completed batch outputs and write annotation.json files.")
    collect.add_argument("--batch-manifest", required=True, help="Path to manifest.json or group_manifest.json.")
    collect.add_argument("--overwrite", action="store_true", help="Replace existing annotation.json files while collecting.")
    collect.set_defaults(func=collect_batches)
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    if getattr(args, "max_requests_per_batch", 1) < 1:
        raise ValueError("--max-requests-per-batch must be >= 1")
    if getattr(args, "max_batch_file_mb", 1.0) <= 0:
        raise ValueError("--max-batch-file-mb must be > 0")
    args.func(args)


if __name__ == "__main__":
    main()
