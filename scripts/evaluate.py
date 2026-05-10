"""Evaluate Bali flood-risk models against existing test annotations.

Predictions are written under evals/{timestamp}/ and never overwrite
data/{run_id}/**/annotation.json.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from bali_flood_prevention.evaluator import (
    EvalSample,
    EvalSummary,
    SampleResult,
    evaluate_sample_with_retries,
    load_local_samples,
    make_llama_backend,
    make_openai_backend,
    model_name,
    render_report,
    save_results,
    start_llama_server,
    stop_server,
    wait_for_server,
)

PROJECT_ROOT = Path(__file__).parent.parent
EVALS_DIR = PROJECT_ROOT / "evals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Bali flood-risk predictions on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to data/{run_id}.")
    parser.add_argument(
        "--split",
        choices=["test"],
        default="test",
        help="Only the held-out test split is supported for this pre-evaluation.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["openai", "local"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help=(
            "OpenAI model name for --backend openai, Hugging Face repo for "
            "--backend local, or local GGUF path."
        ),
    )
    parser.add_argument(
        "--quant",
        default="Q8_0",
        help="GGUF quantization for Hugging Face local backends.",
    )
    parser.add_argument("--mmproj", default=None, help="Path to local mmproj GGUF.")
    parser.add_argument(
        "--chat-template",
        default=None,
        help=(
            "llama.cpp built-in chat template override. Usually leave unset "
            "so GGUF metadata or project defaults are used."
        ),
    )
    parser.add_argument(
        "--chat-template-file",
        default=None,
        help=(
            "llama.cpp chat template file override. SmolVLM defaults to the "
            "project template that preserves image_url markers."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="llama-server port.")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel workers.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="low",
        help="OpenAI reasoning effort.",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="low",
        help="OpenAI image detail.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Stop submitting new samples after this many backend errors. Use 0 to disable.",
    )
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--verbose-server", action="store_true")
    parser.add_argument(
        "--skip-chat-parsing",
        action="store_true",
        help=(
            "Pass --skip-chat-parsing to llama-server for manual debugging."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and count samples, but do not call any model or write eval files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    if args.concurrency is not None and args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")

    if args.backend == "openai" and not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Expected {PROJECT_ROOT / '.env'}")

    if args.backend == "local":
        if not args.model:
            raise ValueError("--model is required when using --backend local")
        if not args.dry_run and not shutil.which("llama-server"):
            raise RuntimeError("llama-server not found on PATH.")
        model_path = Path(args.model)
        if model_path.is_file() and not args.mmproj:
            raise ValueError("Local GGUF artifact evaluation requires --mmproj.")
        if args.mmproj and not Path(args.mmproj).is_file():
            raise FileNotFoundError(f"mmproj not found: {args.mmproj}")
        if args.chat_template_file and not Path(args.chat_template_file).is_file():
            raise FileNotFoundError(
                f"chat template file not found: {args.chat_template_file}"
            )


def local_chat_template(args: argparse.Namespace) -> str | None:
    if args.chat_template:
        return str(args.chat_template)
    return None


def local_chat_template_file(args: argparse.Namespace) -> str | None:
    if args.chat_template_file:
        return str(args.chat_template_file)
    if args.backend == "local" and "smolvlm" in args.model.lower():
        return str(PROJECT_ROOT / "templates" / "smolvlm-image-url.jinja")
    return None


def local_skip_chat_parsing(args: argparse.Namespace) -> bool:
    return bool(args.skip_chat_parsing)


def run_eval_jobs(
    samples: list[EvalSample],
    args: argparse.Namespace,
) -> list[SampleResult]:
    predict = (
        make_openai_backend(
            model=args.model,
            image_detail=args.image_detail,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if args.backend == "openai"
        else make_llama_backend(args.model, args.port)
    )
    concurrency = args.concurrency or (5 if args.backend == "openai" else 1)
    results: list[SampleResult] = []
    sample_iter = iter(samples)
    pending: dict[Future[SampleResult], EvalSample] = {}
    error_count = 0
    stopped_reason: str | None = None

    def submit_next(pool: ThreadPoolExecutor) -> None:
        nonlocal stopped_reason
        while len(pending) < concurrency and stopped_reason is None:
            try:
                sample = next(sample_iter)
            except StopIteration:
                return
            future = pool.submit(
                evaluate_sample_with_retries,
                sample,
                predict,
                args.retries,
                args.retry_delay,
            )
            pending[future] = sample

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        submit_next(pool)
        completed = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                results.append(result)
                completed += 1
                status = "error" if result.error else "ok"
                print(
                    f"[{completed}/{len(samples)}] {result.id} {status} "
                    f"latency={result.latency_s:.2f}s",
                    flush=True,
                )
                if result.error:
                    error_count += 1
                    if args.max_errors and error_count >= args.max_errors:
                        stopped_reason = (
                            f"Stopped after {error_count} backend errors "
                            f"because --max-errors={args.max_errors}."
                        )
            submit_next(pool)

    if stopped_reason:
        print(stopped_reason)
    return results


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    try:
        validate_args(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    dataset_dir = Path(args.dataset)
    samples = load_local_samples(dataset_dir, args.split, limit=args.limit)
    print(
        f"Loaded {len(samples)} {args.split} samples from {dataset_dir} | "
        f"backend={args.backend} | model={args.model}"
    )
    if args.dry_run:
        print("Dry run only; no model calls and no eval files written.")
        return

    server_process = None
    if args.backend == "local":
        print(
            f"Starting llama-server on port {args.port} with "
            f"{model_name(args.backend, args.model, args.quant)} ..."
        )
        server_process = start_llama_server(
            args.model,
            quant=args.quant or None,
            port=args.port,
            verbose=args.verbose_server,
            mmproj=args.mmproj,
            chat_template=local_chat_template(args),
            chat_template_file=local_chat_template_file(args),
            skip_chat_parsing=local_skip_chat_parsing(args),
        )
        try:
            wait_for_server(port=args.port)
        except Exception as exc:
            stop_server(server_process)
            print(f"ERROR: {exc}")
            sys.exit(1)
        print("llama-server ready.")

    try:
        results = run_eval_jobs(samples, args)
    finally:
        if server_process is not None:
            stop_server(server_process)

    sample_order = {sample.id: index for index, sample in enumerate(samples)}
    results.sort(key=lambda result: sample_order.get(result.id, 999999))
    summary = EvalSummary(results=results)
    eval_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mname = model_name(args.backend, args.model, args.quant)
    report = render_report(
        summary,
        dataset=str(dataset_dir),
        backend=args.backend,
        model=mname,
        split=args.split,
        eval_run_id=eval_run_id,
    )

    eval_dir = EVALS_DIR / eval_run_id
    save_results(
        eval_dir,
        summary,
        dataset=str(dataset_dir),
        backend=args.backend,
        model=mname,
        split=args.split,
        eval_run_id=eval_run_id,
        quant=args.quant if args.backend == "local" else None,
        reasoning_effort=args.reasoning_effort if args.backend == "openai" else None,
        image_detail=args.image_detail if args.backend == "openai" else None,
    )
    (eval_dir / "report.md").write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Report saved to {eval_dir / 'report.md'}")
    print(f"Results saved to {eval_dir / 'results.json'}")


if __name__ == "__main__":
    main()
