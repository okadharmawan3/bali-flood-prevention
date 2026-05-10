"""Finalize a Modal fine-tune run into a tested GGUF model.

This wraps the manual post-training flow:
  1. Find the final checkpoint in the Modal volume.
  2. Download only that checkpoint, avoiding ray_logs and symlinks.
  3. Convert/quantize to GGUF via scripts/quantize.py.
  4. Evaluate the GGUF model on the Bali test split.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_PREFIX = "lfm2.5-VL-450M-vlm_sft-bali_flood"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "lfm2.5-vl-bali-flood-Q8_0.gguf"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "20260504_150038"


def run(cmd: list[str], *, capture_json: bool = False) -> Any:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=capture_json,
    )
    if result.returncode != 0:
        if capture_json:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    if capture_json:
        return json.loads(result.stdout)
    return None


def modal_ls(volume: str, path: str) -> list[dict[str, Any]]:
    return run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "ls",
            volume,
            path,
            "--json",
        ],
        capture_json=True,
    )


def basename(filename: str) -> str:
    return filename.replace("\\", "/").rstrip("/").split("/")[-1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def modified_key(entry: dict[str, Any]) -> str:
    return str(entry.get("Created/Modified", ""))


def find_latest_run(volume: str, run_prefix: str) -> str:
    entries = modal_ls(volume, "/")
    candidates = [
        entry
        for entry in entries
        if entry.get("Type") == "dir" and basename(entry["Filename"]).startswith(run_prefix)
    ]
    if not candidates:
        raise SystemExit(
            f"No Modal run directory found with prefix {run_prefix!r}. "
            "Pass --run explicitly."
        )
    chosen = max(candidates, key=modified_key)
    return basename(chosen["Filename"])


def checkpoint_step(name: str) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"-e\d+s(\d+)-", name)
    if match:
        return int(match.group(1))
    match = re.search(r"global_step(\d+)", name)
    if match:
        return int(match.group(1))
    return -1


def find_final_checkpoint(volume: str, run_name: str) -> tuple[str, str]:
    entries = modal_ls(volume, f"/{run_name}")
    candidates = []
    for entry in entries:
        if entry.get("Type") != "dir":
            continue
        name = basename(entry["Filename"])
        if name == "ray_logs":
            continue
        step = checkpoint_step(name)
        if step >= 0:
            candidates.append((step, entry))

    if not candidates:
        raise SystemExit(f"No checkpoint directories found under /{run_name}")

    _, chosen = max(candidates, key=lambda item: (item[0], modified_key(item[1])))
    remote_path = "/" + chosen["Filename"].replace("\\", "/").lstrip("/")
    return basename(chosen["Filename"]), remote_path


def download_checkpoint(volume: str, remote_checkpoint: str, download_root: Path) -> Path:
    checkpoint_name = basename(remote_checkpoint)
    local_checkpoint = download_root / checkpoint_name
    if (local_checkpoint / "model.safetensors").is_file():
        print(f"\nUsing existing local checkpoint: {local_checkpoint}")
        return local_checkpoint

    download_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            volume,
            remote_checkpoint,
            str(download_root),
        ]
    )
    if not (local_checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"Downloaded checkpoint is incomplete: {local_checkpoint}")
    return local_checkpoint


def default_mmproj_path(output: Path) -> Path:
    return output.parent / f"mmproj-{output.stem}.gguf"


def quantize(checkpoint: Path, output: Path, quant: str) -> Path:
    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "quantize.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--quant",
            quant,
        ]
    )
    mmproj = default_mmproj_path(output)
    if not output.is_file() or not mmproj.is_file():
        raise SystemExit("Quantization did not produce both GGUF files.")
    return mmproj


def evaluate(args: argparse.Namespace, model: Path, mmproj: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--dataset",
        str(project_path(args.dataset)),
        "--split",
        args.split,
        "--backend",
        "local",
        "--model",
        str(model),
        "--mmproj",
        str(mmproj),
        "--port",
        str(args.port),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.max_errors is not None:
        cmd.extend(["--max-errors", str(args.max_errors)])
    run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, quantize, and evaluate the latest Bali fine-tune checkpoint."
    )
    parser.add_argument("--volume", default="bali-flood-prevention")
    parser.add_argument(
        "--run",
        default=None,
        help="Modal run directory. If omitted, the newest run matching --run-prefix is used.",
    )
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument(
        "--download-root",
        default=str(PROJECT_ROOT / "outputs" / "modal-checkpoints"),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quant", default="Q8_0")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="test")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local checkpoint path to use with --skip-download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = project_path(args.output)

    if args.skip_download:
        if not args.checkpoint:
            raise SystemExit("--skip-download requires --checkpoint")
        checkpoint = project_path(args.checkpoint)
    else:
        run_name = args.run or find_latest_run(args.volume, args.run_prefix)
        checkpoint_name, remote_checkpoint = find_final_checkpoint(args.volume, run_name)
        print(f"\nSelected run       : {run_name}")
        print(f"Selected checkpoint: {checkpoint_name}")
        checkpoint = download_checkpoint(
            args.volume,
            remote_checkpoint,
            project_path(args.download_root),
        )

    if args.skip_quantize:
        mmproj = default_mmproj_path(output)
        if not output.is_file() or not mmproj.is_file():
            raise SystemExit("--skip-quantize requires existing model and mmproj GGUF files.")
    else:
        mmproj = quantize(checkpoint, output, args.quant)

    if not args.skip_eval:
        evaluate(args, output, mmproj)

    print("\nDone.")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Model GGUF: {output}")
    print(f"  mmproj    : {mmproj}")
    print("\nComparison app:")
    print("  uv run streamlit run app/eval_compare.py")


if __name__ == "__main__":
    main()
