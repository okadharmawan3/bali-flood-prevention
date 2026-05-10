"""Convert a fine-tuned VLM checkpoint to GGUF artifacts.

Produces the two files llama-server needs for VLM inference:
  1. Quantized backbone GGUF
  2. F16 mmproj GGUF

This script prepares artifacts only when the user runs it manually after
training. It is not invoked by the preparation workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
VALID_QUANTS = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required tool on PATH: {name}")


def check_setup_tools(quant: str) -> None:
    require_tool("git")

    needs_quantize_binary = quant != "F16"
    has_quantize_binary = find_quantize_binary(required=False) is not None
    if not needs_quantize_binary or has_quantize_binary:
        return

    require_tool("cmake")
    if not any(shutil.which(tool) for tool in ("cl", "c++", "g++", "clang++")):
        raise RuntimeError(
            "No C++ compiler found on PATH. Install Visual Studio Build Tools "
            "or another CMake-compatible C++ toolchain before quantizing."
        )


def setup_llama_cpp() -> None:
    if not LLAMA_CPP_DIR.exists():
        print(f"Cloning llama.cpp into {LLAMA_CPP_DIR} ...")
        run(["git", "clone", "--depth=1", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR)])

    if find_quantize_binary(required=False) is None:
        print("Building llama-quantize ...")
        run(["cmake", "-B", "build"], cwd=LLAMA_CPP_DIR)
        run(
            ["cmake", "--build", "build", "--config", "Release", "-t", "llama-quantize"],
            cwd=LLAMA_CPP_DIR,
        )


def find_quantize_binary(*, required: bool = True) -> Path | None:
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize",
    ]
    for path in candidates:
        if path.is_file():
            return path

    path_on_path = shutil.which("llama-quantize")
    if path_on_path:
        return Path(path_on_path)

    if required:
        raise FileNotFoundError("llama-quantize binary not found after build")
    return None


def convert_to_f16(checkpoint: Path, f16_output: Path) -> None:
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found: {convert_script}")
    print(f"Converting backbone to F16 GGUF: {f16_output} ...")
    run(
        [
            sys.executable,
            str(convert_script),
            str(checkpoint),
            "--outtype",
            "f16",
            "--outfile",
            str(f16_output),
        ]
    )


def convert_to_mmproj(checkpoint: Path, mmproj_output: Path) -> None:
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    print(f"Converting vision components to mmproj GGUF: {mmproj_output} ...")
    run(
        [
            sys.executable,
            str(convert_script),
            str(checkpoint),
            "--mmproj",
            "--outfile",
            str(mmproj_output),
        ]
    )


def quantize_backbone(f16_path: Path, output: Path, quant: str) -> None:
    if quant == "F16":
        f16_path.replace(output)
        return
    quantize_bin = find_quantize_binary()
    print(f"Quantizing backbone to {quant}: {output} ...")
    run([str(quantize_bin), str(f16_path), str(output), quant])


def default_mmproj_path(output: Path) -> Path:
    return output.parent / f"mmproj-{output.stem}.gguf"


def normalize_tokenizer_config(checkpoint: Path) -> None:
    tokenizer_config = checkpoint / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return

    data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    changed = False

    if data.get("tokenizer_class") == "TokenizersBackend":
        data["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True

    if isinstance(data.get("extra_special_tokens"), list):
        data["extra_special_tokens"] = {}
        changed = True

    if not changed:
        return

    tokenizer_config.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Normalized tokenizer_config.json for transformers/llama.cpp loading"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned VLM checkpoint to GGUF."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output backbone GGUF path, e.g. ./outputs/lfm2.5-vl-bali-flood-Q8_0.gguf.",
    )
    parser.add_argument(
        "--quant",
        default="Q8_0",
        choices=VALID_QUANTS,
        help="Backbone quantization. The mmproj is always F16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    output.parent.mkdir(parents=True, exist_ok=True)
    normalize_tokenizer_config(checkpoint)
    f16_path = output.parent / f"{output.stem}-F16.gguf"
    mmproj = default_mmproj_path(output)

    check_setup_tools(args.quant)
    setup_llama_cpp()
    convert_to_f16(checkpoint, f16_path)
    try:
        quantize_backbone(f16_path, output, args.quant)
    finally:
        if args.quant != "F16" and f16_path.exists():
            f16_path.unlink()
    convert_to_mmproj(checkpoint, mmproj)

    print()
    print("Done.")
    print(f"  Backbone: {output}")
    print(f"  mmproj  : {mmproj}")
    print()
    print("Evaluate with:")
    print(
        "  uv run scripts/evaluate.py --dataset data/20260504_150038 --split test "
        f"--backend local --model {output} --mmproj {mmproj} --port 8082 --concurrency 1"
    )


if __name__ == "__main__":
    main()
