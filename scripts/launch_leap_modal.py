"""Launch leap-finetune Modal jobs from the Bali project environment.

This avoids running ``uv`` inside the upstream ``leap-finetune`` checkout on
Windows, where its lockfile intentionally supports only Linux and macOS. The
actual training environment is still built by Modal from leap-finetune's
Linux-compatible lockfile.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bali_flood_finetune_modal.yaml"
DEFAULT_LEAP_DIR = PROJECT_ROOT / "leap-finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a leap-finetune Modal job from this project."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to the leap-finetune YAML config.",
    )
    parser.add_argument(
        "--leap-dir",
        default=str(DEFAULT_LEAP_DIR),
        help="Path to the local leap-finetune checkout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    leap_dir = Path(args.leap_dir).resolve()
    leap_src = leap_dir / "src"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not (leap_src / "leap_finetune").exists():
        raise FileNotFoundError(
            f"leap-finetune source not found: {leap_src / 'leap_finetune'}"
        )

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to .env, export it in this shell, "
            "or run huggingface-cli login before launching training."
        )

    os.environ.setdefault("LEAP_FINETUNE_DIR", str(leap_dir))
    sys.path.insert(0, str(leap_src))
    sys.argv = ["leap-finetune", str(config_path)]

    from leap_finetune import main as leap_main

    leap_main()


if __name__ == "__main__":
    main()
