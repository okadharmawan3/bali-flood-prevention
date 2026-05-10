"""Finalize a native Transformers SmolVLM2 Bali flood fine-tune run.

This is a thin preset wrapper around ``scripts/finalize_finetune.py`` with the
SmolVLM2 Transformers Trainer run prefix, output name, and evaluation port.
"""

from __future__ import annotations

import sys

from finalize_finetune import main


SMOL_DEFAULTS = [
    "--run-prefix",
    "SmolVLM2-500M-Video-Instruct-transformers-bali_flood",
    "--output",
    "outputs/smolvlm2-transformers-bali-flood-Q8_0.gguf",
    "--port",
    "8084",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *SMOL_DEFAULTS, *sys.argv[1:]]
    main()
