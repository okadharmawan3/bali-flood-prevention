"""Fine-tune SmolVLM2 with native Hugging Face Trainer on Modal.

This is the SmolVLM2 lane for Bali flood prevention. It intentionally avoids
leap-finetune/Ray and follows the standard Transformers Trainer pattern used by
the Hugging Face SmolVLM examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_MOUNT = "/outputs"
DEFAULT_VOLUME = "bali-flood-prevention"
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_REMOTE_TRAIN_JSONL = f"{REMOTE_MOUNT}/data/bali-flood/bali_flood_train.jsonl"
DEFAULT_REMOTE_IMAGE_ROOT = f"{REMOTE_MOUNT}/data/bali-flood/images"
DEFAULT_APP_NAME = "bali-flood-prevention-smolvlm-transformers"


class JsonlDataset:
    def __init__(self, path: Path, *, limit: int | None = None):
        self.path = path
        self.rows = self._load(path, limit=limit)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], path: Path) -> "JsonlDataset":
        dataset = cls.__new__(cls)
        dataset.path = path
        dataset.rows = rows
        return dataset

    @staticmethod
    def _load(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def split_train_validation(
    rows: list[dict[str, Any]],
    *,
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0 <= validation_split < 1):
        raise ValueError("--validation-split must be >= 0 and < 1")
    if validation_split == 0:
        return rows, []

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    shuffled = [rows[index] for index in indices]

    eval_count = max(1, int(len(shuffled) * validation_split))
    train_count = len(shuffled) - eval_count
    if train_count < 1:
        raise ValueError(
            f"Not enough samples ({len(shuffled)}) for validation_split={validation_split}"
        )
    return shuffled[:train_count], shuffled[train_count:]


def compute_leap_style_max_steps(
    train_rows: int,
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
) -> int:
    steps_per_epoch = math.ceil(train_rows / batch_size)
    return max(1, int(steps_per_epoch * epochs) // gradient_accumulation_steps)


def resolve_media_path(value: str, image_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else image_root / path.name


def materialize_messages(row: dict[str, Any], image_root: Path) -> list[dict[str, Any]]:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("training row is missing messages[]")

    messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise ValueError("message must be an object")
        role = str(raw_message.get("role", ""))
        raw_content = raw_message.get("content", [])
        if isinstance(raw_content, str):
            content: list[dict[str, Any]] | str = raw_content
        elif isinstance(raw_content, list):
            content = []
            for item in raw_content:
                if not isinstance(item, dict):
                    raise ValueError("message content item must be an object")
                item_type = item.get("type")
                if item_type == "image":
                    media_value = item.get("path") or item.get("image")
                    if not media_value:
                        raise ValueError("image content item is missing image/path")
                    content.append(
                        {
                            "type": "image",
                            "path": str(resolve_media_path(str(media_value), image_root)),
                        }
                    )
                elif item_type == "text":
                    content.append({"type": "text", "text": str(item.get("text", ""))})
                else:
                    content.append(dict(item))
        else:
            raise ValueError("message content must be a string or list")
        messages.append({"role": role, "content": content})
    return messages


def find_token_id(tokenizer: Any, candidates: tuple[str, ...]) -> int | None:
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            return token_id
    return None


def pad_tensor_list(tensors: list[Any], padding_value: int | float = 0) -> Any:
    import torch

    max_shape = [max(tensor.shape[dim] for tensor in tensors) for dim in range(tensors[0].dim())]
    output_shape = (len(tensors), *max_shape)
    padded = tensors[0].new_full(output_shape, padding_value)
    for index, tensor in enumerate(tensors):
        slices = (index, *[slice(0, size) for size in tensor.shape])
        padded[slices] = tensor
    return padded


def squeeze_single_batch(tensor: Any) -> Any:
    if getattr(tensor, "dim", lambda: 0)() > 0 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


class SmolVlmCollator:
    def __init__(
        self,
        processor: Any,
        image_root: Path,
        *,
        model_dtype: Any,
        mask_prompt: bool = False,
    ):
        self.processor = processor
        self.image_root = image_root
        self.model_dtype = model_dtype
        self.mask_prompt = mask_prompt
        self.image_token_id = find_token_id(
            processor.tokenizer,
            ("<image>", "<fake_token_around_image>", "<image_soft_token>"),
        )
        self.assistant_marker_ids = processor.tokenizer.encode(
            "Assistant:", add_special_tokens=False
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        instances = []
        for example in examples:
            messages = materialize_messages(example, self.image_root)
            instance = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            for key, value in list(instance.items()):
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    instance[key] = value.to(self.model_dtype)
            instances.append(instance)

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        input_ids = pad_sequence(
            [inst["input_ids"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [inst["attention_mask"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [inst["input_ids"].squeeze(0).clone() for inst in instances],
            batch_first=True,
            padding_value=-100,
        )
        labels[labels == pad_token_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        if self.mask_prompt:
            self._mask_prompt_tokens(labels, input_ids)

        output: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        excluded = {"input_ids", "attention_mask"}
        extra_keys = sorted({key for inst in instances for key in inst if key not in excluded})
        for key in extra_keys:
            tensors = [
                squeeze_single_batch(inst[key])
                for inst in instances
                if key in inst and torch.is_tensor(inst[key])
            ]
            if len(tensors) != len(instances):
                continue
            output[key] = pad_tensor_list(tensors, padding_value=0)

        return output

    def _mask_prompt_tokens(self, labels: Any, input_ids: Any) -> None:
        marker = self.assistant_marker_ids
        if not marker:
            return
        for row_index in range(input_ids.shape[0]):
            ids = input_ids[row_index].tolist()
            start = find_last_subsequence(ids, marker)
            if start < 0:
                continue
            labels[row_index, : start + len(marker)] = -100


def find_last_subsequence(values: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    for index in range(len(values) - len(pattern), -1, -1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def train_local(args: argparse.Namespace) -> Path:
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    train_jsonl = Path(args.train_jsonl)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {train_jsonl}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = JsonlDataset(train_jsonl, limit=args.limit)
    if len(full_dataset) == 0:
        raise ValueError("No training rows loaded")
    train_rows, validation_rows = split_train_validation(
        full_dataset.rows,
        validation_split=args.validation_split,
        seed=args.validation_seed,
    )
    train_dataset = JsonlDataset.from_rows(train_rows, train_jsonl)
    eval_dataset = (
        JsonlDataset.from_rows(validation_rows, train_jsonl)
        if validation_rows
        else None
    )

    print(f"Loaded {len(full_dataset)} rows from {train_jsonl}")
    print(
        "Dataset ready: "
        f"train={len(train_dataset)}, validation={len(validation_rows)}, "
        f"validation_split={args.validation_split}"
    )
    print(f"Image root: {image_root}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        print("Dry run only; no model loaded and no training started.")
        return output_dir

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    collator = SmolVlmCollator(
        processor,
        image_root,
        model_dtype=model.dtype,
        mask_prompt=args.mask_prompt,
    )
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = compute_leap_style_max_steps(
            len(train_dataset),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
        )
    print(f"Training max_steps: {max_steps}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        weight_decay=args.weight_decay,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_pin_memory=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )

    metadata = {
        "model_id": args.model_id,
        "train_jsonl": str(train_jsonl),
        "image_root": str(image_root),
        "epochs": args.epochs,
        "max_steps": max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "validation_split": args.validation_split,
        "validation_seed": args.validation_seed,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_rows),
        "eval_strategy": args.eval_strategy if eval_dataset is not None else "no",
        "eval_steps": args.eval_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "mask_prompt": args.mask_prompt,
        "rows": len(full_dataset),
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    final_dir = output_dir / f"final-global_step{trainer.state.global_step}"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    (final_dir / "train_meta.json").write_text(
        json.dumps({**metadata, "global_step": trainer.state.global_step}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"Final checkpoint saved: {final_dir}")
    return final_dir


def default_output_dir(model_id: str, *, remote: bool) -> str:
    slug = model_id.split("/")[-1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{slug}-transformers-bali_flood-{stamp}"
    if remote:
        return f"{REMOTE_MOUNT}/{base}"
    return str(PROJECT_ROOT / "outputs" / base)


def remote_training_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--local",
        "--model-id",
        args.model_id,
        "--train-jsonl",
        args.train_jsonl,
        "--image-root",
        args.image_root,
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps) if args.max_steps is not None else "auto",
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--lr-scheduler-type",
        args.lr_scheduler_type,
        "--logging-steps",
        str(args.logging_steps),
        "--validation-split",
        str(args.validation_split),
        "--validation-seed",
        str(args.validation_seed),
        "--eval-strategy",
        args.eval_strategy,
        "--eval-steps",
        str(args.eval_steps),
        "--save-strategy",
        args.save_strategy,
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--weight-decay",
        str(args.weight_decay),
        "--optim",
        args.optim,
        "--attn-implementation",
        args.attn_implementation,
        "--report-to",
        args.report_to,
        "--seed",
        str(args.seed),
        "--dataloader-num-workers",
        str(args.dataloader_num_workers),
    ]
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    if args.dry_run:
        values.append("--dry-run")
    if args.mask_prompt:
        values.append("--mask-prompt")
    if not args.bf16:
        values.append("--no-bf16")
    if not args.gradient_checkpointing:
        values.append("--no-gradient-checkpointing")
    return values


def launch_modal(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Add it to .env or this shell.")

    import modal

    app = modal.App(args.app_name)
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)

    pip_deps = [
        "accelerate>=1.0",
        "av>=12.0",
        "hf_transfer>=0.1.8",
        "num2words>=0.5",
        "pillow>=10.0",
        "safetensors>=0.4",
        "tensorboard>=2.17",
        "torch>=2.6",
        "torchvision>=0.21",
        "transformers>=4.56,<5",
    ]
    pip_deps.extend(args.extra_pip or [])

    image = (
        modal.Image.from_registry(args.base_image, add_python="3.12")
        .pip_install(*pip_deps)
        .add_local_file(__file__, "/app/train_smolvlm_transformers_modal.py", copy=True)
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
    )

    @app.function(
        image=image,
        gpu=args.gpu,
        timeout=args.timeout,
        volumes={REMOTE_MOUNT: volume},
        secrets=[modal.Secret.from_local_environ(env_keys=["HF_TOKEN"])],
        serialized=True,
    )
    def train_remote(cli_args: list[str]) -> None:
        cmd = [sys.executable, "/app/train_smolvlm_transformers_modal.py", *cli_args]
        env = {
            **os.environ,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        subprocess.run(cmd, check=True, env=env)
        volume.commit()

    cli_args = remote_training_args(args)
    print("Submitting native Transformers Trainer SmolVLM2 job to Modal...")
    print(f"Output will be written under: {args.output_dir}")
    with modal.enable_output():
        with app.run():
            if args.detach:
                call = train_remote.spawn(cli_args)
                print(f"Modal job submitted. Function call ID: {call.object_id}")
            else:
                train_remote.remote(cli_args)


def parse_max_steps(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-steps must be >= 1 or 'auto'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM2 on Bali flood data with Transformers Trainer."
    )
    parser.add_argument("--local", action="store_true", help="Train in this process.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-jsonl", default=DEFAULT_REMOTE_TRAIN_JSONL)
    parser.add_argument("--image-root", default=DEFAULT_REMOTE_IMAGE_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=parse_max_steps,
        default=None,
        help=(
            "Optimizer steps. Default 'auto' uses leap-finetune-style math, "
            "which gives 256 for 1368 train rows, batch 2, accum 8, epochs 3."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--mask-prompt", action="store_true")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--gpu", default="H100:1")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--base-image", default="nvidia/cuda:12.8.0-devel-ubuntu22.04")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--extra-pip", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_id, remote=not args.local)

    if args.local:
        train_local(args)
    else:
        launch_modal(args)


if __name__ == "__main__":
    main()
