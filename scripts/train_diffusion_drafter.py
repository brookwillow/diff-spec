#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_drafter import DiffusionJsonDataset, MaskedDrafterConfig
from src.evaluation import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked-diffusion JSON drafter.")
    parser.add_argument("--config", default="configs/diffusion_drafter.yaml")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--smoke-steps", type=int, default=0, help="Run only N steps for sanity checks.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = parse_simple_config(Path(args.config).read_text(encoding="utf-8"))
    for key, value in {"train_file": args.train_file, "output_dir": args.output_dir, "model": args.model}.items():
        if value is not None:
            config[key] = value
    return config


def parse_simple_config(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"unsupported config line: {line}")
        key, value = stripped.split(":", 1)
        config[key.strip()] = _parse_scalar(value.strip())
    return config


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def split_rows(rows: list[dict[str, Any]], validation_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 and validation_ratio > 0 else 0
    if val_size == 0:
        return shuffled, []
    return shuffled[val_size:], shuffled[:val_size]


def main() -> int:
    args = parse_args()
    config = load_config(args)

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

    rows = load_jsonl(config["train_file"])
    train_rows, eval_rows = split_rows(rows, float(config.get("validation_ratio", 0.02)), int(config.get("seed", 42)))
    drafter_config = MaskedDrafterConfig(
        max_length=int(config.get("max_length", 512)),
        max_target_tokens=int(config.get("max_target_tokens", 128)),
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(config["model"], trust_remote_code=True)
    train_dataset = DiffusionJsonDataset(train_rows, tokenizer, drafter_config)
    eval_dataset = DiffusionJsonDataset(eval_rows, tokenizer, drafter_config) if eval_rows else None

    use_cuda = torch.cuda.is_available()
    use_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    train_args_kwargs = {
        "output_dir": config["output_dir"],
        "num_train_epochs": float(config.get("num_train_epochs", 5)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "logging_steps": int(config.get("logging_steps", 20)),
        "eval_steps": int(config.get("eval_steps", 200)),
        "save_strategy": "steps",
        "save_steps": int(config.get("save_steps", 200)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "report_to": "none",
        "seed": int(config.get("seed", 42)),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "use_mps_device": use_mps,
    }
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    train_args_kwargs[eval_key] = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(**{k: v for k, v in train_args_kwargs.items() if k in sig})
    if args.smoke_steps > 0:
        training_args.max_steps = args.smoke_steps

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print(
        json.dumps(
            {
                "model": config["model"],
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "output_dir": config["output_dir"],
                "device": "cuda" if use_cuda else "mps" if use_mps else "cpu",
            },
            ensure_ascii=False,
        )
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    Path(config["output_dir"]).joinpath("drafter_config.json").write_text(
        json.dumps(drafter_config.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
