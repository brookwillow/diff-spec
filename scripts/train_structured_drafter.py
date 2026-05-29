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

from scripts.train_diffusion_drafter import parse_simple_config
from src.evaluation import load_jsonl
from src.structured_drafter import SLOT_NAMES, label_sizes, load_label_space
from src.structured_model import build_structured_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train structured classification/slot drafter.")
    parser.add_argument("--config", default="configs/structured_drafter.yaml")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = parse_simple_config(Path(args.config).read_text(encoding="utf-8"))
    overrides = {
        "train_file": args.train_file,
        "label_space": args.label_space,
        "output_dir": args.output_dir,
        "model": args.model,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def split_rows(rows: list[dict[str, Any]], validation_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 and validation_ratio > 0 else 0
    if val_size == 0:
        return shuffled, []
    return shuffled[val_size:], shuffled[:val_size]


def build_training_args_kwargs(config: dict[str, Any], has_eval: bool, use_cuda: bool, use_bf16: bool, use_fp16: bool, use_mps: bool) -> dict[str, Any]:
    return {
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
        "remove_unused_columns": False,
        "save_safetensors": False,
        "eval_strategy": "steps" if has_eval else "no",
    }


def contiguous_state_dict(model: Any) -> dict[str, Any]:
    state = {}
    for name, tensor in model.state_dict().items():
        state[name] = tensor.detach().cpu().contiguous()
    return state


class StructuredDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["prompt"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "kind_labels": int(row["kind_id"]),
            "tool_labels": int(row["tool_id"]),
        }
        for slot in SLOT_NAMES:
            item[f"{slot}_labels"] = int(row["slot_ids"][slot])
        return item


def main() -> int:
    args = parse_args()
    config = load_config(args)

    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    class SafeSavingTrainer(Trainer):
        def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
            output_dir = output_dir or self.args.output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(contiguous_state_dict(self.model), Path(output_dir) / "pytorch_model.bin")
            if getattr(self.model, "config", None) is not None and hasattr(self.model.config, "save_pretrained"):
                self.model.config.save_pretrained(output_dir)

    rows = load_jsonl(config["train_file"])
    train_rows, eval_rows = split_rows(rows, float(config.get("validation_ratio", 0.02)), int(config.get("seed", 42)))
    space = load_label_space(config["label_space"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    model = build_structured_model(config["model"], label_sizes(space))

    train_dataset = StructuredDataset(train_rows, tokenizer, int(config.get("max_length", 512)))
    eval_dataset = StructuredDataset(eval_rows, tokenizer, int(config.get("max_length", 512))) if eval_rows else None

    use_cuda = torch.cuda.is_available()
    use_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    train_args_kwargs = build_training_args_kwargs(config, eval_dataset is not None, use_cuda, use_bf16, use_fp16, use_mps)
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    train_args_kwargs[eval_key] = train_args_kwargs.pop("eval_strategy")
    if "save_safetensors" not in sig:
        train_args_kwargs.pop("save_safetensors", None)
    training_args = TrainingArguments(**{k: v for k, v in train_args_kwargs.items() if k in sig})
    if args.smoke_steps > 0:
        training_args.max_steps = args.smoke_steps

    trainer = SafeSavingTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
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
    Path(config["output_dir"], "label_space.json").write_text(
        Path(config["label_space"]).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
