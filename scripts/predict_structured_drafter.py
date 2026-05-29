#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator, load_jsonl, load_tool_schemas
from src.prepare_diffusion_data import render_prompt
from src.structured_drafter import load_label_space, render_prediction, select_ids_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured drafter predictions on the text eval set.")
    parser.add_argument("--model", default="outputs/structured_drafter")
    parser.add_argument("--base-model", default="hfl/chinese-macbert-base")
    parser.add_argument("--eval-file", default="data/eval_text/all.jsonl")
    parser.add_argument("--system", default="data/system-prompt.txt")
    parser.add_argument("--tools", default="data/tools.json")
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--output", default="predictions/structured_drafter_eval.jsonl")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, space) -> Any:
    import torch

    from src.structured_drafter import label_sizes
    from src.structured_model import build_structured_model

    model = build_structured_model(args.base_model, label_sizes(space))
    state_path = Path(args.model) / "pytorch_model.bin"
    safetensors_path = Path(args.model) / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path))
    else:
        state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model


def get_model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        first_param = next(model.parameters())
    except (StopIteration, AttributeError, TypeError):
        return "cpu"
    return first_param.device


def predict_one(model: Any, tokenizer: Any, prompt: str, max_length: int, space, schemas: dict[str, dict[str, Any]]) -> str:
    import torch

    encoded = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    device = get_model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    example = select_ids_from_logits(output.logits)
    return render_prediction(example, space, schemas)


def write_predictions(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    rows = load_jsonl(args.eval_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    system_prompt = Path(args.system).read_text(encoding="utf-8")
    schemas = load_tool_schemas(args.tools)
    label_space_path = args.label_space or str(Path(args.model) / "label_space.json")
    space = load_label_space(label_space_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args, space)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            messages = row.get("messages")
            if not isinstance(messages, list):
                messages = [{"role": "user", "content": str(row.get("query", ""))}]
            prompt = render_prompt(messages, system_prompt)
            prediction = predict_one(model, tokenizer, prompt, args.max_length, space, schemas)
            handle.write(
                json.dumps({"id": row.get("id"), "prediction": prediction}, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            if index % 20 == 0 or index == len(rows):
                print(f"[predict] {index}/{len(rows)}", flush=True)


def evaluate_predictions(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = Evaluator(load_tool_schemas(args.tools))
    summary = evaluator.evaluate_files_detailed(args.eval_file, args.output, limit=args.limit).as_dict()
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(text + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    write_predictions(args)
    if not args.no_evaluate:
        evaluate_predictions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
