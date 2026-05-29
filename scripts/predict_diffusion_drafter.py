#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_drafter import MaskedDrafterConfig, build_prediction_inputs, trim_decoded_prediction
from src.evaluation import Evaluator, load_jsonl, load_tool_schemas
from src.prepare_diffusion_data import render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run masked-diffusion drafter predictions on the text eval set.")
    parser.add_argument("--model", default="outputs/diffusion_drafter", help="Trained drafter directory.")
    parser.add_argument("--eval-file", default="data/eval_text/all.jsonl")
    parser.add_argument("--system", default="data/system-prompt.txt")
    parser.add_argument("--tools", default="data/tools.json")
    parser.add_argument("--output", default="predictions/diffusion_drafter_eval.jsonl")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def load_drafter_config(model_dir: str, max_target_tokens: int | None) -> MaskedDrafterConfig:
    path = Path(model_dir) / "drafter_config.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = MaskedDrafterConfig(**payload)
    else:
        config = MaskedDrafterConfig()
    if max_target_tokens is not None:
        config = MaskedDrafterConfig(max_length=config.max_length, max_target_tokens=max_target_tokens)
    return config


def load_model_and_tokenizer(model_dir: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tokenizer


def predict_one(model: Any, tokenizer: Any, prompt: str, config: MaskedDrafterConfig) -> str:
    import torch

    encoded = build_prediction_inputs(tokenizer, prompt, config)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=model.device)
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=model.device)
    mask_positions = encoded["mask_positions"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
    predicted_ids = logits[mask_positions].argmax(dim=-1).detach().cpu().tolist()
    text = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    return trim_decoded_prediction(text)


def write_predictions(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.eval_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    system_prompt = Path(args.system).read_text(encoding="utf-8")
    config = load_drafter_config(args.model, args.max_target_tokens)
    model, tokenizer = load_model_and_tokenizer(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            messages = row.get("messages")
            if not isinstance(messages, list):
                messages = [{"role": "user", "content": str(row.get("query", ""))}]
            prompt = render_prompt(messages, system_prompt)
            prediction = predict_one(model, tokenizer, prompt, config)
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
