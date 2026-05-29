#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator, load_jsonl, load_tool_schemas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen SFT/LoRA predictions on the text eval set.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model id or local path.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter/checkpoint path.")
    parser.add_argument("--eval-file", default="data/eval_text/all.jsonl", help="Evaluation JSONL file.")
    parser.add_argument("--system", default="data/system-prompt.txt", help="System prompt path.")
    parser.add_argument("--tools", default="data/tools.json", help="Tool schema JSON file for metrics.")
    parser.add_argument("--output", default="predictions/qwen_sft_eval.jsonl", help="Prediction JSONL path.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick checks.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--no-evaluate", action="store_true", help="Only write predictions, skip metrics.")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    device_map: str | dict[str, int] = args.device
    if args.device == "auto":
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("peft is required when --adapter is provided. Install with: pip install peft") from exc
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


def prompt_messages(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt = [message for message in messages if isinstance(message, dict)]
        if prompt and prompt[-1].get("role") == "assistant":
            prompt = prompt[:-1]
        if prompt and prompt[0].get("role") == "system":
            prompt[0] = {"role": "system", "content": system_prompt}
        else:
            prompt.insert(0, {"role": "system", "content": system_prompt})
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt if "role" in m and "content" in m]
    query = row.get("query")
    if not isinstance(query, str):
        raise ValueError("eval row must contain messages or query")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]


def generate_prediction(model, tokenizer, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_predictions(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.eval_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    system_prompt = Path(args.system).read_text(encoding="utf-8")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            prediction = generate_prediction(model, tokenizer, prompt_messages(row, system_prompt), args)
            payload = {
                "id": row.get("id"),
                "prediction": prediction,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
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
