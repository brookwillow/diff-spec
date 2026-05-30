#!/usr/bin/env python3
"""Interactive / batch inference for the structured drafter.

Usage:
    # Interactive mode (type queries, get predictions):
    python scripts/infer_structured_drafter.py

    # Batch mode (read input JSONL, write predictions):
    python scripts/infer_structured_drafter.py --input data/eval_text/all.jsonl --output predictions/structured_draft.jsonl

    # With confidence scores:
    python scripts/infer_structured_drafter.py --show-confidence

Output format matches the Qwen autoregressive model, e.g.:
    {"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}
    Reject
    请问您想操作哪个功能？
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_jsonl, load_tool_schemas
from src.prepare_structured_drafter_data import render_structured_prompt
from src.structured_drafter import (
    StructuredExample,
    StructuredLabelSpace,
    label_sizes,
    load_label_space,
    render_prediction,
    select_ids_from_logits,
    SLOT_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured drafter inference.")
    parser.add_argument("--model", default="outputs/structured_drafter", help="Trained model directory.")
    parser.add_argument("--base-model", default="hfl/chinese-macbert-base", help="Base encoder model.")
    parser.add_argument("--tools", default="data/tools.json", help="Tool schema JSON.")
    parser.add_argument("--label-space", default=None, help="Label space JSON (default: <model>/label_space.json).")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--input", default=None, help="Input JSONL for batch mode.  Each line needs 'messages' or 'query'.")
    parser.add_argument("--output", default=None, help="Output JSONL for batch mode.")
    parser.add_argument("--show-confidence", action="store_true", help="Print confidence scores.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_model(model_dir: str, base_model: str, space: StructuredLabelSpace):
    import torch
    from src.structured_model import build_structured_model

    model = build_structured_model(base_model, label_sizes(space))
    state_path = Path(model_dir) / "pytorch_model.bin"
    safetensors_path = Path(model_dir) / "model.safetensors"
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


def get_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def infer(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    space: StructuredLabelSpace,
    schemas: dict[str, dict[str, Any]],
    show_confidence: bool = False,
) -> dict[str, Any]:
    """Run inference and return prediction + optional confidence info."""
    import torch

    encoded = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    device = get_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    logits = output["logits"] if isinstance(output, dict) else output.logits
    example = select_ids_from_logits(logits)
    prediction = render_prediction(example, space, schemas)

    result: dict[str, Any] = {"prediction": prediction}
    if show_confidence:
        kind_probs = torch.softmax(logits["kind"], dim=-1)[0]
        tool_probs = torch.softmax(logits["tool"], dim=-1)[0]
        kind_name = space.id_to_kind.get(example.kind_id, "?")
        tool_name = space.id_to_tool.get(example.tool_id, "?")
        result["confidence"] = {
            "kind": kind_name,
            "kind_prob": round(kind_probs[example.kind_id].item(), 4),
            "tool": tool_name,
            "tool_prob": round(tool_probs[example.tool_id].item(), 4),
        }
    return result


def messages_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    query = row.get("query")
    if isinstance(query, str):
        return [{"role": "user", "content": query}]
    return [{"role": "user", "content": str(row.get("content", ""))}]


def run_batch(args, model, tokenizer, space, schemas) -> None:
    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    output_path = Path(args.output or "predictions/structured_draft.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            messages = messages_from_row(row)
            prompt = render_structured_prompt(messages)
            result = infer(model, tokenizer, prompt, args.max_length, space, schemas, args.show_confidence)
            entry = {"id": row.get("id"), **result}
            fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            if i % 50 == 0 or i == len(rows):
                print(f"[infer] {i}/{len(rows)}", flush=True)
    print(f"Wrote {len(rows)} predictions to {output_path}")


def run_interactive(args, model, tokenizer, space, schemas) -> None:
    print("Structured drafter interactive mode. Type a query (or 'quit' to exit).")
    print("For multi-turn, separate turns with '|||', e.g.: 打开车窗|||请问打开哪个？|||主驾的")
    print()
    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() in ("quit", "exit", "q"):
            break
        # Support multi-turn via ||| separator
        turns = [t.strip() for t in raw.split("|||") if t.strip()]
        messages: list[dict[str, str]] = []
        for idx, turn in enumerate(turns):
            role = "user" if idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})

        prompt = render_structured_prompt(messages)
        result = infer(model, tokenizer, prompt, args.max_length, space, schemas, show_confidence=True)
        print(f"  Output: {result['prediction']}")
        if "confidence" in result:
            c = result["confidence"]
            print(f"  Kind: {c['kind']} ({c['kind_prob']:.1%})  Tool: {c['tool']} ({c['tool_prob']:.1%})")
        print()


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    label_space_path = args.label_space or str(Path(args.model) / "label_space.json")
    space = load_label_space(label_space_path)
    schemas = load_tool_schemas(args.tools)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.base_model, space)
    print(f"Model loaded from {args.model} (device: {get_device(model)})")

    if args.input:
        run_batch(args, model, tokenizer, space, schemas)
    else:
        run_interactive(args, model, tokenizer, space, schemas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
