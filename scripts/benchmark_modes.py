#!/usr/bin/env python3
"""Benchmark all three inference modes on the eval set.

Runs Structured Drafter, Qwen AR, and Speculative Decoding on the eval data,
measures accuracy and latency, and outputs a comparison table.

Usage:
    python scripts/benchmark_modes.py
    python scripts/benchmark_modes.py --limit 100
    python scripts/benchmark_modes.py --output outputs/benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator, load_jsonl, load_tool_schemas
from src.prepare_structured_drafter_data import render_structured_prompt
from src.structured_drafter import (
    label_sizes,
    load_label_space,
    render_prediction,
    select_ids_from_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 3 inference modes.")
    parser.add_argument("--structured-model", default="outputs/structured_drafter")
    parser.add_argument("--structured-base", default="hfl/chinese-macbert-base")
    parser.add_argument("--structured-max-length", type=int, default=512)
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--qwen-adapter", default=None)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--eval-file", default="data/eval_text/all.jsonl")
    parser.add_argument("--system", default="data/system-prompt.txt")
    parser.add_argument("--tools", default="data/tools.json")
    parser.add_argument("--output", default="outputs/benchmark_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_structured_drafter(args):
    import torch
    from transformers import AutoTokenizer
    from src.structured_model import build_structured_model

    label_space_path = str(Path(args.structured_model) / "label_space.json")
    space = load_label_space(label_space_path)
    tokenizer = AutoTokenizer.from_pretrained(args.structured_model, trust_remote_code=True)
    model = build_structured_model(args.structured_base, label_sizes(space))
    state_path = Path(args.structured_model) / "pytorch_model.bin"
    safetensors_path = Path(args.structured_model) / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path))
    else:
        state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer, space


def load_qwen_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.qwen_dtype]
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, device_map=device_map, torch_dtype=dtype, trust_remote_code=True,
    )
    if args.qwen_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.qwen_adapter)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def predict_structured(row, sd_model, sd_tokenizer, sd_space, schemas, max_length):
    import torch

    messages = row.get("messages", [{"role": "user", "content": row.get("query", "")}])
    prompt = render_structured_prompt(messages)

    t0 = time.perf_counter()
    encoded = sd_tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    device = next(sd_model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = sd_model(**encoded)
    logits = output["logits"] if isinstance(output, dict) else output.logits
    example = select_ids_from_logits(logits)
    prediction = render_prediction(example, sd_space, schemas)
    elapsed = (time.perf_counter() - t0) * 1000
    return prediction, elapsed


def predict_qwen(row, qwen_model, qwen_tokenizer, system_prompt, max_new_tokens):
    import torch

    messages = row.get("messages", [{"role": "user", "content": row.get("query", "")}])
    prompt_msgs = [m for m in messages if isinstance(m, dict)]
    if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
        prompt_msgs = prompt_msgs[:-1]
    if prompt_msgs and prompt_msgs[0].get("role") == "system":
        prompt_msgs[0] = {"role": "system", "content": system_prompt}
    else:
        prompt_msgs.insert(0, {"role": "system", "content": system_prompt})

    t0 = time.perf_counter()
    encoded = qwen_tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    input_ids = input_ids.to(qwen_model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[1]:]
    prediction = qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()
    elapsed = (time.perf_counter() - t0) * 1000
    return prediction, elapsed


def predict_speculative(row, sd_model, sd_tokenizer, sd_space, schemas, sd_max_length,
                        qwen_model, qwen_tokenizer, system_prompt, max_new_tokens):
    import torch

    # Draft phase
    messages = row.get("messages", [{"role": "user", "content": row.get("query", "")}])
    prompt = render_structured_prompt(messages)

    t0 = time.perf_counter()
    encoded = sd_tokenizer(prompt, truncation=True, max_length=sd_max_length, padding="max_length", return_tensors="pt")
    device_sd = next(sd_model.parameters()).device
    encoded = {k: v.to(device_sd) for k, v in encoded.items()}
    with torch.no_grad():
        sd_output = sd_model(**encoded)
    logits = sd_output["logits"] if isinstance(sd_output, dict) else sd_output.logits
    example = select_ids_from_logits(logits)
    draft_text = render_prediction(example, sd_space, schemas)
    draft_ms = (time.perf_counter() - t0) * 1000

    # Verify phase: build Qwen prompt
    prompt_msgs = [m for m in messages if isinstance(m, dict)]
    if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
        prompt_msgs = prompt_msgs[:-1]
    if prompt_msgs and prompt_msgs[0].get("role") == "system":
        prompt_msgs[0] = {"role": "system", "content": system_prompt}
    else:
        prompt_msgs.insert(0, {"role": "system", "content": system_prompt})

    prompt_encoded = qwen_tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(prompt_encoded, Mapping):
        prompt_ids = prompt_encoded["input_ids"]
    else:
        prompt_ids = prompt_encoded
    prompt_ids = prompt_ids.to(qwen_model.device)
    prompt_len = prompt_ids.shape[1]

    draft_token_ids = qwen_tokenizer.encode(draft_text, add_special_tokens=False)
    if not draft_token_ids:
        draft_token_ids = []

    draft_tensor = torch.tensor([draft_token_ids], device=qwen_model.device)
    input_with_draft = torch.cat([prompt_ids, draft_tensor], dim=1)

    t_verify = time.perf_counter()
    with torch.no_grad():
        outputs = qwen_model(input_ids=input_with_draft)
    verify_logits = outputs.logits
    verify_ms = (time.perf_counter() - t_verify) * 1000

    # Token-by-token comparison
    accepted_tokens = 0
    n_draft = len(draft_token_ids)
    for i in range(n_draft):
        logit_pos = prompt_len - 1 + i
        qwen_pred = verify_logits[0, logit_pos].argmax(dim=-1).item()
        if qwen_pred == draft_token_ids[i]:
            accepted_tokens += 1
        else:
            break

    # Build final output
    if accepted_tokens == n_draft and n_draft > 0:
        final_prediction = draft_text
        accept_status = "fully_accepted"
    elif accepted_tokens > 0 or n_draft == 0:
        accepted_ids = draft_token_ids[:accepted_tokens]
        if accepted_tokens < n_draft:
            diverge_token = verify_logits[0, prompt_len - 1 + accepted_tokens].argmax(dim=-1).item()
            continuation_start = torch.cat([prompt_ids, torch.tensor([accepted_ids + [diverge_token]], device=qwen_model.device)], dim=1)
        else:
            continuation_start = torch.cat([prompt_ids, torch.tensor([accepted_ids], device=qwen_model.device)], dim=1)
        remaining_budget = max_new_tokens - accepted_tokens - (1 if accepted_tokens < n_draft else 0)
        if remaining_budget > 0:
            with torch.no_grad():
                attention_mask = torch.ones_like(continuation_start)
                gen_output = qwen_model.generate(
                    input_ids=continuation_start,
                    attention_mask=attention_mask,
                    max_new_tokens=remaining_budget,
                    do_sample=False,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                    eos_token_id=qwen_tokenizer.eos_token_id,
                )
            all_generated = gen_output[0, prompt_len:]
        else:
            all_generated = continuation_start[0, prompt_len:]
        final_prediction = qwen_tokenizer.decode(all_generated, skip_special_tokens=True).strip()
        accept_status = "partial"
    else:
        with torch.no_grad():
            attention_mask = torch.ones_like(prompt_ids)
            gen_output = qwen_model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=qwen_tokenizer.eos_token_id,
                eos_token_id=qwen_tokenizer.eos_token_id,
            )
        generated = gen_output[0, prompt_len:]
        final_prediction = qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()
        accept_status = "rejected"

    total_ms = (time.perf_counter() - t0) * 1000
    return final_prediction, total_ms, {
        "draft_ms": draft_ms,
        "verify_ms": verify_ms,
        "accepted_tokens": accepted_tokens,
        "draft_tokens": n_draft,
        "accept_status": accept_status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    import torch

    print("Loading models...")
    sd_model, sd_tokenizer, sd_space = load_structured_drafter(args)
    qwen_model, qwen_tokenizer = load_qwen_model(args)
    schemas = load_tool_schemas(args.tools)
    system_prompt = Path(args.system).read_text(encoding="utf-8")

    rows = load_jsonl(args.eval_file)
    if args.limit is not None:
        rows = rows[:args.limit]
    total = len(rows)
    print(f"Evaluating {total} samples...")

    # Warmup
    if args.warmup > 0 and rows:
        print(f"Warming up ({args.warmup} iterations)...")
        for _ in range(args.warmup):
            predict_structured(rows[0], sd_model, sd_tokenizer, sd_space, schemas, args.structured_max_length)
            predict_qwen(rows[0], qwen_model, qwen_tokenizer, system_prompt, args.qwen_max_new_tokens)

    # Collect predictions and latencies
    sd_preds, qwen_preds, spec_preds = [], [], []
    sd_times, qwen_times, spec_times = [], [], []
    spec_details: list[dict] = []

    for i, row in enumerate(rows, 1):
        # Structured Drafter
        pred_sd, t_sd = predict_structured(row, sd_model, sd_tokenizer, sd_space, schemas, args.structured_max_length)
        sd_preds.append(pred_sd)
        sd_times.append(t_sd)

        # Qwen AR
        pred_qwen, t_qwen = predict_qwen(row, qwen_model, qwen_tokenizer, system_prompt, args.qwen_max_new_tokens)
        qwen_preds.append(pred_qwen)
        qwen_times.append(t_qwen)

        # Speculative Decoding
        pred_spec, t_spec, detail = predict_speculative(
            row, sd_model, sd_tokenizer, sd_space, schemas, args.structured_max_length,
            qwen_model, qwen_tokenizer, system_prompt, args.qwen_max_new_tokens,
        )
        spec_preds.append(pred_spec)
        spec_times.append(t_spec)
        spec_details.append(detail)

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] SD={t_sd:.0f}ms  Qwen={t_qwen:.0f}ms  Spec={t_spec:.0f}ms ({detail['accept_status']})", flush=True)

    # Write prediction files
    pred_dir = Path("predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, preds in [("structured", sd_preds), ("qwen_ar", qwen_preds), ("speculative", spec_preds)]:
        path = pred_dir / f"benchmark_{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row, pred in zip(rows, preds):
                fh.write(json.dumps({"id": row.get("id"), "prediction": pred}, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Evaluate accuracy
    evaluator = Evaluator(schemas)
    results = {}
    for name, preds in [("structured_drafter", sd_preds), ("qwen_ar", qwen_preds), ("speculative", spec_preds)]:
        path = pred_dir / f"benchmark_{name}.jsonl"
        summary = evaluator.evaluate_files_detailed(args.eval_file, str(path), limit=args.limit).as_dict()
        results[name] = summary

    # Latency stats
    def latency_stats(times: list[float]) -> dict[str, float]:
        arr = np.array(times)
        return {
            "mean_ms": round(float(arr.mean()), 1),
            "p50_ms": round(float(np.percentile(arr, 50)), 1),
            "p90_ms": round(float(np.percentile(arr, 90)), 1),
            "p99_ms": round(float(np.percentile(arr, 99)), 1),
            "min_ms": round(float(arr.min()), 1),
            "max_ms": round(float(arr.max()), 1),
        }

    # Speculative decoding stats
    accept_statuses = [d["accept_status"] for d in spec_details]
    fully_accepted = sum(1 for s in accept_statuses if s == "fully_accepted")
    partial = sum(1 for s in accept_statuses if s == "partial")
    rejected = sum(1 for s in accept_statuses if s == "rejected")
    total_draft_tokens = sum(d["draft_tokens"] for d in spec_details)
    total_accepted_tokens = sum(d["accepted_tokens"] for d in spec_details)

    # Build final report
    report = {
        "eval_file": args.eval_file,
        "total_samples": total,
        "modes": {
            "structured_drafter": {
                "accuracy": {
                    "exact_match_rate": results["structured_drafter"]["exact_match_rate"],
                    "classification_accuracy": results["structured_drafter"]["classification_accuracy"],
                    "tool_selection_accuracy": results["structured_drafter"].get("tool_selection_accuracy", 0),
                    "parameter_fill_accuracy": results["structured_drafter"].get("parameter_fill_accuracy", 0),
                    "schema_valid_rate": results["structured_drafter"]["schema_valid_rate"],
                },
                "latency": latency_stats(sd_times),
            },
            "qwen_ar": {
                "accuracy": {
                    "exact_match_rate": results["qwen_ar"]["exact_match_rate"],
                    "classification_accuracy": results["qwen_ar"]["classification_accuracy"],
                    "tool_selection_accuracy": results["qwen_ar"].get("tool_selection_accuracy", 0),
                    "parameter_fill_accuracy": results["qwen_ar"].get("parameter_fill_accuracy", 0),
                    "schema_valid_rate": results["qwen_ar"]["schema_valid_rate"],
                },
                "latency": latency_stats(qwen_times),
            },
            "speculative": {
                "accuracy": {
                    "exact_match_rate": results["speculative"]["exact_match_rate"],
                    "classification_accuracy": results["speculative"]["classification_accuracy"],
                    "tool_selection_accuracy": results["speculative"].get("tool_selection_accuracy", 0),
                    "parameter_fill_accuracy": results["speculative"].get("parameter_fill_accuracy", 0),
                    "schema_valid_rate": results["speculative"]["schema_valid_rate"],
                },
                "latency": latency_stats(spec_times),
                "speculative_stats": {
                    "fully_accepted": fully_accepted,
                    "fully_accepted_rate": round(fully_accepted / total, 4) if total else 0,
                    "partial": partial,
                    "rejected": rejected,
                    "token_accept_rate": round(total_accepted_tokens / total_draft_tokens, 4) if total_draft_tokens else 0,
                    "avg_draft_tokens": round(total_draft_tokens / total, 1) if total else 0,
                    "avg_accepted_tokens": round(total_accepted_tokens / total, 1) if total else 0,
                },
            },
        },
        "speedup": {
            "spec_vs_qwen": round(np.mean(qwen_times) / np.mean(spec_times), 2) if np.mean(spec_times) > 0 else 0,
            "sd_vs_qwen": round(np.mean(qwen_times) / np.mean(sd_times), 2) if np.mean(sd_times) > 0 else 0,
        },
    }

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'BENCHMARK RESULTS':^80}")
    print("=" * 80)
    print(f"{'Metric':<30} {'Structured':<16} {'Qwen AR':<16} {'Speculative':<16}")
    print("-" * 80)
    for metric in ["exact_match_rate", "classification_accuracy", "tool_selection_accuracy", "parameter_fill_accuracy", "schema_valid_rate"]:
        sd_v = report["modes"]["structured_drafter"]["accuracy"][metric]
        qwen_v = report["modes"]["qwen_ar"]["accuracy"][metric]
        spec_v = report["modes"]["speculative"]["accuracy"][metric]
        print(f"{metric:<30} {sd_v:<16.4f} {qwen_v:<16.4f} {spec_v:<16.4f}")
    print("-" * 80)
    for metric in ["mean_ms", "p50_ms", "p90_ms", "p99_ms"]:
        sd_v = report["modes"]["structured_drafter"]["latency"][metric]
        qwen_v = report["modes"]["qwen_ar"]["latency"][metric]
        spec_v = report["modes"]["speculative"]["latency"][metric]
        print(f"{metric:<30} {sd_v:<16.1f} {qwen_v:<16.1f} {spec_v:<16.1f}")
    print("-" * 80)
    ss = report["modes"]["speculative"]["speculative_stats"]
    print(f"{'Spec: fully accepted':<30} {'':<16} {'':<16} {ss['fully_accepted_rate']:.1%} ({ss['fully_accepted']}/{total})")
    print(f"{'Spec: token accept rate':<30} {'':<16} {'':<16} {ss['token_accept_rate']:.1%}")
    print(f"{'Spec: avg tokens accepted':<30} {'':<16} {'':<16} {ss['avg_accepted_tokens']:.1f}/{ss['avg_draft_tokens']:.1f}")
    print("-" * 80)
    print(f"{'Speedup vs Qwen AR':<30} {report['speedup']['sd_vs_qwen']:.1f}x{'':<11} {'1.0x':<16} {report['speedup']['spec_vs_qwen']:.1f}x")
    print("=" * 80)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nFull results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
