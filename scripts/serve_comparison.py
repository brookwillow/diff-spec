#!/usr/bin/env python3
"""Dual-model comparison server: structured drafter + Qwen AR model.

Loads both models and serves a Gradio UI for side-by-side comparison.

Usage:
    python scripts/serve_comparison.py
    python scripts/serve_comparison.py --qwen-model Qwen/Qwen2.5-1.5B-Instruct --qwen-adapter outputs/qwen2_5_1_5b_tool_lora
    python scripts/serve_comparison.py --port 7860 --share
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-model comparison server.")
    # Structured drafter args
    parser.add_argument("--structured-model", default="outputs/structured_drafter")
    parser.add_argument("--structured-base", default="hfl/chinese-macbert-base")
    parser.add_argument("--structured-max-length", type=int, default=512)
    # Qwen AR model args
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--qwen-adapter", default=None)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    # Shared
    parser.add_argument("--system", default="data/system-prompt.txt")
    parser.add_argument("--tools", default="data/tools.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", default="0.0.0.0", help="Bind address. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_structured_drafter(args):
    import torch
    from transformers import AutoTokenizer

    from src.structured_drafter import label_sizes, load_label_space
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

def infer_structured(query: str, history: list[list[str]], model, tokenizer, space, schemas, max_length: int) -> dict[str, Any]:
    import torch
    from src.prepare_structured_drafter_data import render_structured_prompt
    from src.structured_drafter import render_prediction, select_ids_from_logits

    messages = _build_messages(query, history)
    prompt = render_structured_prompt(messages)

    t0 = time.perf_counter()
    encoded = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logits = output["logits"] if isinstance(output, dict) else output.logits
    example = select_ids_from_logits(logits)
    prediction = render_prediction(example, space, schemas)

    kind_probs = torch.softmax(logits["kind"], dim=-1)[0]
    tool_probs = torch.softmax(logits["tool"], dim=-1)[0]
    kind_name = space.id_to_kind.get(example.kind_id, "?")
    tool_name = space.id_to_tool.get(example.tool_id, "?")

    return {
        "prediction": prediction,
        "latency_ms": round(elapsed_ms, 1),
        "kind": kind_name,
        "kind_prob": round(kind_probs[example.kind_id].item(), 4),
        "tool": tool_name,
        "tool_prob": round(tool_probs[example.tool_id].item(), 4),
    }


def infer_qwen(query: str, history: list[list[str]], model, tokenizer, system_prompt: str, max_new_tokens: int) -> dict[str, Any]:
    import torch
    from collections.abc import Mapping

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": query})

    t0 = time.perf_counter()
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids = encoded
        attention_mask = None
    input_ids = input_ids.to(model.device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    generated = output_ids[0, input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {
        "prediction": prediction,
        "latency_ms": round(elapsed_ms, 1),
    }


def _build_messages(query: str, history: list[list[str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": query})
    return messages


def _format_output(result: dict[str, Any], label: str) -> str:
    lines = [f"**{label}**", ""]
    pred = result["prediction"]
    # Pretty-print JSON if possible
    try:
        parsed = json.loads(pred)
        lines.append(f"```json\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n```")
    except (json.JSONDecodeError, TypeError):
        lines.append(f"`{pred}`")
    lines.append(f"\n⏱ **{result['latency_ms']:.1f} ms**")
    if "kind" in result:
        lines.append(f"  Kind: {result['kind']} ({result['kind_prob']:.1%})  |  Tool: {result['tool']} ({result['tool_prob']:.1%})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(args):
    import gradio as gr
    from src.evaluation import load_tool_schemas

    print("Loading tool schemas...")
    schemas = load_tool_schemas(args.tools)
    system_prompt = Path(args.system).read_text(encoding="utf-8")

    print("Loading structured drafter...")
    sd_model, sd_tokenizer, sd_space = load_structured_drafter(args)

    print("Loading Qwen model...")
    qwen_model, qwen_tokenizer = load_qwen_model(args)

    print("Models loaded. Starting server...")

    executor = ThreadPoolExecutor(max_workers=2)

    def predict(query: str, history: list[list[str]]):
        if not query.strip():
            return "", ""

        # Run both models concurrently
        sd_future = executor.submit(
            infer_structured, query, history,
            sd_model, sd_tokenizer, sd_space, schemas, args.structured_max_length,
        )
        qwen_future = executor.submit(
            infer_qwen, query, history,
            qwen_model, qwen_tokenizer, system_prompt, args.qwen_max_new_tokens,
        )
        sd_result = sd_future.result()
        qwen_result = qwen_future.result()

        sd_out = _format_output(sd_result, "Structured Drafter")
        qwen_out = _format_output(qwen_result, "Qwen AR")
        return sd_out, qwen_out

    with gr.Blocks(title="Automotive Tool-Call Comparison") as app:
        gr.Markdown("# 🚗 Structured Drafter vs Qwen AR 对比")
        gr.Markdown("输入车载指令，同时对比两个模型的预测结果和延迟。")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot_history = gr.State([])
                query_input = gr.Textbox(
                    label="用户指令",
                    placeholder="例如：把空调温度调到26度",
                    lines=1,
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空历史")

        with gr.Row():
            sd_output = gr.Markdown(label="Structured Drafter")
            qwen_output = gr.Markdown(label="Qwen AR")

        with gr.Accordion("对话历史 (多轮)", open=False):
            history_display = gr.JSON(label="History")

        def on_submit(query, history):
            sd_out, qwen_out = predict(query, history)
            # Use the Qwen prediction as the "assistant response" for history
            # (in a real system you'd pick one or merge)
            new_history = history + [[query, ""]]
            return sd_out, qwen_out, new_history, new_history, ""

        def on_clear():
            return "", "", [], [], ""

        submit_btn.click(
            on_submit,
            inputs=[query_input, chatbot_history],
            outputs=[sd_output, qwen_output, chatbot_history, history_display, query_input],
        )
        query_input.submit(
            on_submit,
            inputs=[query_input, chatbot_history],
            outputs=[sd_output, qwen_output, chatbot_history, history_display, query_input],
        )
        clear_btn.click(
            on_clear,
            outputs=[sd_output, qwen_output, chatbot_history, history_display, query_input],
        )

    return app


def main() -> int:
    args = parse_args()
    app = build_app(args)
    app.launch(server_name=args.server_name, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
