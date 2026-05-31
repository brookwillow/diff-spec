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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-model comparison server.")
    # Structured drafter args
    parser.add_argument("--structured-model", default="outputs/structured_drafter")
    parser.add_argument("--structured-base", default="hfl/chinese-macbert-base")
    parser.add_argument("--structured-max-length", type=int, default=512)
    # Diffusion drafter args
    parser.add_argument("--diffusion-model", default="outputs/diffusion_drafter")
    parser.add_argument("--no-diffusion", action="store_true", help="Skip diffusion drafter.")
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


def load_diffusion_drafter(args):
    import json as json_mod
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from src.diffusion_drafter import MaskedDrafterConfig

    model_dir = Path(args.diffusion_model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), trust_remote_code=True)
    config_path = model_dir / "drafter_config.json"
    if config_path.exists():
        payload = json_mod.loads(config_path.read_text(encoding="utf-8"))
        config = MaskedDrafterConfig(**payload)
    else:
        config = MaskedDrafterConfig()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer, config


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


def infer_diffusion(query: str, history: list[list[str]], model, tokenizer, config, schemas) -> dict[str, Any]:
    """Run diffusion drafter inference."""
    import torch
    from src.diffusion_drafter import build_prediction_inputs, restore_tool_case, trim_decoded_prediction
    from src.prepare_diffusion_data import render_diffusion_prompt

    messages = _build_messages(query, history)
    prompt = render_diffusion_prompt(messages)

    t0 = time.perf_counter()
    encoded = build_prediction_inputs(tokenizer, prompt, config)
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=device)
    mask_positions = encoded["mask_positions"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
    predicted_ids = logits[mask_positions].argmax(dim=-1).detach().cpu().tolist()
    raw_text = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    trimmed = trim_decoded_prediction(raw_text)
    prediction = restore_tool_case(trimmed, schemas)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  [Diffusion] {elapsed_ms:.1f}ms", flush=True)
    print(f"    raw decode : {raw_text[:200]}", flush=True)
    print(f"    trimmed    : {trimmed[:200]}", flush=True)
    if trimmed != prediction:
        print(f"    case fixed : {prediction[:200]}", flush=True)

    return {
        "prediction": prediction,
        "latency_ms": round(elapsed_ms, 1),
    }


def infer_speculative(
    query: str,
    history: list[list[str]],
    draft_text: str,
    draft_ms: float,
    qwen_model,
    qwen_tokenizer,
    system_prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Speculative decoding: verify a pre-computed BERT draft with Qwen."""
    import torch
    from collections.abc import Mapping

    t0 = time.perf_counter()

    # Step 1: Build Qwen prompt
    qwen_messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        qwen_messages.append({"role": "user", "content": user_msg})
        qwen_messages.append({"role": "assistant", "content": assistant_msg})
    qwen_messages.append({"role": "user", "content": query})

    prompt_encoded = qwen_tokenizer.apply_chat_template(
        qwen_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(prompt_encoded, Mapping):
        prompt_ids = prompt_encoded["input_ids"]
    else:
        prompt_ids = prompt_encoded
    prompt_ids = prompt_ids.to(qwen_model.device)
    prompt_len = prompt_ids.shape[1]

    # Step 2: Tokenize the draft and append to prompt
    draft_token_ids = qwen_tokenizer.encode(draft_text, add_special_tokens=False)
    if not draft_token_ids:
        draft_token_ids = []

    draft_tensor = torch.tensor([draft_token_ids], device=qwen_model.device)
    input_with_draft = torch.cat([prompt_ids, draft_tensor], dim=1)

    # Step 3: Single Qwen forward pass to verify all draft tokens
    t_verify = time.perf_counter()
    with torch.no_grad():
        outputs = qwen_model(input_ids=input_with_draft)
    verify_logits = outputs.logits  # [1, seq_len, vocab_size]
    verify_time_ms = (time.perf_counter() - t_verify) * 1000

    # Step 5: Compare Qwen's predictions with draft tokens
    # At position i, the model predicts token i+1
    # So logits at positions [prompt_len-1 ... prompt_len+len(draft)-2] predict draft tokens
    accepted_tokens = 0
    n_draft = len(draft_token_ids)
    for i in range(n_draft):
        logit_pos = prompt_len - 1 + i  # logits at this position predict the next token
        qwen_pred = verify_logits[0, logit_pos].argmax(dim=-1).item()
        if qwen_pred == draft_token_ids[i]:
            accepted_tokens += 1
        else:
            break

    # Step 6: Build final output
    if accepted_tokens == n_draft and n_draft > 0:
        # Draft fully accepted!
        final_prediction = draft_text
        accept_status = "fully_accepted"
    elif accepted_tokens > 0 or n_draft == 0:
        # Partially accepted or empty draft — use Qwen to generate from the divergence point
        # Accept tokens up to the divergence, then generate the rest
        accepted_ids = draft_token_ids[:accepted_tokens]
        # Get Qwen's token at the divergence point
        if accepted_tokens < n_draft:
            diverge_token = verify_logits[0, prompt_len - 1 + accepted_tokens].argmax(dim=-1).item()
            continuation_start = torch.cat([prompt_ids, torch.tensor([accepted_ids + [diverge_token]], device=qwen_model.device)], dim=1)
        else:
            continuation_start = torch.cat([prompt_ids, torch.tensor([accepted_ids], device=qwen_model.device)], dim=1)

        # Generate remaining tokens
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
        accept_status = "partial" if n_draft > 0 else "empty_draft"
    else:
        # No tokens accepted — Qwen disagrees from the start, generate fully
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

    total_ms = (time.perf_counter() - t0) * 1000 + draft_ms

    return {
        "prediction": final_prediction,
        "latency_ms": round(total_ms, 1),
        "draft_text": draft_text,
        "draft_ms": round(draft_ms, 1),
        "verify_ms": round(verify_time_ms, 1),
        "accepted_tokens": accepted_tokens,
        "draft_tokens": n_draft,
        "accept_status": accept_status,
    }


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
    if "accept_status" in result:
        status_emoji = {"fully_accepted": "✅", "partial": "⚠️", "rejected": "❌", "empty_draft": "⏭️"}
        emoji = status_emoji.get(result["accept_status"], "")
        lines.append(f"\n{emoji} **{result['accept_status']}** — accepted {result['accepted_tokens']}/{result['draft_tokens']} tokens")
        lines.append(f"  Draft: {result['draft_ms']:.1f}ms | Verify: {result['verify_ms']:.1f}ms")
        if result.get("draft_text"):
            try:
                draft_parsed = json.loads(result["draft_text"])
                draft_str = json.dumps(draft_parsed, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                draft_str = result["draft_text"]
            lines.append(f"  Draft was: `{draft_str}`")
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

    use_diffusion = not args.no_diffusion
    if use_diffusion:
        print("Loading diffusion drafter...")
        diff_model, diff_tokenizer, diff_config = load_diffusion_drafter(args)
    else:
        diff_model = diff_tokenizer = diff_config = None

    print("Loading Qwen model...")
    qwen_model, qwen_tokenizer = load_qwen_model(args)

    print("Models loaded. Starting server...")

    def predict(query: str, history: list[list[str]]):
        if not query.strip():
            return ("", "", "", "") if use_diffusion else ("", "", "")

        print(f"\n{'='*60}", flush=True)
        print(f"[Query] {query}", flush=True)
        if history:
            print(f"[History] {len(history)} turns", flush=True)
        print("-" * 60, flush=True)

        # Step 1: Run structured drafter (fast, ~15ms)
        sd_result = infer_structured(
            query, history,
            sd_model, sd_tokenizer, sd_space, schemas, args.structured_max_length,
        )
        print(f"  [Structured] {sd_result['latency_ms']:.1f}ms | kind={sd_result['kind']}({sd_result['kind_prob']:.1%}) tool={sd_result['tool']}({sd_result['tool_prob']:.1%})", flush=True)
        print(f"    output: {sd_result['prediction'][:150]}", flush=True)

        # Step 2: Run diffusion drafter if available
        if use_diffusion:
            diff_result = infer_diffusion(
                query, history,
                diff_model, diff_tokenizer, diff_config, schemas,
            )

        # Step 3: Run Qwen AR sequentially (no GPU contention)
        qwen_result = infer_qwen(
            query, history,
            qwen_model, qwen_tokenizer, system_prompt, args.qwen_max_new_tokens,
        )
        print(f"  [Qwen AR] {qwen_result['latency_ms']:.1f}ms", flush=True)
        print(f"    output: {qwen_result['prediction'][:150]}", flush=True)

        # Step 4: Speculative decoding reuses the draft from step 1
        spec_result = infer_speculative(
            query, history,
            sd_result["prediction"], sd_result["latency_ms"],
            qwen_model, qwen_tokenizer, system_prompt, args.qwen_max_new_tokens,
        )
        print(f"  [Speculative] {spec_result['latency_ms']:.1f}ms | {spec_result['accept_status']} ({spec_result['accepted_tokens']}/{spec_result['draft_tokens']} tokens)", flush=True)
        print(f"    draft : {spec_result['draft_text'][:150]}", flush=True)
        print(f"    output: {spec_result['prediction'][:150]}", flush=True)
        print("=" * 60, flush=True)

        sd_out = _format_output(sd_result, "Structured Drafter (BERT)")
        qwen_out = _format_output(qwen_result, "Qwen AR (自回归)")
        spec_out = _format_output(spec_result, "Speculative Decoding (BERT→Qwen)")
        if use_diffusion:
            diff_out = _format_output(diff_result, "Diffusion Drafter (BERT)")
            return sd_out, diff_out, qwen_out, spec_out
        return sd_out, qwen_out, spec_out

    with gr.Blocks(title="Automotive Tool-Call Comparison") as app:
        if use_diffusion:
            gr.Markdown("# 🚗 四模式对比: Structured / Diffusion / Qwen AR / Speculative")
            gr.Markdown("输入车载指令，对比四种推理模式的预测结果和延迟。")
        else:
            gr.Markdown("# 🚗 三模式对比: Structured Drafter / Qwen AR / Speculative Decoding")
            gr.Markdown("输入车载指令，对比三种推理模式的预测结果和延迟。")

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

        if use_diffusion:
            with gr.Row():
                sd_output = gr.Markdown(label="Structured Drafter")
                diff_output = gr.Markdown(label="Diffusion Drafter")
                qwen_output = gr.Markdown(label="Qwen AR")
                spec_output = gr.Markdown(label="Speculative Decoding")
        else:
            with gr.Row():
                sd_output = gr.Markdown(label="Structured Drafter")
                qwen_output = gr.Markdown(label="Qwen AR")
                spec_output = gr.Markdown(label="Speculative Decoding")

        with gr.Accordion("对话历史 (多轮)", open=False):
            history_display = gr.JSON(label="History")

        if use_diffusion:
            def on_submit(query, history):
                sd_out, diff_out, qwen_out, spec_out = predict(query, history)
                new_history = history + [[query, ""]]
                return sd_out, diff_out, qwen_out, spec_out, new_history, new_history, ""

            def on_clear():
                return "", "", "", "", [], [], ""

            submit_btn.click(
                on_submit,
                inputs=[query_input, chatbot_history],
                outputs=[sd_output, diff_output, qwen_output, spec_output, chatbot_history, history_display, query_input],
            )
            query_input.submit(
                on_submit,
                inputs=[query_input, chatbot_history],
                outputs=[sd_output, diff_output, qwen_output, spec_output, chatbot_history, history_display, query_input],
            )
            clear_btn.click(
                on_clear,
                outputs=[sd_output, diff_output, qwen_output, spec_output, chatbot_history, history_display, query_input],
            )
        else:
            def on_submit(query, history):
                sd_out, qwen_out, spec_out = predict(query, history)
                new_history = history + [[query, ""]]
                return sd_out, qwen_out, spec_out, new_history, new_history, ""

            def on_clear():
                return "", "", "", [], [], ""

            submit_btn.click(
                on_submit,
                inputs=[query_input, chatbot_history],
                outputs=[sd_output, qwen_output, spec_output, chatbot_history, history_display, query_input],
            )
            query_input.submit(
                on_submit,
                inputs=[query_input, chatbot_history],
                outputs=[sd_output, qwen_output, spec_output, chatbot_history, history_display, query_input],
            )
            clear_btn.click(
                on_clear,
                outputs=[sd_output, qwen_output, spec_output, chatbot_history, history_display, query_input],
            )

    return app


def main() -> int:
    args = parse_args()
    app = build_app(args)
    app.launch(server_name=args.server_name, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
