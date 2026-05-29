from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation import expected_output, load_jsonl, parse_assistant_output
from src.prepare_qwen_sft_data import is_trainable_messages, write_jsonl


ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def render_prompt(messages: list[dict[str, Any]], system_prompt: str) -> str:
    prompt_messages = [message for message in messages if isinstance(message, dict)]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages = prompt_messages[:-1]

    if prompt_messages and prompt_messages[0].get("role") == "system":
        prompt_messages[0] = {"role": "system", "content": system_prompt}
    else:
        prompt_messages.insert(0, {"role": "system", "content": system_prompt})

    rendered: list[str] = []
    for message in prompt_messages:
        role = ROLE_LABELS.get(str(message.get("role")))
        content = message.get("content")
        if role is None or not isinstance(content, str):
            continue
        rendered.append(f"{role}:\n{content.strip()}")
    rendered.append("Assistant:\n")
    return "\n\n".join(rendered)


def _tool_name(target: str) -> str | None:
    parsed = parse_assistant_output(target)
    if parsed.tool_calls:
        name = parsed.tool_calls[0].get("name")
        return name if isinstance(name, str) else None
    return None


def _row_kind(row: dict[str, Any], target: str) -> str:
    expected_type = row.get("expected_type")
    if expected_type in {"Action", "Reject", "Clarify"}:
        return str(expected_type)
    parsed = parse_assistant_output(target)
    if parsed.tool_calls:
        return "Action"
    if parsed.raw == "Reject":
        return "Reject"
    return "Clarify"


def collect_diffusion_rows(paths: list[Path], system_prompt: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            messages = row.get("messages")
            if not is_trainable_messages(messages):
                continue
            target = expected_output(row)
            normalized = {
                "id": row.get("id"),
                "prompt": render_prompt(messages, system_prompt),
                "target": target,
                "kind": _row_kind(row, target),
                "tool_name": _tool_name(target),
            }
            key = json.dumps({"prompt": normalized["prompt"], "target": target}, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)
    return rows


def write_diffusion_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    write_jsonl(rows, path)
