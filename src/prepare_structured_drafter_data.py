from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation import load_jsonl
from src.prepare_diffusion_data import ROLE_LABELS, render_prompt
from src.prepare_qwen_sft_data import write_jsonl
from src.structured_drafter import StructuredLabelSpace, build_label_space, labels_from_row


def render_structured_prompt(messages: list[dict[str, Any]]) -> str:
    """Render conversation turns only (no system prompt) for the structured classifier.

    The structured drafter uses a BERT encoder with max 512 tokens.
    Including the full tool-schema system prompt would consume all tokens
    and truncate the actual user message.  The classifier learns tool/slot
    mappings implicitly from labels, so only user/assistant turns are needed.
    """
    prompt_messages = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages = prompt_messages[:-1]
    rendered: list[str] = []
    for message in prompt_messages:
        role = ROLE_LABELS.get(str(message.get("role")))
        content = message.get("content")
        if role is None or not isinstance(content, str):
            continue
        rendered.append(f"{role}:\n{content.strip()}")
    return "\n\n".join(rendered) if rendered else ""


def _is_usable_messages(messages: Any) -> bool:
    """Like is_trainable_messages but allows 'tool' role for multi-turn tool-call dialogues."""
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if any(not isinstance(m, dict) for m in messages):
        return False
    if messages[-1].get("role") != "assistant":
        return False
    if not any(m.get("role") == "user" for m in messages[:-1]):
        return False
    return all(isinstance(m.get("content"), str) and m["content"] for m in messages)


def collect_structured_rows(
    paths: list[Path],
    schemas: dict[str, dict[str, Any]],
    system_prompt: str,
) -> tuple[list[dict[str, Any]], StructuredLabelSpace]:
    source_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            messages = row.get("messages")
            if not _is_usable_messages(messages):
                continue
            key = json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            prepared = dict(row)
            prepared["prompt"] = render_structured_prompt(messages)
            source_rows.append(prepared)

    space = build_label_space(schemas, source_rows)
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        labels = labels_from_row(row, space)
        rows.append(
            {
                "id": row.get("id"),
                "prompt": labels.prompt,
                "kind_id": labels.kind_id,
                "tool_id": labels.tool_id,
                "slot_ids": labels.slot_ids,
            }
        )
    return rows, space


def write_label_space(space: StructuredLabelSpace, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(space), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_structured_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    write_jsonl(rows, path)
