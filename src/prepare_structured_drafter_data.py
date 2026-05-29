from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation import load_jsonl
from src.prepare_diffusion_data import render_prompt
from src.prepare_qwen_sft_data import is_trainable_messages, write_jsonl
from src.structured_drafter import StructuredLabelSpace, build_label_space, labels_from_row


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
            if not is_trainable_messages(messages):
                continue
            key = json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            prepared = dict(row)
            prepared["prompt"] = render_prompt(messages, system_prompt)
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
