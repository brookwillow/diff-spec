from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {"system", "user", "assistant"}


def is_trainable_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if any(not isinstance(message, dict) for message in messages):
        return False
    if any(message.get("role") not in ALLOWED_ROLES for message in messages):
        return False
    if messages[-1].get("role") != "assistant":
        return False
    if not any(message.get("role") == "user" for message in messages[:-1]):
        return False
    return all(isinstance(message.get("content"), str) and message["content"] for message in messages)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def collect_sft_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            messages = row.get("messages")
            if not is_trainable_messages(messages):
                continue
            normalized = {"messages": messages}
            key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
