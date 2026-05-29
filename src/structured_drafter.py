from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SLOT_NAMES = ("action", "device", "feature", "position", "value", "query", "index", "contact", "phone")
KIND_NAMES = ("Action", "Reject", "Clarify")
NONE_VALUE = "NONE"


@dataclass(frozen=True)
class StructuredLabelSpace:
    kind_to_id: dict[str, int]
    id_to_kind: dict[int, str]
    tool_to_id: dict[str, int]
    id_to_tool: dict[int, str]
    slot_value_to_id: dict[str, dict[str, int]]
    id_to_slot_value: dict[str, dict[int, str]]


@dataclass(frozen=True)
class StructuredExample:
    prompt: str
    kind_id: int
    tool_id: int
    slot_ids: dict[str, int]


def build_label_space(schemas: dict[str, dict[str, Any]], rows: list[dict[str, Any]] | None = None) -> StructuredLabelSpace:
    kind_to_id = {name: index for index, name in enumerate(KIND_NAMES)}
    id_to_kind = {index: name for name, index in kind_to_id.items()}
    tools = sorted(schemas)
    tool_to_id = {NONE_VALUE: 0, **{name: index + 1 for index, name in enumerate(tools)}}
    id_to_tool = {index: name for name, index in tool_to_id.items()}

    values: dict[str, set[str]] = {slot: {NONE_VALUE} for slot in SLOT_NAMES}
    for schema in schemas.values():
        properties = schema.get("inputSchema", {}).get("properties", {})
        for slot in SLOT_NAMES:
            enum = properties.get(slot, {}).get("enum", [])
            values[slot].update(str(item) for item in enum if isinstance(item, str))
    for row in rows or []:
        for call in row.get("expected_tool_calls") or []:
            args = call.get("arguments") if isinstance(call, dict) else None
            if not isinstance(args, dict):
                continue
            for slot in SLOT_NAMES:
                value = args.get(slot)
                if isinstance(value, (str, int)):
                    values[slot].add(str(value))

    slot_value_to_id = {
        slot: {value: index for index, value in enumerate([NONE_VALUE, *sorted(items - {NONE_VALUE})])}
        for slot, items in values.items()
    }
    id_to_slot_value = {
        slot: {index: value for value, index in mapping.items()} for slot, mapping in slot_value_to_id.items()
    }
    return StructuredLabelSpace(kind_to_id, id_to_kind, tool_to_id, id_to_tool, slot_value_to_id, id_to_slot_value)


def load_label_space(path: str | Path) -> StructuredLabelSpace:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return StructuredLabelSpace(
        kind_to_id={str(k): int(v) for k, v in payload["kind_to_id"].items()},
        id_to_kind={int(k): str(v) for k, v in payload["id_to_kind"].items()},
        tool_to_id={str(k): int(v) for k, v in payload["tool_to_id"].items()},
        id_to_tool={int(k): str(v) for k, v in payload["id_to_tool"].items()},
        slot_value_to_id={
            str(slot): {str(k): int(v) for k, v in mapping.items()}
            for slot, mapping in payload["slot_value_to_id"].items()
        },
        id_to_slot_value={
            str(slot): {int(k): str(v) for k, v in mapping.items()}
            for slot, mapping in payload["id_to_slot_value"].items()
        },
    )


def labels_from_row(row: dict[str, Any], space: StructuredLabelSpace) -> StructuredExample:
    kind = row.get("expected_type") if row.get("expected_type") in KIND_NAMES else _infer_kind(row)
    tool_name = NONE_VALUE
    arguments: dict[str, Any] = {}
    calls = row.get("expected_tool_calls")
    if kind == "Action" and isinstance(calls, list) and calls:
        first = calls[0]
        if isinstance(first, dict):
            tool_name = str(first.get("name") or NONE_VALUE)
            if isinstance(first.get("arguments"), dict):
                arguments = first["arguments"]

    slot_ids = {}
    for slot in SLOT_NAMES:
        value = arguments.get(slot, NONE_VALUE)
        slot_ids[slot] = space.slot_value_to_id[slot].get(str(value), space.slot_value_to_id[slot][NONE_VALUE])

    return StructuredExample(
        prompt=str(row.get("prompt", "")),
        kind_id=space.kind_to_id[str(kind)],
        tool_id=space.tool_to_id.get(tool_name, space.tool_to_id[NONE_VALUE]),
        slot_ids=slot_ids,
    )


def render_prediction(example: StructuredExample, space: StructuredLabelSpace, schemas: dict[str, dict[str, Any]]) -> str:
    kind = space.id_to_kind.get(example.kind_id, "Reject")
    if kind == "Reject":
        return "Reject"
    if kind == "Clarify":
        return "请问您想操作哪个功能？"

    tool = space.id_to_tool.get(example.tool_id, NONE_VALUE)
    if tool == NONE_VALUE or tool not in schemas:
        return "Reject"

    schema = schemas[tool].get("inputSchema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    arguments: dict[str, Any] = {}
    for slot in SLOT_NAMES:
        if slot not in properties:
            continue
        value = space.id_to_slot_value[slot].get(example.slot_ids.get(slot, 0), NONE_VALUE)
        if value == NONE_VALUE:
            continue
        expected_type = properties[slot].get("type")
        if expected_type == "integer":
            try:
                arguments[slot] = int(value)
            except ValueError:
                continue
        else:
            arguments[slot] = value

    if any(field not in arguments for field in required):
        return "Reject"
    return json.dumps({"name": tool, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))


def label_sizes(space: StructuredLabelSpace) -> dict[str, int]:
    sizes = {
        "kind": len(space.kind_to_id),
        "tool": len(space.tool_to_id),
    }
    for slot in SLOT_NAMES:
        sizes[slot] = len(space.slot_value_to_id[slot])
    return sizes


def select_ids_from_logits(logits: dict[str, Any]) -> StructuredExample:
    slot_ids = {}
    for slot in SLOT_NAMES:
        values = logits[slot]
        slot_ids[slot] = int(values.argmax(dim=-1).item())
    return StructuredExample(
        prompt="",
        kind_id=int(logits["kind"].argmax(dim=-1).item()),
        tool_id=int(logits["tool"].argmax(dim=-1).item()),
        slot_ids=slot_ids,
    )


def _infer_kind(row: dict[str, Any]) -> str:
    calls = row.get("expected_tool_calls")
    if isinstance(calls, list) and calls:
        return "Action"
    return "Reject"
