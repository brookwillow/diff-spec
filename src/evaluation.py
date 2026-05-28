from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class OutputKind(str, Enum):
    TOOL_CALLS = "tool_calls"
    REJECT = "reject"
    TEXT = "text"
    INVALID_JSON = "invalid_json"


@dataclass(frozen=True)
class ParsedOutput:
    kind: OutputKind
    raw: str
    tool_calls: list[dict[str, Any]]
    text: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


@dataclass(frozen=True)
class PairScore:
    exact_match: bool
    expected_kind: OutputKind
    predicted_kind: OutputKind
    schema_valid: bool
    errors: list[str]


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    exact_matches: int
    schema_valid: int
    invalid_json: int
    wrong_kind: int

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total if self.total else 0.0

    @property
    def schema_valid_rate(self) -> float:
        return self.schema_valid / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "exact_matches": self.exact_matches,
            "exact_match_rate": self.exact_match_rate,
            "schema_valid": self.schema_valid,
            "schema_valid_rate": self.schema_valid_rate,
            "invalid_json": self.invalid_json,
            "wrong_kind": self.wrong_kind,
        }


@dataclass(frozen=True)
class DatasetValidationSummary:
    files: int
    total: int
    schema_valid: int
    invalid_json: int
    errors: list[str]

    @property
    def schema_valid_rate(self) -> float:
        return self.schema_valid / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "total": self.total,
            "schema_valid": self.schema_valid,
            "schema_valid_rate": self.schema_valid_rate,
            "invalid_json": self.invalid_json,
            "errors": self.errors,
        }


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


def load_tool_schemas(path: str | Path) -> dict[str, dict[str, Any]]:
    tools = json.loads(Path(path).read_text(encoding="utf-8"))
    return {tool["name"]: tool for tool in tools}


def parse_assistant_output(output: str) -> ParsedOutput:
    raw = output.strip()
    if raw == "Reject":
        return ParsedOutput(OutputKind.REJECT, raw, [], text=raw)

    if not raw.startswith(("{", "[")):
        return ParsedOutput(OutputKind.TEXT, raw, [], text=raw)

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParsedOutput(OutputKind.INVALID_JSON, raw, [], error=str(exc))

    if isinstance(decoded, dict):
        return ParsedOutput(OutputKind.TOOL_CALLS, raw, [decoded])
    if isinstance(decoded, list) and all(isinstance(item, dict) for item in decoded):
        return ParsedOutput(OutputKind.TOOL_CALLS, raw, decoded)
    return ParsedOutput(OutputKind.INVALID_JSON, raw, [], error="tool output must be object or array")


def canonicalize_output(output: str) -> str:
    parsed = parse_assistant_output(output)
    if parsed.kind == OutputKind.TOOL_CALLS:
        value: Any = parsed.tool_calls[0] if len(parsed.tool_calls) == 1 else parsed.tool_calls
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return parsed.raw


def expected_output(row: dict[str, Any]) -> str:
    if "expected_tool_calls" in row or "expected_type" in row:
        calls = row.get("expected_tool_calls") or []
        if calls:
            value: Any = calls[0] if len(calls) == 1 else calls
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        expected_type = row.get("expected_type")
        if expected_type == "Reject":
            return "Reject"
        if expected_type == "Clarify":
            return "Clarify"
        if expected_type == "Action":
            return json.dumps([], ensure_ascii=False, separators=(",", ":"))

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("row must contain non-empty messages")
    content = messages[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("last message must contain string content")
    return content


def prediction_output(row: dict[str, Any]) -> str:
    for key in ("prediction", "output", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    if isinstance(row.get("messages"), list):
        return expected_output(row)
    raise ValueError("prediction row must contain prediction, output, content, or messages")


class Evaluator:
    def __init__(self, schemas: dict[str, dict[str, Any]]):
        self.schemas = schemas

    def validate_tool_call(self, call: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        name = call.get("name")
        arguments = call.get("arguments")

        if not isinstance(name, str):
            return ValidationResult(False, ["missing or invalid tool name"])
        if name not in self.schemas:
            return ValidationResult(False, [f"unknown tool: {name}"])
        if not isinstance(arguments, dict):
            return ValidationResult(False, ["missing or invalid arguments object"])

        schema = self.schemas[name]["inputSchema"]
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for field in required:
            if field not in arguments:
                errors.append(f"missing required argument: {field}")

        for field, value in arguments.items():
            field_schema = properties.get(field)
            if field_schema is None:
                errors.append(f"unknown argument: {field}")
                continue
            errors.extend(self._validate_value(field, value, field_schema))

        return ValidationResult(not errors, errors)

    def score_pair(self, expected: str, predicted: str) -> PairScore:
        expected_parsed = parse_assistant_output(expected)
        predicted_parsed = parse_assistant_output(predicted)
        errors: list[str] = []

        schema_valid = True
        if predicted_parsed.kind == OutputKind.TOOL_CALLS:
            for call in predicted_parsed.tool_calls:
                result = self.validate_tool_call(call)
                if not result.valid:
                    schema_valid = False
                    errors.extend(result.errors)
        elif predicted_parsed.kind == OutputKind.INVALID_JSON:
            schema_valid = False
            if predicted_parsed.error:
                errors.append(predicted_parsed.error)

        return PairScore(
            exact_match=canonicalize_output(expected) == canonicalize_output(predicted),
            expected_kind=expected_parsed.kind,
            predicted_kind=predicted_parsed.kind,
            schema_valid=schema_valid,
            errors=errors,
        )

    def score_row(self, expected_row: dict[str, Any], prediction_row: dict[str, Any]) -> PairScore:
        expected = expected_output(expected_row)
        predicted = prediction_output(prediction_row)
        score = self.score_pair(expected, predicted)
        if expected_row.get("expected_type") == "Clarify":
            predicted_parsed = parse_assistant_output(predicted)
            return PairScore(
                exact_match=predicted_parsed.kind == OutputKind.TEXT,
                expected_kind=OutputKind.TEXT,
                predicted_kind=predicted_parsed.kind,
                schema_valid=score.schema_valid,
                errors=score.errors,
            )
        return score

    def evaluate_files(
        self,
        gold_path: str | Path,
        prediction_path: str | Path,
        limit: int | None = None,
    ) -> EvaluationSummary:
        gold_rows = load_jsonl(gold_path)
        prediction_rows = load_jsonl(prediction_path)
        if limit is not None:
            gold_rows = gold_rows[:limit]
            prediction_rows = prediction_rows[:limit]
        if len(gold_rows) != len(prediction_rows):
            raise ValueError(
                f"gold/prediction row count mismatch: {len(gold_rows)} != {len(prediction_rows)}"
            )

        exact_matches = 0
        schema_valid = 0
        invalid_json = 0
        wrong_kind = 0
        for gold, prediction in zip(gold_rows, prediction_rows):
            score = self.score_row(gold, prediction)
            exact_matches += int(score.exact_match)
            schema_valid += int(score.schema_valid)
            invalid_json += int(score.predicted_kind == OutputKind.INVALID_JSON)
            wrong_kind += int(score.expected_kind != score.predicted_kind)

        return EvaluationSummary(
            total=len(gold_rows),
            exact_matches=exact_matches,
            schema_valid=schema_valid,
            invalid_json=invalid_json,
            wrong_kind=wrong_kind,
        )

    def validate_dataset(self, paths: list[Path]) -> DatasetValidationSummary:
        total = 0
        schema_valid = 0
        invalid_json = 0
        errors: list[str] = []

        for path in paths:
            for index, row in enumerate(load_jsonl(path), start=1):
                total += 1
                parsed = parse_assistant_output(expected_output(row))
                if parsed.kind == OutputKind.INVALID_JSON:
                    invalid_json += 1
                    errors.append(f"{path}:{index}: invalid JSON: {parsed.error}")
                    continue
                if parsed.kind != OutputKind.TOOL_CALLS:
                    schema_valid += 1
                    continue

                row_valid = True
                for call in parsed.tool_calls:
                    result = self.validate_tool_call(call)
                    if not result.valid:
                        row_valid = False
                        errors.append(f"{path}:{index}: {'; '.join(result.errors)}")
                schema_valid += int(row_valid)

        return DatasetValidationSummary(
            files=len(paths),
            total=total,
            schema_valid=schema_valid,
            invalid_json=invalid_json,
            errors=errors,
        )

    def _validate_value(self, field: str, value: Any, schema: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        expected_type = schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"{field} must be string")
            return errors
        if expected_type == "integer" and not isinstance(value, int):
            errors.append(f"{field} must be integer")
            return errors

        enum = schema.get("enum")
        if enum and isinstance(value, str) and value not in enum:
            if not _allows_freeform_value(enum, value):
                errors.append(f"{field} value not in enum: {value}")
        return errors


def _allows_freeform_value(enum: list[Any], value: str) -> bool:
    if any(isinstance(item, str) and item.startswith("<") and item.endswith(">") for item in enum):
        return True
    return bool(re.fullmatch(r"\d+(\.\d+)?%?", value))
