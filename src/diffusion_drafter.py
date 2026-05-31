from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MaskedDrafterConfig:
    max_length: int = 512
    max_target_tokens: int = 128


def _required_token_id(tokenizer: Any, name: str) -> int:
    value = getattr(tokenizer, name, None)
    if not isinstance(value, int):
        raise ValueError(f"tokenizer must define integer {name}")
    return value


def build_masked_example(tokenizer: Any, prompt: str, target: str, config: MaskedDrafterConfig) -> dict[str, list[int]]:
    cls_id = _required_token_id(tokenizer, "cls_token_id")
    sep_id = _required_token_id(tokenizer, "sep_token_id")
    pad_id = _required_token_id(tokenizer, "pad_token_id")
    mask_id = _required_token_id(tokenizer, "mask_token_id")

    target_ids = tokenizer.encode(target, add_special_tokens=False)[: config.max_target_tokens]
    if not target_ids:
        raise ValueError("target must produce at least one token")
    supervised_ids = [*target_ids, sep_id]

    prompt_budget = config.max_length - len(supervised_ids) - 3
    if prompt_budget < 1:
        raise ValueError("max_length is too small for the configured target span")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[-prompt_budget:]
    input_ids = [cls_id, *prompt_ids, sep_id, *([mask_id] * len(supervised_ids)), sep_id]
    labels = [-100] * (len(prompt_ids) + 2) + supervised_ids + [-100]
    attention_mask = [1] * len(input_ids)

    pad_count = config.max_length - len(input_ids)
    if pad_count > 0:
        input_ids.extend([pad_id] * pad_count)
        labels.extend([-100] * pad_count)
        attention_mask.extend([0] * pad_count)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DiffusionJsonDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, config: MaskedDrafterConfig):
        self.rows = rows
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        return build_masked_example(self.tokenizer, str(row["prompt"]), str(row["target"]), self.config)


def build_prediction_inputs(
    tokenizer: Any,
    prompt: str,
    config: MaskedDrafterConfig,
    target_tokens: int | None = None,
) -> dict[str, list[int]]:
    cls_id = _required_token_id(tokenizer, "cls_token_id")
    sep_id = _required_token_id(tokenizer, "sep_token_id")
    pad_id = _required_token_id(tokenizer, "pad_token_id")
    mask_id = _required_token_id(tokenizer, "mask_token_id")

    content_mask_count = target_tokens or config.max_target_tokens
    content_mask_count = max(1, min(content_mask_count, config.max_target_tokens))
    mask_count = content_mask_count + 1
    prompt_budget = config.max_length - mask_count - 3
    if prompt_budget < 1:
        raise ValueError("max_length is too small for the configured prediction span")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[-prompt_budget:]
    input_ids = [cls_id, *prompt_ids, sep_id, *([mask_id] * mask_count), sep_id]
    attention_mask = [1] * len(input_ids)
    mask_positions = list(range(len(prompt_ids) + 2, len(prompt_ids) + 2 + mask_count))

    pad_count = config.max_length - len(input_ids)
    if pad_count > 0:
        input_ids.extend([pad_id] * pad_count)
        attention_mask.extend([0] * pad_count)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_positions": mask_positions,
    }


def trim_decoded_prediction(text: str) -> str:
    cleaned = text.strip()
    for marker in ("[SEP]", "[PAD]", "[UNK]", "Assistant:", "User:", "System:"):
        index = cleaned.find(marker)
        if index >= 0:
            cleaned = cleaned[:index].strip()
    # BERT tokenizer inserts spaces between tokens on decode.
    # Our targets are compact JSON without spaces, so remove them.
    cleaned = cleaned.replace(" ", "")
    return cleaned


def restore_tool_case(prediction: str, schemas: dict | list) -> str:
    """Restore PascalCase tool names that BERT's lowercase tokenizer destroyed.

    Builds a lowercase→original mapping from tool schemas and replaces
    occurrences in the prediction string.
    """
    case_map: dict[str, str] = {}
    if isinstance(schemas, dict):
        names = list(schemas.keys())
    else:
        names = [
            (t.get("name") or t.get("function", {}).get("name", "")) if isinstance(t, dict) else str(t)
            for t in schemas
        ]
    for name in names:
        if name:
            case_map[name.lower()] = name
    for lower, original in sorted(case_map.items(), key=lambda x: -len(x[0])):
        prediction = prediction.replace(lower, original)
    return prediction
