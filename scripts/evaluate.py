#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator, load_tool_schemas


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate automotive tool-call predictions.")
    parser.add_argument("--gold", required=True, help="Gold JSONL file in messages format.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file.")
    parser.add_argument("--tools", default="data/tools.json", help="Tool schema JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--output", default=None, help="Optional path for JSON summary.")
    args = parser.parse_args()

    evaluator = Evaluator(load_tool_schemas(args.tools))
    summary = evaluator.evaluate_files(args.gold, args.predictions, limit=args.limit)
    payload = summary.as_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
