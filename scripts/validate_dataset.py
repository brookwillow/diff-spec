#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator, load_tool_schemas


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate gold automotive tool-call datasets.")
    parser.add_argument("--data-dir", default="data/splits", help="Directory containing JSONL splits.")
    parser.add_argument("--tools", default="data/tools.json", help="Tool schema JSON file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if validation errors exist.")
    args = parser.parse_args()

    paths = sorted(Path(args.data_dir).glob("**/*.jsonl"))
    evaluator = Evaluator(load_tool_schemas(args.tools))
    summary = evaluator.validate_dataset(paths)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))

    return 1 if args.strict and summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
