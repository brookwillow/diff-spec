#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_tool_schemas
from src.prepare_structured_drafter_data import collect_structured_rows, write_label_space, write_structured_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare structured drafter training JSONL.")
    parser.add_argument("--source-dir", default="data/splits")
    parser.add_argument("--system", default="data/system-prompt.txt")
    parser.add_argument("--tools", default="data/tools.json")
    parser.add_argument("--output", default="data/structured/train.jsonl")
    parser.add_argument("--label-space-output", default="data/structured/label_space.json")
    args = parser.parse_args()

    paths = sorted(Path(args.source_dir).glob("**/*.jsonl"))
    system_prompt = Path(args.system).read_text(encoding="utf-8")
    schemas = load_tool_schemas(args.tools)
    rows, space = collect_structured_rows(paths, schemas, system_prompt)
    write_structured_rows(rows, args.output)
    write_label_space(space, args.label_space_output)
    print(
        json.dumps(
            {
                "source_files": len(paths),
                "rows": len(rows),
                "output": args.output,
                "label_space": args.label_space_output,
                "tools": len(space.tool_to_id) - 1,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
