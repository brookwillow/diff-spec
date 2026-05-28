#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare_qwen_sft_data import collect_sft_rows, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ms-swift messages JSONL for Qwen SFT.")
    parser.add_argument("--source-dir", default="data/splits", help="Directory containing source JSONL splits.")
    parser.add_argument("--output", default="data/sft/qwen_train.jsonl", help="Output training JSONL path.")
    args = parser.parse_args()

    paths = sorted(Path(args.source_dir).glob("**/*.jsonl"))
    rows = collect_sft_rows(paths)
    write_jsonl(rows, args.output)
    print(json.dumps({"source_files": len(paths), "rows": len(rows), "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
