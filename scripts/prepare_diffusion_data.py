#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare_diffusion_data import collect_diffusion_rows, write_diffusion_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare masked-diffusion drafter training JSONL.")
    parser.add_argument("--source-dir", default="data/splits", help="Directory containing source JSONL splits.")
    parser.add_argument("--system", default="data/system-prompt.txt", help="System prompt path.")
    parser.add_argument("--output", default="data/diffusion/train.jsonl", help="Output diffusion JSONL path.")
    args = parser.parse_args()

    paths = sorted(Path(args.source_dir).glob("**/*.jsonl"))
    system_prompt = Path(args.system).read_text(encoding="utf-8")
    rows = collect_diffusion_rows(paths, system_prompt)
    write_diffusion_rows(rows, args.output)
    print(json.dumps({"source_files": len(paths), "rows": len(rows), "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
