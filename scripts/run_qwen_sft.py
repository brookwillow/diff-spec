#!/usr/bin/env python3
"""Prepare Qwen SFT data and launch ms-swift with device-aware defaults."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from shlex import quote


REPO_ROOT = Path(__file__).resolve().parents[1]


def format_command(command: list[str]) -> str:
    return " ".join(quote(part) for part in command)


def run(command: list[str], env: dict[str, str] | None = None) -> int:
    print("Running: " + format_command(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


def require_conda() -> None:
    if shutil.which("conda") is None:
        conda_env = os.environ.get("CONDA_ENV", "qwen-omni")
        raise SystemExit(f"conda is required; expected environment: {conda_env}")


def detect_torch_device(conda_env: str) -> str:
    override = os.environ.get("SWIFT_DEVICE")
    if override:
        if override not in {"cuda", "mps", "cpu"}:
            raise SystemExit(f"Unsupported SWIFT_DEVICE='{override}'. Expected cuda, mps, or cpu.")
        return override

    code = (
        "import torch; "
        "print('cuda' if torch.cuda.is_available() else "
        "'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() "
        "else 'cpu')"
    )
    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", conda_env, "python", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    device = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if device not in {"cuda", "mps", "cpu"}:
        raise SystemExit(f"Unable to detect a supported training device: {device!r}")
    return device


def build_device_args(device: str, env: dict[str, str]) -> list[str]:
    if device == "cuda":
        return []

    args = [
        "--torch_dtype",
        "float32",
        "--fp16",
        "false",
        "--bf16",
        "false",
        "--attn_impl",
        "eager",
    ]
    if device == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return ["--device_map", "mps:0", *args]
    return ["--device_map", "cpu", *args]


def print_dry_run(conda_env: str, config_path: str, device_args: list[str], env: dict[str, str]) -> None:
    if env.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        print(f"Environment: PYTORCH_ENABLE_MPS_FALLBACK={env['PYTORCH_ENABLE_MPS_FALLBACK']}")
    command = ["conda", "run", "--no-capture-output", "-n", conda_env, "swift", "sft", config_path, *device_args]
    print("Command: " + format_command(command), flush=True)


def main(argv: list[str]) -> int:
    config_path = argv[1] if len(argv) > 1 else "configs/qwen_sft_lora.yaml"
    conda_env = os.environ.get("CONDA_ENV", "qwen-omni")
    sft_data_path = os.environ.get("SFT_DATA_PATH", "data/sft/qwen_train.jsonl")

    require_conda()

    if os.environ.get("SKIP_PREPARE") != "1":
        code = run(
            [
                sys.executable,
                "scripts/prepare_qwen_sft_data.py",
                "--source-dir",
                "data/splits",
                "--output",
                sft_data_path,
            ]
        )
        if code != 0:
            return code

    env = os.environ.copy()
    device = detect_torch_device(conda_env)
    device_args = build_device_args(device, env)

    print(f"Detected device: {device}", flush=True)

    if os.environ.get("DRY_RUN") == "1":
        print_dry_run(conda_env, config_path, device_args, env)
        return 0

    command = ["conda", "run", "--no-capture-output", "-n", conda_env, "swift", "sft", config_path, *device_args]
    return run(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
