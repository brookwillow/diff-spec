# Repository Guidelines

## Project Scope

This repository supports **Diffusion-Guided Speculative Decoding: A Hybrid Approach for Accelerating Tool-Calling in Automotive Assistants**. Focus on low-latency automotive tool-call generation with a small diffusion drafter and lightweight autoregressive verifier.

## Project Structure & Module Organization

- `src/`: Python code for data processing, diffusion drafting, AR verification, constrained JSON decoding, and evaluation.
- `configs/`: model, training, quantization, and benchmark configs.
- `data/`: generated automotive tool-call samples. Do not commit private or large raw data.
- `data/eval_text/`: canonical text-only offline evaluation set.
- `data/sft/`: generated Qwen SFT training JSONL.
- `scripts/`: repeatable training, evaluation, export, and profiling commands.
- `tests/`: unit and regression tests mirroring `src/`.
- `docs/`: proposal text, experiment notes, figures, and paper drafts.

## Build, Test, and Development Commands

No runnable tooling is committed yet. When implementation starts, prefer stable entry points:

- `python -m unittest tests/test_evaluation.py`: run evaluation tests.
- `python scripts/validate_dataset.py --data-dir data/splits --tools data/tools.json`: validate gold data against tool schemas.
- `python3 scripts/prepare_qwen_sft_data.py --source-dir data/splits --output data/sft/qwen_train.jsonl`: regenerate Qwen SFT data.
- `bash scripts/run_qwen_sft.sh`: run ms-swift LoRA SFT in the `qwen-omni` conda environment.
- `python scripts/evaluate.py --gold <gold.jsonl> --predictions <pred.jsonl>`: measure exact match, schema validity, invalid JSON, and output kind errors.

Document new commands in `README.md`; keep defaults runnable on an RTX 3090-class machine.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, public type hints, and `snake_case` for files, functions, and variables. Name modules by role, for example `diffusion_drafter.py`, `ar_verifier.py`, and `json_constraints.py`.

Keep model code separate from experiment orchestration. Put paths, model names, thresholds, and device IDs in `configs/`.

## Testing Guidelines

Tests should cover JSON validation, constrained token filtering, confidence routing, AR fallback, and metrics. Use deterministic fixtures for HVAC, navigation, media, windows, and seat controls.

Name tests after behavior, for example `test_routes_low_confidence_draft_to_verifier`.

## Experiment & Evaluation Standards

Compare against pure 3B autoregressive decoding, pure diffusion decoding, and standard speculative decoding. Report P50/P99 latency, exact match, JSON validity, tool-call accuracy, memory, and throughput.

Record hardware, quantization mode, checkpoint, dataset split, confidence threshold, draft length, and denoising steps.

## Commit & Pull Request Guidelines

This directory currently has no git history, so use concise imperative commits such as `Add latency profiler` or `Document hybrid decoding plan`.

Pull requests should include scope, motivation, commands run, key metrics, and changed configs. For experiment PRs, attach a short result table and state whether accuracy remains within the 90-93% target.

## Agent-Specific Instructions

Do not invent benchmark numbers. If results are not measured, mark them as planned or estimated. Keep documentation aligned with the hybrid diffusion-plus-autoregressive architecture.
