# Qwen SFT Baseline

This project uses `ms-swift` in the `qwen-omni` conda environment for the first autoregressive verifier baseline.

## Data

Training data is generated from `data/splits/**/*.jsonl`:

```bash
python3 scripts/prepare_qwen_sft_data.py \
  --source-dir data/splits \
  --output data/sft/qwen_train.jsonl
```

The script keeps direct text-to-assistant examples and multi-turn text histories, then filters tool execution transcripts that contain `role=tool`.

Current generated size:

```text
data/sft/qwen_train.jsonl: 6885 rows
data/eval_text/all.jsonl: 1787 rows
```

## Training

Default LoRA config:

```bash
conda run -n qwen-omni swift sft configs/qwen_sft_lora.yaml
```

Convenience wrapper:

```bash
python3 scripts/run_qwen_sft.py
```

The wrapper detects devices in this order: CUDA, Apple MPS, then CPU. On Apple Silicon with an MPS-enabled PyTorch build, it runs swift with `--device_map mps:0`, `float32`, and `eager` attention, plus `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops. You can force a branch or inspect the command without starting training:

```bash
SWIFT_DEVICE=mps DRY_RUN=1 python3 scripts/run_qwen_sft.py
SWIFT_DEVICE=cpu DRY_RUN=1 python3 scripts/run_qwen_sft.py
```

`scripts/run_qwen_sft.sh` remains as a thin compatibility wrapper around the Python entrypoint.

## Supervision

The config sets `loss_scale: last_round`, so ms-swift masks system/user tokens and previous assistant turns, then computes loss only on the final assistant response. This matches the verifier objective: given the full prompt and conversation history, predict the current turn's tool JSON, clarification, or rejection.

## Swift Invocation

This baseline intentionally uses the `swift sft` CLI through `scripts/run_qwen_sft.py`. The CLI path is enough for Qwen text-only LoRA because native `loss_scale: last_round` covers final-response supervision. Use the Swift SDK Trainer path only when we move to Qwen2.5-Omni thinker-only training, parameter freezing audits, or custom label-span construction.

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. The SFT config uses `max_length: 4096` and the full prompt at `data/system-prompt.txt`, so each sample includes the tool schema during training. On offline machines, edit `configs/qwen_sft_lora.yaml` and set `model` to a local model path.

## Evaluation Contract

After inference, write one prediction per line:

```json
{"prediction":"{\"name\":\"ClimateControl\",\"arguments\":{\"action\":\"打开\",\"device\":\"空调\"}}"}
```

Then run:

```bash
python3 scripts/evaluate.py \
  --gold data/eval_text/all.jsonl \
  --predictions predictions.jsonl \
  --tools data/tools.json
```
