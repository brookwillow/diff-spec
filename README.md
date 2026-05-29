# Diffusion-Guided Speculative Decoding

面向车载语音助手工具调用任务的低延迟混合解码研究与工程实践。

## 项目目标

本项目研究 **Diffusion-Guided Speculative Decoding: A Hybrid Approach for Accelerating Tool-Calling in Automotive Assistants**。核心目标是在保持工具调用准确率基本不下降的前提下，将车载助手从自然语言指令到结构化工具调用 JSON 的响应延迟压缩到 500ms 以内。

## 核心方案

系统采用“小扩散模型起草 + 轻量自回归模型验证”的混合架构：

1. 扩散草稿模型并行生成完整工具调用 JSON，例如 `{"action": "set_temp", "value": 22}`。
2. 结构校验器检查 JSON 合法性、字段约束和置信度。
3. 低风险草稿直接接受并执行。
4. 低置信度或非法草稿交给自回归验证器逐 token 修正。
5. 评估模块记录 P50/P99 延迟、准确率、JSON 合法性、显存和吞吐。

## 计划目录

```text
src/
  diffusion_drafter.py      # 扩散草稿生成
  ar_verifier.py            # 自回归验证与纠错
  json_constraints.py       # 工具调用 JSON 约束
  evaluation.py             # 输出解析、schema 校验和指标统计
  prepare_qwen_sft_data.py  # Qwen SFT 数据准备
configs/
  qwen_sft_lora.yaml        # ms-swift LoRA SFT 配置
data/
  sft/qwen_train.jsonl      # Qwen SFT 训练集
scripts/
  evaluate.py
  prepare_qwen_sft_data.py
  run_qwen_sft.py
  run_qwen_sft.sh
  validate_dataset.py
tests/
docs/
  paper-draft.md            # 论文初稿
```

## 工程实践计划

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 明确课题、技术路线和指标 | 已完成 |
| 2 | 整理论文初稿与项目 README | 已完成 |
| 3 | 构建车载工具调用数据集与 tool schema | 已完成 |
| 4 | 实现 JSON schema 与评估环境 | 已完成 |
| 5 | 实现扩散草稿模型推理接口 | 未开始 |
| 6 | 接入自回归验证器与投机接受逻辑 | 未开始 |
| 7 | 整理 text-only eval 集并接入评估器 | 已完成 |
| 8 | 准备 Qwen ms-swift SFT 脚手架 | 已完成 |
| 9 | 完成 Qwen baseline、消融和延迟评估 | 未开始 |

## 预期评估

计划比较以下系统：

- 纯 3B 自回归模型，例如 Qwen2.5-3B。
- 纯扩散模型直接解码。
- 小自回归草稿 + 大自回归验证的标准投机解码。
- 本项目的扩散草稿 + 自回归验证混合方案。

指标包括首包延迟、总响应延迟、P50/P99、JSON 合法率、精确匹配率、工具调用准确率、吞吐量和显存占用。当前所有性能数字均为研究目标或估计值，尚未实测。

## 评估环境

训练/开发数据集位于 `data/splits/`，标准离线评估集位于 `data/eval_text/all.jsonl`，tool schema 位于 `data/tools.json`，系统提示词位于 `data/system-prompt.txt`。`data/eval_text/` 是当前唯一保留的评估集，仅包含文本输入、历史、期望工具调用和类型标签。

```bash
python scripts/validate_dataset.py --data-dir data/splits --tools data/tools.json
python scripts/evaluate.py \
  --gold data/eval_text/all.jsonl \
  --predictions predictions.jsonl \
  --tools data/tools.json
```

预测文件每行可使用 `{"prediction":"..."}`、`{"output":"..."}`、`{"content":"..."}`，也可复用 `messages` 格式。评估器会统计 exact match、schema valid、invalid JSON 和输出类型错误。Clarify 样本按类型评估，只要求输出自然语言追问，不要求固定文案完全一致。

## 当前进度

- 已确定论文题目与研究问题。
- 已形成混合解码方案概况。
- 已创建仓库贡献指南 `AGENTS.md`。
- 已创建论文初稿 `docs/paper-draft.md`。
- 已接入 `data/tools.json`、`data/system-prompt.txt` 和 `data/splits/`。
- 已实现 `src/evaluation.py`、`scripts/evaluate.py` 和 `scripts/validate_dataset.py`。
- 已保留 text-only 标准评估集 `data/eval_text/all.jsonl`，共 1787 条。
- 已准备 Qwen LoRA SFT 数据脚本、ms-swift 配置和启动脚本。

## 使用说明

当前仓库已具备第一版离线评估环境。优先使用以下入口：

```bash
python -m unittest tests/test_evaluation.py
python scripts/validate_dataset.py --data-dir data/splits --tools data/tools.json
python3 scripts/prepare_qwen_sft_data.py --source-dir data/splits --output data/sft/qwen_train.jsonl
python scripts/evaluate.py --gold data/eval_text/all.jsonl --predictions predictions.jsonl
```

## Qwen SFT 训练

训练入口是 `scripts/run_qwen_sft.py`，默认使用 `configs/qwen_sft_lora.yaml`，并在启动前自动生成 `data/sft/qwen_train.jsonl`。

AutoDL 或新环境先确认 PyTorch 已安装：

```bash
conda run -n qwen-omni python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果缺少 `torch`，先安装 PyTorch。RTX 50 系列/Blackwell CUDA 服务器优先使用较新的 CUDA wheel：

```bash
conda run -n qwen-omni pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
python3 scripts/run_qwen_sft.py
```

指定配置文件：

```bash
python3 scripts/run_qwen_sft.py configs/qwen_sft_lora.yaml
```

只查看将要执行的 swift 命令，不启动训练：

```bash
DRY_RUN=1 python3 scripts/run_qwen_sft.py
```

强制使用 Apple MPS、CUDA 或 CPU：

```bash
SWIFT_DEVICE=mps python3 scripts/run_qwen_sft.py
SWIFT_DEVICE=cuda python3 scripts/run_qwen_sft.py
SWIFT_DEVICE=cpu python3 scripts/run_qwen_sft.py
```

`scripts/run_qwen_sft.py` 会自动按 CUDA、Apple MPS、CPU 的顺序选择训练设备。Apple Silicon 可用时会使用 `mps:0`，并切换到 `float32` 与 `eager` attention 以提高兼容性。SFT 配置当前使用 `data/system-prompt.txt` 完整提示词，并将 `max_length` 设为 4096 以保留 tool schema 和较长多轮上下文。`loss_scale: last_round` 只监督最后一个 assistant 输出，历史 assistant 仅作为上下文。当前 Qwen text verifier 训练继续使用 `swift sft` CLI；Omni thinker-only、冻结审计或自定义 label span 需求再切 SDK Trainer。`scripts/run_qwen_sft.sh` 仅作为兼容入口转发到 Python 脚本。
启动训练时脚本会打印完整 `conda run ... swift sft ...` 命令，并使用 `conda run --no-capture-output` 透传 Swift 实时日志；如果终端长时间没有新日志，通常是在模型下载、加载或 MPS 编译阶段。
