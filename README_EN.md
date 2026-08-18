# GPT Teacher

Train a small Chinese GPT from scratch on CPU — complete with inference and a live demo.

## Project Structure

```
GPT_teacher-3.37M-cn/
├── core/       ← Shared library (GPT model, tokenizer, inference, visualization)
├── train/      ← Training module (data, config, checkpoints, web demo) → [README](train/README.md)
├── distill/    ← Knowledge distillation experiments → [README](distill/README.md)
├── pyproject.toml
└── uv.lock
```

| Module | Description | Quick Start |
|--------|-------------|-------------|
| **train** | Complete training pipeline for a 3.37M-param Chinese GPT | `uv run python train/run.py` |
| **distill** | Knowledge distillation experiments with a large-model teacher | `uv run python -m distill.train --kd --teacher Qwen/Qwen2.5-1.5B-Instruct` |
| **core** | Shared code: model, tokenizer, inference, evaluation, visualization | Used by the two modules above |

## Quick Start

```bash
# Install dependencies (Python 3.10-3.12 required)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv sync

# One command: tokenizer → train → evaluate → web demo
uv run python train/run.py
```

**Expected result**: `Result: 6/6 passed (100%)` — the model learned to answer all test questions.

See the module READMEs (in Chinese) for details:
- [train/README.md](train/README.md) — full tutorial: training, evaluation, inference, web demo
- [distill/README.md](distill/README.md) — experiment log: three distillation directions, all failed

## Tech Stack

- Architecture: Decoder-only Transformer (GQA + SwiGLU + RMSNorm + RoPE + weight tying)
- Devices: CPU / MPS (Apple Silicon) / CUDA (NVIDIA GPU)
- Package management: uv
- Unit tests: 44 (`pytest core/tests/`)

Project: https://github.com/helloworldtang/GPT_teacher-3.37M-cn
