# GPT Teacher

从 0 到 1 在 CPU 上训练可推理可演示的小参数中文 GPT。

## 项目结构

```
GPT_teacher-3.37M-cn/
├── core/       ← 共享库（GPT 模型、分词器、推理引擎、可视化）
├── train/      ← 训练模块（数据、配置、checkpoint、Web Demo）→ [README](train/README.md)
├── distill/    ← 蒸馏模块（知识蒸馏训练与推理）→ [README](distill/README.md)
├── pyproject.toml
└── uv.lock
```

| 模块 | 说明 | 快速开始 |
|------|------|----------|
| **train** | 3.37M 参数中文 GPT 的完整训练流程 | `uv run python train/run.py` |
| **distill** | 用大模型做 teacher 的知识蒸馏实验 | `uv run python -m distill.train --kd --teacher Qwen/Qwen2.5-1.5B-Instruct` |
| **core** | 共享代码：模型定义、分词器、推理引擎、评估、可视化 | 被上面两个模块引用 |

## 快速开始

```bash
# 安装依赖（需要 Python 3.10-3.12）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv sync

# 一键跑通：分词器 → 训练 → 验收 → Web Demo
uv run python train/run.py
```

详细使用说明见各模块 README：
- [train/README.md](train/README.md) — 训练、验收、推理、Web Demo 的完整教程
- [distill/README.md](distill/README.md) — 知识蒸馏三种方向的实验记录

## 技术栈

- 架构：Decoder-only Transformer（GQA + SwiGLU + RMSNorm + RoPE + 权重共享）
- 设备：CPU / MPS (Apple Silicon) / CUDA (NVIDIA GPU)
- 依赖管理：uv
- 单元测试：44 个（`pytest core/tests/`）

*项目地址：https://github.com/helloworldtang/GPT_teacher-3.37M-cn*
