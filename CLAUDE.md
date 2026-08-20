# GPT Teacher 项目规范

## 代码规范

- 写任何 Python 代码前，先加载 `python-code-style` skill 并遵循其规范。
- 本项目已落地的工具链（配置见 `pyproject.toml`，提交前必须三清）：
  - `uv run ruff check .` + `uv run ruff format --check .`（120 列，E/W/F/I/B/C4/UP/SIM）
  - `uv run mypy .`（strict，tests 豁免 untyped-def）
- 公共 API 用 Google 风格 docstring；导入一律绝对导入。

## 改代码红线（用户 2026-08-19 指定）

任何改动完成前，必须全部通过：

1. `uv run pytest core/tests/`（56 个测试）
2. `uv run python -m core.evaluate` 验收 6/6
3. `train/README.md` 里的示例命令（推理等）行为正常
4. 涉及生成行为改动时：`uv run python train/scripts/regression.py` 金样本 27 条全过（基线勿随意 `--update` 重建）
