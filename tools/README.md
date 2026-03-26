# 工具目录

本目录包含数据处理和优化工具。

## 工具列表

### 数据处理
- `batch_test_inference.py` - 批量测试推理
- `fix_data_distribution.py` - 修复数据分布
- `fix_format_error.py` - 修复格式错误
- `optimize_dataset.py` - 优化数据集
- `standardize_data.py` - 标准化数据

## 使用方法

```bash
# 批量测试推理
uv run python -m tools.batch_test_inference

# 数据修复和优化
uv run python -m tools.fix_data_distribution
uv run python -m tools.fix_format_error
uv run python -m tools.optimize_dataset
uv run python -m tools.standardize_data
```
