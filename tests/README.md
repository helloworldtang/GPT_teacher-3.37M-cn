# 测试目录

本目录包含项目的测试脚本和验证工具。

## 测试脚本

### 检查脚本
- `check_checkpoint.py` - 检查模型检查点文件
- `check_data.py` - 检查训练数据格式
- `check_original_data.py` - 检查原始数据
- `check_raw_completion.py` - 检查原始完成数据
- `check_test_answers.py` - 检查测试答案

### 验证脚本
- `verify_data.py` - 验证数据完整性
- `verify_inference_quality.py` - 验证推理质量

### 测试脚本
- `test_generalization.py` - 测试模型泛化能力
- `test_inference_format.py` - 测试推理输出格式

## 使用方法

```bash
# 运行单个测试
uv run python -m tests.check_data

# 运行所有检查脚本
uv run python -m tests.check_checkpoint
uv run python -m tests.check_data
uv run python -m tests.verify_data
```
