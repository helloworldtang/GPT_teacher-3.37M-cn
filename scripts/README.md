# 脚本目录

本目录包含项目的实用脚本和工具。

## 脚本列表

### `augment_data.py`
数据增强脚本，通过同义词替换、句子重组等方式扩充训练数据。

### `fewshot_infer.py`
少样本推理脚本，支持通过提供示例来引导模型生成更准确的回答。

## 使用方法

```bash
# 数据增强
uv run python -m scripts.augment_data

# 少样本推理
uv run python -m scripts.fewshot_infer --prompt "你的问题"
```
