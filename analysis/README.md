# 分析目录

本目录包含数据分析和调试脚本。

## 分析脚本

### 数据分析
- `analyze_data_topics.py` - 分析数据主题
- `analyze_model_params.py` - 分析模型参数
- `analyze_original_data.py` - 分析原始数据
- `analyze_original_topics.py` - 分析原始主题
- `analyze_short_questions.py` - 分析短问题

### 调试脚本
- `debug_forward.py` - 调试前向传播
- `debug_special_tokens.py` - 调试特殊token
- `debug_tokenization.py` - 调试分词
- `investigate_token_180.py` - 调查特定token

## 使用方法

```bash
# 数据分析
uv run python -m analysis.analyze_data_topics
uv run python -m analysis.analyze_model_params

# 调试
uv run python -m analysis.debug_forward
uv run python -m analysis.debug_tokenization
```
