# Distill — 知识蒸馏实验

用大模型当 teacher，探索小模型能否通过蒸馏获得新能力。实验结论：**三个方向全部失败**——但失败本身比成功更有价值。

## 项目结构

```
distill/
├── train.py         知识蒸馏训练（支持标准训练和 logits 级 KD）
├── data.py          数据生成工具（teacher 增强 / 主题扩展）
├── infer.py         蒸馏模型推理
├── data/            蒸馏生成数据（gitignored）
└── checkpoints/     蒸馏产出（gitignored）
```

## 三个方向的实验结论

### 方向一：增大参数（3M → 12M）

| 配置 | 参数量 | 验证 loss | 验收 | 泛化（15 个新问题） |
|------|--------|----------|------|-------------------|
| 原始 | 3M | 0.005 | 6/6 | 0/15（全返回同一答案） |
| 加宽 | 12M | 0.0035 | 6/6 | 0/15（全返回同一答案） |

**结论**：参数量翻 4 倍，泛化能力零提升。瓶颈在数据多样性，不在模型大小。

### 方向二：数据级蒸馏（teacher 生成答案 → student 学习）

| Teacher | 问题 |
|---------|------|
| `uer/gpt2-chinese-cluecorpussmall` (102M) | Base 模型只会续写不会答题 |
| `Qwen2.5-1.5B-Instruct` | 常识题正确，ML 专业概念（RoPE）自信胡说 |

用 Qwen 生成的数据训练 12M student（byte tokenizer），1.7 分钟 early stopping 触发，输出全是乱码。

**根因**：tokenizer 不匹配。Qwen 的 tiktoken（vocab=151K）每个汉字 1-2 token，byte tokenizer（vocab=259）每个汉字 3 token。信息密度太低。

### 方向三：logits 级蒸馏（teacher/student 共享 tokenizer）

| 配置 | 参数量 |
|------|--------|
| Teacher: DeepSeek-R1-Distill-Qwen-1.5B | 1777M |
| Student: GPT (vocab=151,643) | 41.8M |

训练 loss 从 340 降到 6.5，看起来成功。但输出全是乱码和重复词。

**根因**：embedding 占 93% 参数（38.9M / 41.8M），transformer 层只有 2.9M。模型只学会了输出高频 token 来"匹配" teacher 的概率分布。

### 不可能三角

三个失败指向同一个结论：**小模型蒸馏面临三个约束，无法同时满足。**

1. **模型容量** — 参数太少学不到复杂模式
2. **词表大小** — 大词表吃掉 embedding 容量；小词表无法与 teacher 对齐
3. **Tokenizer 效率** — byte-level（vocab=259）每个汉字 3 token；大 vocab（151K）又吃光参数

## 使用方法

### 数据生成

```bash
# 主题扩展：生成 30 个主题的训练数据
uv run python -m distill.data --method expand --topics 30

# Teacher 增强：用 Qwen 生成训练答案
uv run python -m distill.data --method augment --teacher Qwen/Qwen2.5-1.5B-Instruct
```

### logits 级蒸馏训练

```bash
uv run python -m distill.train --kd --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_embd` | 512 | Student 嵌入维度 |
| `--n_head` | 8 | 注意力头数 |
| `--alpha` | 0.5 | CE loss 权重（1-alpha 为 KL 权重） |
| `--temperature` | 2.0 | 蒸馏温度 |
| `--max_steps` | 5000 | 最大训练步数 |

### 标准训练（委托给 train 模块）

```bash
uv run python -m distill.train
```

### 蒸馏模型推理

```bash
uv run python -m distill.infer
uv run python -m distill.infer --prompt "什么是深度学习？"
```

## 工程踩坑记录

| 坑 | 现象 | 解法 |
|----|------|------|
| `tokenizer.vocab_size ≠ model.vocab_size` | KL 散度维度不匹配 | 用 `model.config.vocab_size` |
| Python `min()` 单参数传 int | `TypeError: 'int' object is not iterable` | 始终传两个参数 |
| Base 模型当 teacher | 输出全是续写不是答题 | 只用 Instruct 版本 |
| `PYTHONUNBUFFERED` 未设 | 后台训练看不到实时输出 | `PYTHONUNBUFFERED=1` |
| val loss 下降但输出是垃圾 | embedding 占 93%，模型只学高频词 | 一定要跑推理验证 |

详细实验过程见系列文章第 9 篇：《知识蒸馏：三个方向，全部失败》
