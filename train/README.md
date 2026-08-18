# Train — 从零训练一个能说中文的 GPT

3.37M 参数的中文 GPT 教学模型，支持 CPU/MPS/CUDA 训练，内置验收测试和 Web 交互演示。

## 项目结构

```
train/
├── train.py         训练主循环（早停、loss 曲线、自动验收）
├── config.yml       模型和训练配置
├── tokenizer.json   ByteLevel BPE 分词器
├── data/            训练数据
│   ├── train.jsonl  训练集（600 条）
│   ├── val.jsonl    验证集（5 条）
│   └── test.jsonl   测试集（6 条）
├── checkpoints/     训练产出（gitignored）
│   ├── best.pt      最佳模型
│   └── loss_curve.png
├── web_demo.py      Gradio Web 交互演示
├── run.py           一键启动脚本
├── Dockerfile       Docker 部署
├── scripts/         工具脚本（推送 HF Hub）
├── docs/            可视化图表
└── notebooks/       教学笔记本
```

## 快速开始

### 方式一：一键跑通（推荐）

```bash
# 1. 安装依赖（项目根目录执行）
uv sync

# 2. 一键跑通：分词器 → 训练 → 验收 → Web Demo
uv run python train/run.py
```

**跑对了吗？** 应该看到 `结果: 5/6 通过 (83%)` 以上，然后浏览器打开 http://127.0.0.1:7860 。

### 方式二：分步执行

#### 1. 构建分词器

```bash
uv run python -m core.build_tokenizer
```

#### 2. 训练

```bash
# 自动检测最佳设备（CUDA > MPS > CPU）
uv run python -m train.train

# 指定设备
uv run python -m train.train --device mps    # Apple M 芯片（推荐）
uv run python -m train.train --device cuda   # NVIDIA GPU
uv run python -m train.train --device cpu    # 纯 CPU
```

训练结束后自动运行验收测试并保存最佳模型到 `train/checkpoints/best.pt`。

**跑对了吗？**
- loss 应从 ~2.5 快速降到 ~0.5 以内（前 500 步）
- 最终 loss 在 0.01-0.05 之间
- 验收通过率 >= 83%

#### 3. 验收测试

```bash
uv run python -m core.evaluate
```

#### 4. Web Demo

```bash
uv run python -m train.web_demo
```

打开 http://127.0.0.1:7860 ，可以调节 Temperature / Top-K / Top-P 观察模型输出变化。

#### 5. 命令行推理

```bash
uv run python -m core.infer --prompt "什么是注意力机制？" --temperature 0.0
uv run python -m core.infer --prompt "RoPE 是什么？" --temperature 0.0
uv run python -m core.infer --prompt "15 乘以 6 等于多少？" --temperature 0.0
```

## 配置说明

`train/config.yml` 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_layer` | 4 | Transformer 层数 |
| `n_head` | 4 | 注意力头数 |
| `n_kv_head` | 2 | KV 头数（GQA） |
| `n_embd` | 256 | 嵌入维度 |
| `seq_len` | 128 | 序列长度 |
| `max_steps` | 5000 | 训练步数 |
| `lr` | 0.001 | 学习率 |
| `batch_size` | 16 | 有效批大小 |

总参数量约 **3.37M**（含权重共享），MPS 训练约 3-6 分钟。

## 训练时间参考

| 设备 | 耗时 |
|------|------|
| MPS (Apple M) | 3-6 分钟 |
| CUDA (GPU) | 5-10 分钟 |
| CPU | 30-60 分钟 |

## 自定义训练数据

编辑 `train/data/prepare_data.py`，添加新的问答对：

```python
{"prompt": "你的问题", "completion": "你的答案"},
```

然后重新生成数据、构建分词器、训练：

```bash
uv run python train/data/prepare_data.py
uv run python -m core.build_tokenizer
uv run python -m train.train
```

## 期望输出

```
模型参数量: 3,147,520 (3.15M)
设备: mps | 序列长度: 128 | 训练集: 600 条
==================================================
训练: 100%|████████| 5000/5000 [05:27<00:00]

==================================================
验收测试：共 6 题
==================================================
[PASS] Q1: 什么是注意力机制？
  实际: 注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。
[PASS] Q2: RoPE 是什么？
  实际: RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。
...
结果: 6/6 通过 (100%)
```

## 常见问题

- **torch 版本不兼容**：`uv python install 3.11` 后重试，torch 只支持 3.10-3.12
- **乱码输出**：重新构建分词器 `uv run python -m core.build_tokenizer`
- **空输出**：训练步数至少 1500 步
- **MPS 不可用**：检查 `torch.backends.mps.is_available()`
