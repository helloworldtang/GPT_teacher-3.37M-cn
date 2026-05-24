# GPT Teacher — 从 0 到 1 在 CPU 上训练可推理可演示的小参数中文 GPT，实现中文 GPT 训练平民化，实现训练中文 GPT 自由

## 为什么做这个项目

网上很多教程只讲了理论，但按理论跑出来的模型回答效果不好——**理论没有被实践验证，没有形成闭环**。初学者跑完之后心里没底："是我理解错了，还是模型就这样？"

这个项目的目标是：**让想入门机器学习的小伙伴，通过一个真正能跑通的项目入门。** 从训练到推理到验收，每一步都有明确的"跑对了"标准，理论能被实践验证。

具体来说：

- **一条命令跑通全流程**：构建分词器 → 训练 → 验收 → Web 演示，不需要记住多个步骤
- **训练完自动验证**：训练结束后自动跑验收测试，直观看到"模型学会了哪些问题"
- **Loss 曲线可视化**：训练过程生成 loss 曲线图，直观理解"训练在收敛"
- **Web 交互演示**：Gradio 界面直接体验模型效果，调节参数实时观察变化

## 项目适合谁

- 想入门机器学习但不知从何下手的小伙伴
- 学了理论但没跑过完整项目的学生
- 想快速理解 GPT 核心流程的开发者

本项目面向课堂教学，[目标是让初学者用一台普通 CPU 电脑，在 45 分钟内从零跑通一个小参数的中文 GPT](https://mp.weixin.qq.com/s/gUgY_TIRSoEyzZ2YRVGB2w)：看清核心流程、跑通训练、得到"可用的中文回答"，并支持简单的推理演示。

## 项目收获

- 了解 GPT 的核心原理：
  - **核心原因**：自注意力机制（捕捉token之间的关系）
  - **核心原因**：因果掩码（确保只看到历史）
  - **核心原因**：Transformer架构（高效的序列建模）
  - 完整链路：词嵌入 → 位置编码 → 自注意力 → 前馈网络 → 残差连接 → 归一化 → 因果掩码 → 损失 → 反向 → 优化 → 保存 → 推理
- 掌握高效小模型技术：RMSNorm、RoPE、GQA、SwiGLU、权重共享、短序列、小词表、量化、早停
- 学习 nanoGPT 的亮点：
  - RMSNorm（更简洁的归一化）
  - RoPE（相对位置编码）
  - 权重共享（词嵌入与输出层共享）
  - 简洁的架构（核心代码约500行）
  - Flash Attention优化（可选）
- 学会仅用 CPU 训练：控制模型/数据规模、梯度累积、学习率预热与退火、禁用无关耗时特性
- 学会可用答案保障：目标对齐（忽略前缀）、推理首步约束与后处理、停止词、提示词规范化
- 本项目的其他技术亮点：
  - MPS加速训练（比CPU快20-40倍）
  - 自动设备选择（auto：CUDA > MPS > CPU）
  - 早停机制（防止过拟合）
  - 学习率预热和余弦退火
  - 动态量化（加速CPU推理）
  - 默认配置路径（config/config.yml）
  - uv依赖管理（比pip快10-100倍）

## 代码结构

- `src/model.py`：GPT 核心（GQA自注意力/SwiGLU前馈/RMSNorm/RoPE/权重共享/Flash Attention）
- `src/data.py`：分词器加载、指令和 LM 数据处理、批处理与目标构造
- `src/train.py`：CPU/GPU/MPS 训练主循环、评估、保存、loss可视化、自动验收、早停
- `src/infer.py`：推理 CLI（温度、top-k、top-p、重复惩罚、停止词、输出清理）
- `src/evaluate.py`：验收测试脚本（自动跑 test.jsonl，输出 pass/fail 汇总）
- `src/tokenizer.py`：分词器加载（HF ByteLevel BPE）
- `src/build_tokenizer.py`：构建 HF ByteLevel BPE 中文分词器（带解码器，避免乱码）
- `config/config.yml`：统一管理模型、训练、数据、分词器与保存路径
- `data/*.jsonl`：教学数据（`{"prompt": "...", "completion": "..."}`）
- `run.py`：一键启动脚本（构建分词器→训练→验收→Web Demo）
- `src/visualize.py`：核心概念可视化（因果掩码/注意力权重/loss 曲线解读/模型结构图）

## 使用技术

- 模型结构：Decoder-only Transformer（因果自注意力 + GQA）
- 归一化：RMSNorm（简洁高效）
- 位置编码：RoPE（相对位置，计算高效）
- 前馈：SwiGLU（Llama风格，门控机制）
- 注意力：GQA（分组查询注意力，减少KV Cache）
- 权重共享：词嵌入与输出层共享，降参数且表示一致
- 训练：AdamW、学习率线性预热+余弦退火、早停、梯度裁剪
- 数据与分词：外置`jsonl`+HF ByteLevel BPE（显式设置 ByteLevel 解码器，避免乱码）
- 推理：温度/top-k/top-p 采样、重复惩罚、停止词、输出清理、提示规范化
- 量化：导出动态量化权重以加速 CPU 推理
- 设备支持：CPU、CUDA (NVIDIA GPU)、MPS (Apple Silicon Neural Engine)
- 依赖管理：uv（快速、可靠的依赖管理）
- Flash Attention：可选，提升训练速度约30%

## 如何仅用 CPU 训练

- 小模型+短序列：`n_layer=4, n_head=4, n_kv_head=2, n_embd=256, seq_len=128`
- 小词表：HF ByteLevel BPE（带解码器，约1500词）
- 梯度累积：`batch_size=16, micro_batch=4`（有效批 64）
- 学习率策略：线性预热 10%+余弦退火
- 资源控制：`torch.set_num_threads(os.cpu_count())`，DataLoader 禁用多进程（macOS 下更稳）
- 训练结束导出量化权重，演示更流畅

## 如何使用 MPS 加速训练（推荐）

- MPS (Apple Silicon Neural Engine)：比 CPU 快 20-40 倍
- 设备选择：`--device mps` 或 `--device auto`（自动检测最佳设备）
- 训练时间：20-30 分钟（相比 CPU 的 8-12 小时）
- 内存占用：与 CPU 相同

## GPT 训练是如何实现的

- 因果自注意力：下三角 mask 确保只看历史（`src/model.py:95–103`）
- Teacher Forcing：输入拼接"用户/prompt + 助手 + 答案"，仅对答案段计算损失（`src/data.py:24–31`）
- 目标对齐：忽略区间为`len(prefix)-1`，确保第一个答案 token 参与训练（`src/data.py:28–31`）
- 损失函数：`CrossEntropyLoss(ignore_index=-100)`（`src/train.py:50`）

## 如何保证"回答可用"

- 分词器解码器：HF BPE 设置 ByteLevel 解码器，彻底消除中文乱码（`src/build_tokenizer.py:6–8`）
- 目标对齐修复：忽略`len(prefix)-1`，避免答案错位（`src/data.py:28–31`）
- 推理约束：
  - 首步屏蔽`PAD/BOS/UNK`，前若干步屏蔽`EOS`，避免空答（`src/infer.py:41–50, 68–73`）
  - 开头清理：剔除首字符标点与空白（`src/infer.py:26–30, 82–83`）
  - 提示词规范化：去中文空格（`src/infer.py:35–37`）
  - 停止词/重复惩罚：控制尾部冗余与重复（`src/infer.py:68–73`）
- 训练耗时与评估：日志实时打印`eval loss`与`elapsed`，课时内可观测质量（`src/train.py:71–82`）

## 从 0 到 1：一步步跑通

> 每个步骤后都有 **「为什么」** 和 **「跑对了吗」** 两个小节，帮你理解理论、验证结果。

依赖 **Python 3.10-3.12**（torch 不支持 3.13+）。如果系统 Python 版本不对，uv 会自动处理，无需手动安装。

### 方式一：一键跑通全流程（推荐）

```bash
# 1. 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装 Python 3.11 并同步依赖（uv 自动管理，不需要改系统 Python）
uv python install 3.11
uv sync

# 3. 一键跑通：构建分词器 → 训练 → 验收 → Web 演示
uv run python run.py
```

**跑对了吗？** 应该看到验收结果 `结果: 5/6 通过 (83%)` 以上，然后浏览器打开 http://127.0.0.1:7860 体验 Web Demo。

```bash
# 指定设备
uv run python run.py --device mps    # Apple M 芯片
uv run python run.py --device cuda   # NVIDIA GPU
```

### 方式二：分步执行

#### 1. 安装依赖

**推荐方式：使用 uv（极速、自动同步）**

```bash
# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 Python 3.11 并同步依赖
uv python install 3.11
uv sync
```

**备选方式：使用 pip**

```bash
# 注意：需要 Python 3.10-3.12，不支持 3.13+
python3.11 -m pip install -r requirements.txt
```

#### 2. 构建中文分词器（HF ByteLevel BPE）

**为什么需要分词器？**
计算机不认识中文，需要先把文字切成一个个"词"（token），再转成数字。比如 "注意力机制" → [3个token] → [42, 187, 503]。好的分词器能让常用短语保持完整，模型更容易学会。

**为什么用 BPE？**
BPE（Byte Pair Encoding）会自动把经常一起出现的字组合成词。比如 "注意力" 出现多了就会被合并成一个 token，比逐字切分高效得多。

```bash
uv run python -m src.build_tokenizer
```

**跑对了吗？** 应该看到 `✓ 分词器已保存到: tokenizer/tokenizer.json`。

```bash
# 使用默认参数
uv run python -m src.build_tokenizer
```

#### 3. （可选）检查数据

**说明**：如果数据文件已经存在，这一步可以跳过。

```bash
# 当前训练数据：data/train.jsonl（约600条问答对）
# 验证数据：data/val.jsonl（5条问答对）
# 测试数据：data/test.jsonl（6条测试问题）

# 查看训练数据
head -n 3 data/train.jsonl

# 统计数据量
wc -l data/train.jsonl data/val.jsonl data/test.jsonl
```

#### 4. （可选）配置核对

**说明**：如果使用默认配置，这一步可以跳过。如果要自定义训练参数（如max_steps），这一步就是必要的。

- `config/config.yml` 中：
  - `data.train_path: data/train.jsonl`
  - `data.val_path: data/val.jsonl`
  - `tokenizer.type: hf_tokenizers`
  - `tokenizer.path: tokenizer/tokenizer.json`
  - `training.max_steps: 5000`（默认，MPS 约 5-6 分钟）
  - `training.device: auto|cpu|cuda|mps`

#### 5. 训练（CPU/GPU/MPS）

```bash
# 自动检测最佳设备（推荐：优先 CUDA > MPS > CPU）
uv run python -m src.train

# 使用 MPS (Apple Neural Engine，推荐）
uv run python -m src.train --device mps

# 使用 CPU
uv run python -m src.train --device cpu

# 使用 CUDA (NVIDIA GPU)
uv run python -m src.train --device cuda

# 禁用 Flash Attention
uv run python -m src.train --no-flash
```

- 观察日志：
  - 进度条显示训练进度和预计剩余时间
  - 每`eval_interval`步打印`eval loss`与累计`elapsed Xs`
  - 结束后保存：`checkpoints/last.pt`、`checkpoints/best.pt`、`checkpoints/quantized.pt`、`checkpoints/loss_curve.png`、`checkpoints/train_time.txt`
  - **训练完成后自动运行验收测试**，输出 pass/fail 汇总
- 设备入参：
  - `--device auto|cpu|cuda|mps`
  - `auto`：优先 CUDA，其次 MPS，最后 CPU
  - `cpu`：强制使用 CPU
  - `cuda`：强制使用 CUDA（如果可用）
  - `mps`：强制使用 MPS（如果可用）

**训练时发生了什么？**
1. **前向传播**：模型读入问题，尝试预测答案的每个字
2. **计算 Loss**：对比模型的预测和正确答案，差距越大 loss 越高
3. **反向传播**：根据差距调整模型参数，让下次预测更准
4. **重复**：一遍遍读数据，loss 越来越低，模型越来越准

**为什么要 warmup（学习率预热）？**
训练刚开始时模型是随机的，直接用大的学习率会"震荡"。先用小的学习率让模型稳定下来，再逐渐加大，就像开车先热车再加速。

**为什么要 cosine annealing（余弦退火）？**
训练后期模型已经接近最优，需要越来越小的步长来精细调整。余弦退火让学习率从大到小平滑下降，避免跳过最优点。

**为什么要 early stopping（早停）？**
如果验证 loss 连续 15 次评估都没改善，说明模型开始"死记硬背"训练数据了（过拟合）。及时停止，保留最好的模型。

**跑对了吗？**
- loss 应该从 ~2.5 快速下降到 ~0.5 以内（前 500 步）
- 最终 loss 应该在 0.01-0.05 之间
- 训练结束后应自动显示验收测试结果
- 如果 loss 不下降或震荡剧烈，检查数据是否正确、分词器是否重建

**验证你的理解**：运行 `uv run python -m src.visualize` 查看 loss 曲线解读图（`docs/loss_interpretation.png`），对照你的训练过程。

#### 6. 验收测试

训练完成后会自动运行。也可以手动运行：

```bash
uv run python -m src.evaluate
```

期望输出示例：
```
==================================================
验收测试：共 6 题
==================================================

[PASS] Q1: 什么是注意力机制？
  期望: 注意力机制通过计算相关性分配权重。
  实际: 注意力机制通过计算查询向量和键向量的相关性，为值向量分配权重...

[PASS] Q2: RoPE 的全称是什么？
  ...

==================================================
结果: 5/6 通过 (83%)
验收通过！模型已学会回答测试集中的大部分问题。
==================================================
```

**这就是闭环！**
- 理论说：用 Transformer + 训练数据 → 模型能学会回答问题
- 实践验证：训练后问模型 "什么是注意力机制？" → 它真的回答对了
- **理论预测的结果 = 实际跑出来的结果 → 闭环成立**

如果验收不通过（通过率 < 50%），说明理论和实践之间有差距，检查：
1. 分词器是否用训练数据重建了？（`uv run python -m src.build_tokenizer`）
2. 训练步数够不够？（至少 2000 步）
3. Loss 是否降到了 0.1 以下？

#### 7. 推理演示

**方式一：Web 界面演示（推荐）**
```bash
# 启动 Web 演示 (uv 会自动确保依赖已同步)
uv run python -m src.web_demo
```

打开浏览器访问 http://127.0.0.1:7860 ，可以：
- 点击预设问题按钮快速体验
- 调节 Temperature / Top-K / Top-P 参数观察变化
- 查看每次推理的耗时

**Temperature 是什么？**
Temperature 控制模型输出的"确定性"。0 = 每次选概率最高的词（最稳定），1 = 按概率随机选（更多样）。试试同一个问题用 Temperature=0 和 1 的区别——这就是"确定性 vs 创造性"的直观体现。

**为什么模型能回答没见过的问题？**
因为模型学到了模式，不是死记硬背。比如学会了 "15×6=90" 和 "125÷5=25"，你问 "8×7=?" 它虽然没见过，但能基于学到的计算模式生成答案。

**跑对了吗？**
- 问训练数据中的问题（如"什么是注意力机制？"）→ 应该得到和训练数据语义一致的回答
- 问新问题 → 可能回答不完全对，但应该是中文通顺、和问题相关的
- 如果回答和问题完全无关或大量重复 → 检查 repetition_penalty（Web Demo 默认 1.5）

**方式二：命令行推理**
```bash
# 解释 RoPE 的作用 (推荐演示问题)
uv run python -m src.infer --prompt "解释RoPE的作用" --ckpt checkpoints/best.pt --temperature 0.0

# 解释蒸馏水与纯水区别
uv run python -m src.infer --prompt "解释蒸馏水与纯水区别？" --ckpt checkpoints/best.pt --temperature 0.0

# 什么是注意力机制
uv run python -m src.infer --prompt "什么是注意力机制？" --ckpt checkpoints/best.pt --temperature 0.0

# 什么是 RMSNorm
uv run python -m src.infer --prompt "什么是RMSNorm？" --ckpt checkpoints/best.pt --temperature 0.0
```

#### 8. （可选）量化权重推理（仅 CPU）

```bash
# 使用量化权重在 CPU 上推理演示
uv run python -m src.infer --prompt "解释RoPE的作用" --ckpt checkpoints/quantized.pt --temperature 0.0 --device cpu
```

## 核心代码参考（行号）

- 因果 mask 与前向（`src/model.py:95–103`）
- RoPE 相对位置（`src/model.py:18–31`）
- 自注意力前向（`src/model.py:41–58`）
- Flash Attention（`src/model.py:32–38, 111–112`）
- 指令目标构造与忽略前缀（`src/data.py:24–31`）
- 训练主循环（`src/train.py:56–80`）
- 推理生成与采样管线（`src/infer.py:32–83`）
- ByteLevel 解码器设置（`src/build_tokenizer.py:6–8`）

> 说明：`src/model.py`核心约百行即可完整呈现 GPT 最小闭环；其余文件各自职责清晰，便于教学与修改。

## 核心代码解析（从架构 → 流程 → 方法级）

- 整体架构
  - `src/model.py`：定义 GPT 的模块化组件（RMSNorm、RoPE、自注意力、前馈、残差与权重共享、Flash Attention）
  - `src/data.py`：将`{"prompt","completion"}`样本转换为模型输入与训练目标（teacher forcing）
  - `src/train.py`：CPU/GPU/MPS 训练主循环、评估与保存（含动态量化、早停与耗时统计）
  - `src/infer.py`：命令行推理与采样策略（含输出清理与停止词）
  - `src/tokenizer.py`/`src/build_tokenizer.py`：HF ByteLevel BPE 分词器加载与构建（含 ByteLevel 解码器）

- 训练流程（管线）
  - 加载配置与分词器（`src/train.py:21–26`，`src/tokenizer.py:20–39`）
  - 构建数据集与 DataLoader（`src/data.py:66–70`，`src/train.py:39–40`）
  - 前向与损失（`src/train.py:60–62`）→ 反向与梯度累积（`src/train.py:62–68`）→ 优化与调度（`src/train.py:65–69`）
  - 评估与保存（`src/train.py:71–80`）→ 导出量化（`src/train.py:81–82`）

- 模型方法级解析
  - 归一化：`RMSNorm.forward`（`src/model.py:13–16`）用均方根缩放激活，结构简洁、计算高效
  - 位置编码：`rope`（`src/model.py:18–31`）将相对位置信息旋转注入 Q/K，提高位置泛化
  - 自注意力：`SelfAttention.forward`（`src/model.py:41–58`）Q/K/V 分解 → 注意力权重 → 输出投影；含因果 mask 与 Dropout
  - Flash Attention：`flash_attention`（`src/model.py:32–38`）优化注意力计算，提升训练速度约30%
  - 残差块：`Block.forward`（`src/model.py:81–84`）注意力与前馈的残差堆叠，RMSNorm 在前
  - 模型前向：`GPT.forward`（`src/model.py:95–103`）嵌入 → 堆叠 Block → 归一化 → LM Head（共享权重）

- 数据与目标（方法级）
  - 指令样本构造：`InstructDataset.__init__`（`src/data.py:10–35`）拼接`用户:prompt\n助手:`为前缀，答案接在后；截断至`seq_len`
  - 目标构造：`tar = ids[1:] + [EOS]`（`src/data.py:24–27`）并忽略`len(prefix)-1`位置（`src/data.py:28–31`），只对答案段计算损失
  - 批处理：`collate`（`src/data.py:54–64`）按`seq_len`补齐并返回张量，忽略目标用`-100`

- 推理与采样（方法级）
  - `generate`（`src/infer.py:32–83`）：
    - 规范化提示词（去中文空格）（`src/infer.py:35–37`）
    - 前 5 步屏蔽`EOS`，首步屏蔽`PAD/BOS/UNK`（`src/infer.py:41–50, 68–73`）
    - 采样：温度、top-k、top-p、重复惩罚、停止词
    - 输出清理：剔除开头标点与空白（`src/infer.py:26–30, 82–83`）

- 可用性策略（总结）
  - 乱码防御：HF BPE 设置 ByteLevel 解码器（`src/build_tokenizer.py:6–8`）
  - 目标对齐：忽略`len(prefix)-1`确保第一个答案 token 参与训练（`src/data.py:28–31`）
  - 空答防御：生成初期屏蔽`EOS`与无效 token；必要时添加`--stop_strings`
  - A/B 验证与权重切换：扩充数据+2000 步训练，固定问题验证通过后才切换默认权重

## 核心代码行数统计

- `src/model.py`：193 行（GQA + SwiGLU）
- `src/data.py`：64 行
- `src/train.py`：332 行（含 loss 可视化、自动验收、进度条）
- `src/infer.py`：170 行（含 greedy 模式重复惩罚）
- `src/evaluate.py`：95 行（验收测试脚本）
- `src/tokenizer.py`：39 行
- `src/build_tokenizer.py`：43 行
- 合计：936 行

## 实测与分析（答案可用、与问题相关）

- 环境与数据：Apple M 系列（MPS 加速），使用 600 条训练集与 5 条验证集，MPS 训练 5000 步。
- 耗时与结果：
  - MPS 训练：约 5-6 分钟
  - CPU 训练：约 30-60 分钟
  - 验收测试通过率：100%（6/6），中文回答干净完整
- 配置要点（与 `config/config.yml` 一致）：
  - 默认配置：`n_layer=4, n_head=4, n_kv_head=2, n_embd=256, seq_len=128`（约 3.16M 参数，CPU 友好）
  - GQA 分组查询注意力 + SwiGLU 前馈层（Llama 风格架构）
  - HF ByteLevel BPE 分词器（约 1500 词表）
  - 权重共享、RMSNorm、RoPE
  - AdamW+预热+余弦+早停
  - 梯度累积与裁剪
- 效率核心原因（技术角度）：
  - 小模型与短序列降低注意力计算开销；权重共享减少参数与内存占用；RMSNorm 与 RoPE 计算简洁且稳定
  - MPS 加速训练：20-40倍加速，内存占用相同
  - Flash Attention优化：提升训练速度约30%
  - 仅 CPU 优化：限制 DataLoader 开销、设线程数；预热+退火提升早期收敛效率；梯度累积在小内存下获得有效大批次
  - 可用性保障：目标对齐只对答案段计算损失；推理阶段首步/初期屏蔽无效与 `EOS`、提示规范化、输出清理与停止词，避免空答与乱码

## 参数规模与计算

### 默认配置（3.34M 参数，CPU 友好）

- 设定：嵌入维度 `d=256`、层数 `L=4`、头数 `H=4`、KV头数 `H_kv=2`（GQA）、词表大小 `V≈1509`
- 架构：GQA + SwiGLU + RMSNorm + RoPE + 权重共享
- 参数：约 3,337,728（3.34M）
- 实际训练验证：MPS 2-3 分钟，验收通过率 83%

### 高配参考

- 设定：`d=384, L=6, V=1509`（可通过修改 `config/config.yml` 切换）
- 需要 MPS 或 GPU，训练时间约 10-20 分钟

## 常见问题与排查

- **`uv sync` 报错 torch 不支持当前 Python 版本**：运行 `uv python install 3.11` 后重试。torch 只支持 Python 3.10-3.12
- **`uv run python -m src.evaluate` 报错未找到模型文件**：需要先运行训练 `uv run python -m src.train`，训练完成后会自动验收
- **`uv run python -m src.evaluate` 报错找不到模块**：先运行 `uv sync` 安装依赖
- 乱码输出：确保已运行 `src/build_tokenizer.py`，且其设置了 `ByteLevel` 解码器；查看推理是否使用 HF 分词器路径
- 空输出：提升步数（≥1500），并确认推理阶段首步屏蔽与停止词设置是否生效
- 两个不同问题给出同一答案：小数据短训记忆偏移，提升步数并适度扩充问句变体；或在推理时加`--stop_strings`
- MPS 训练失败：检查 `torch.backends.mps.is_available()` 是否为 True
- CUDA 训练失败：检查 `torch.cuda.is_available()` 是否为 True
- 训练太慢：尝试使用 MPS 或 GPU 而不是 CPU

## 教学建议（45 分钟）

1. 看可视化图（5 分钟）
   - 运行 `uv run python -m src.visualize` 生成 4 张图
   - 对照 `docs/causal_mask.png` 讲因果掩码
   - 对照 `docs/model_structure.png` 讲数据流
2. 训练与观察（15–20 分钟）
   - 运行 `uv run python run.py`，观察 loss 下降
   - 对照 `docs/loss_interpretation.png` 理解每个阶段
3. 验收测试（5 分钟）
   - 训练结束自动验收 → 看到理论变成现实
4. Web Demo 互动（10 分钟）
   - 调节 Temperature，观察从"精确"到"随机"
   - 问训练数据中的问题 vs 新问题，对比效果
5. 实验作业（5 分钟）

## 实验作业

### 入门级（理解参数）

1. **温度实验**：在 Web Demo 中，分别用 Temperature=0、0.5、1.0 问同一个问题，观察回答的变化
   - 思考：为什么 Temperature=0 时回答最稳定？
2. **步数实验**：把 `config/config.yml` 中的 `max_steps` 改成 100，重新训练，观察验收测试结果
   - 思考：步数太少会怎样？loss 曲线是什么形状？

### 进阶级（修改数据）

3. **教模型新知识**：在 `data/prepare_data.py` 中添加一条新的问答（比如"什么是深度学习？"），重新运行 `python data/prepare_data.py` 生成数据，再训练，看模型能否学会
   - 提示：训练后运行验收测试验证
4. **数据量实验**：把 `prepare_data.py` 中的重复次数从 20 改成 5，观察训练效果变化

### 挑战级（修改模型）

5. **模型大小实验**：把 `config/config.yml` 中 `n_layer` 从 4 改成 2，重新训练，观察 loss 和验收结果的变化
   - 思考：更小的模型能学会同样的知识吗？
6. **序列长度实验**：把 `seq_len` 从 128 改成 64，训练后观察哪些问题的回答变差了
   - 思考：为什么序列长度会影响长答案的质量？

## 期望结果参考

### 训练过程（默认配置，MPS）

```
模型参数量: 3,161,600 (3.16M)
设备: mps | 序列长度: 128 | 训练集: 600 条
==================================================
训练: 100%|████████| 5000/5000 [05:27<00:00, loss=0.0049]
eval loss 0.0049 elapsed 327.1s

✓ 训练完成，总用时: 327.1s (5.5min)
  最佳验证损失: 0.0049 (step 4950)
  Loss 曲线已保存: checkpoints/loss_curve.png
```

### 验收测试

```
==================================================
验收测试：共 6 题
==================================================
[PASS] Q1: 什么是注意力机制？
  实际: 注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。
[PASS] Q2: RoPE 是什么？
  实际: RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。
[PASS] Q3: 15 乘以 6 等于多少？
  实际: 15 乘以 2 是 30，30 乘以 3 是 90。
[PASS] Q4: 你是谁？
  实际: 我是一个基于 Transformer 的小型 GPT 教学演示模型。
[PASS] Q5: 太阳系有哪些行星？
  实际: 八大行星：水星、金星、火星、海王星。
[PASS] Q6: 蒸馏水和纯水有什么区别？
  实际: 蒸馏水通过蒸馏冷凝制得，纯水杂质极低。

结果: 6/6 通过 (100%)
验收通过！模型已学会回答测试集中的大部分问题。
==================================================
```

### 命令行推理示例

```
$ uv run python -m src.infer --prompt "什么是注意力机制？" --temperature 0.0
注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。
```

如果你看到类似上面的输出，说明训练成功，闭环已跑通！

---

祝学习顺利！如需更高质量中文问答效果，可以在 `data/prepare_data.py` 中添加更多问答对，然后重新生成数据、重建分词器、训练。课堂演示建议使用 `temperature=0.0`，Web Demo 体验最佳。
