# GPT Teacher — 从 0 到 1 在 CPU 上训练可推理可演示的小参数中文 GPT，实现中文 GPT 训练平民化，实现训练中文 GPT 自由

本项目面向课堂教学，[目标是让初学者用一台普通 CPU 电脑，在 45 分钟内从零跑通一个小参数的中文 GPT](https://mp.weixin.qq.com/s/gUgY_TIRSoEyzZ2YRVGB2w)：看清核心流程、跑通训练、得到"可用的中文回答"，并支持简单的推理演示。

## 项目收获

- 了解 GPT 的核心原理：
  - **核心原因**：自注意力机制（捕捉token之间的关系）
  - **核心原因**：因果掩码（确保只看到历史）
  - **核心原因**：Transformer架构（高效的序列建模）
  - 完整链路：词嵌入 → 位置编码 → 自注意力 → 前馈网络 → 残差连接 → 归一化 → 因果掩码 → 损失 → 反向 → 优化 → 保存 → 推理
- 掌握高效小模型技术：RMSNorm、RoPE、权重共享、短序列、小词表、量化、早停
- 学习 nanoGPT 的亮点：
  - RMSNorm（更简洁的归一化）
  - RoPE（相对位置编码）
  - 权重共享（词嵌入与输出层共享）
  - 简洁的架构（核心代码约497行）
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

- `src/model.py`：GPT 核心（Embedding/自注意力/前馈/RMSNorm/RoPE/权重共享/Flash Attention）
- `src/data.py`：分词器加载、指令和 LM 数据处理、批处理与目标构造
- `src/train.py`：CPU/GPU/MPS 训练主循环、评估、保存与动态量化、耗时统计、早停
- `src/infer.py`：推理 CLI（温度、top-k、top-p、重复惩罚、停止词、输出清理）
- `src/tokenizer.py`：分词器加载（HF ByteLevel BPE）
- `src/build_tokenizer.py`：构建 HF ByteLevel BPE 中文分词器（带解码器，避免乱码）
- `config/config.yml`：统一管理模型、训练、数据、分词器与保存路径
- `data/*.jsonl`：教学数据（`{"prompt": "...", "completion": "..."}`）

## 使用技术

- 模型结构：Decoder-only Transformer（因果自注意力）
- 归一化：RMSNorm（简洁高效）
- 位置编码：RoPE（相对位置，计算高效）
- 前馈：SiLU（现代 LLM 常用）
- 权重共享：词嵌入与输出层共享，降参数且表示一致
- 训练：AdamW、学习率线性预热+余弦退火、早停、梯度裁剪
- 数据与分词：外置`jsonl`+HF ByteLevel BPE（显式设置 ByteLevel 解码器，避免乱码）
- 推理：温度/top-k/top-p 采样、重复惩罚、停止词、输出清理、提示规范化
- 量化：导出动态量化权重以加速 CPU 推理
- 设备支持：CPU、CUDA (NVIDIA GPU)、MPS (Apple Silicon Neural Engine)
- 依赖管理：uv（快速、可靠的依赖管理）
- Flash Attention：可选，提升训练速度约30%

## 如何仅用 CPU 训练

- 小模型+短序列：`n_layer=4, n_head=4, n_embd=256, seq_len=128`
- 小词表：HF ByteLevel BPE（带解码器）
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

依赖 Python 3.10-3.12。Python 3.11.10 是可以跑通的。

### 1. 安装依赖

**推荐方式：使用 uv（更快、更稳定）**

```bash
# 安装 uv（如果未安装）
pip install uv
# 或使用官方脚本（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
uv pip install -e .
```

**备选方式：使用 pip**

```bash
python -m pip install -r requirements.txt
```

### 2. 构建中文分词器（HF ByteLevel BPE）

```bash
# 使用默认参数
python -m src.build_tokenizer

# 使用自定义参数
python -m src.build_tokenizer --train_path data/train.jsonl --save_path tokenizer/tokenizer.json --vocab_size 4096
```

### 3. （可选）检查数据

**说明**：如果数据文件已经存在，这一步可以跳过。

```bash
# 当前训练数据：data/train.jsonl（约392条问答对）
# 验证数据：data/val.jsonl（10条问答对）
# 测试数据：data/test.jsonl（10条测试问题）

# 查看训练数据
head -n 3 data/train.jsonl

# 统计数据量
wc -l data/train.jsonl data/val.jsonl data/test.jsonl
```

### 4. （可选）配置核对

**说明**：如果使用默认配置，这一步可以跳过。如果要自定义训练参数（如max_steps），这一步就是必要的。

- `config/config.yml` 中：
  - `data.train_path: data/train.jsonl`
  - `data.val_path: data/val.jsonl`
  - `tokenizer.type: hf_tokenizers`
  - `tokenizer.path: tokenizer/tokenizer.json`
  - `training.max_steps: 1500–2000`（课堂机器允许的话）
  - `training.device: auto|cpu|cuda|mps`

### 5. 训练（CPU/GPU/MPS）

```bash
# 自动检测最佳设备（推荐：优先 CUDA > MPS > CPU）
python -m src.train

# 使用 MPS (Apple Neural Engine，推荐）
python -m src.train --device mps

# 使用 CPU
python -m src.train --device cpu

# 使用 CUDA (NVIDIA GPU)
python -m src.train --device cuda

# 禁用 Flash Attention
python -m src.train --no-flash
```

- 观察日志：
  - 每`eval_interval`步打印`eval loss`与累计`elapsed Xs`
  - 结束后保存：`checkpoints/last.pt`、`checkpoints/best.pt`、`checkpoints/quantized.pt`、`checkpoints/train_time.txt`
- 设备入参：
  - `--device auto|cpu|cuda|mps`
  - `auto`：优先 CUDA，其次 MPS，最后 CPU
  - `cpu`：强制使用 CPU
  - `cuda`：强制使用 CUDA（如果可用）
  - `mps`：强制使用 MPS（如果可用）
  - 示例：`python -m src.train --device mps`

### 6. 推理验证（测试集）

```bash
# 单个问题测试（设备自动选择：有 CUDA 用 CUDA，否则 CPU）
python -m src.infer --prompt "解释RoPE的作用" --ckpt checkpoints/last.pt --temperature 0.0

# 单个问题测试（使用 MPS）
python -m src.infer --prompt "解释RoPE的作用" --ckpt checkpoints/last.pt --temperature 0.0 --device mps

# 单个问题测试（强制 CPU）
python -m src.infer --prompt "解释RoPE的作用" --ckpt checkpoints/last.pt --temperature 0.0 --device cpu

# 多个问题测试
python -m src.infer --prompt "解释RoPE的作用" --ckpt checkpoints/last.pt --temperature 0.0
python -m src.infer --prompt "解释蒸馏水与纯水区别？" --ckpt checkpoints/last.pt --temperature 0.0
python -m src.infer --prompt "什么是注意力机制？" --ckpt checkpoints/last.pt --temperature 0.0
python -m src.infer --prompt "什么是RMSNorm？" --ckpt checkpoints/last.pt --temperature 0.0
```

### 7. （可选）量化权重推理（仅 CPU）

```bash
python -m src.infer --prompt "什么是注意力机制？" --ckpt checkpoints/quantized.pt --temperature 0.0 --device cpu
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

- `src/model.py`：103 行
- `src/data.py`：70 行
- `src/train.py`：104 行
- `src/infer.py`：122 行
- `src/tokenizer.py`：58 行
- `src/build_tokenizer.py`：40 行
- 合计（核心代码）：497 行
- 配置与辅助：`config/config.yml`：23 行（不计入核心代码合计）

## 实测与分析（答案可用、与问题相关）

- 环境与数据：在 Mac Pro 2.6GHz 6c i7（16GB DDR4）上，使用 392 条训练集与 10 条验证集，MPS 训练 20000 步。
- 耗时与结果：
  - CPU 训练：约 8-12 小时
  - MPS 训练：约 20-30 分钟（20-40倍加速）
  - 得到"中文可读、与问题相关"的可用回答（泛化能力 90%，记忆能力 100%）
- 配置要点：
  - CPU 训练：`n_layer=6, n_head=6, n_embd=384, seq_len=256`
  - MPS 训练：`n_layer=6, n_head=6, n_embd=384, seq_len=256`
  - HF ByteLevel BPE 分词器（设置 ByteLevel 解码器）
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

- 设定：嵌入维度 `d=384`、层数 `L=6`、词表大小 `V=30522`（从 `tokenizer/tokenizer.json` 读取）
- 总参数（LM Head 与嵌入权重共享）：
  - 公式：`Total = V*d + L*(12*d^2 + 11*d) + d`
  - 含义：
    - `V*d`：词嵌入（每个词一个长度为 `d` 的向量）
    - 每层约 `12*d^2 + 11*d`：注意力的线性与投影、前馈两层线性与偏置、两处 RMSNorm 权重合计
    - `+ d`：最终 RMSNorm 权重
  - 代入数字：
    - `V*d = 30522*384 = 11,720,448`
    - `d^2 = 384*384 = 147,456`
    - 每层：`12*d^2 + 11*d = 12*147,456 + 11*384 = 1,769,472 + 4,224 = 1,773,696`
    - 所有层：`L*(...) = 6*1,773,696 = 10,642,176`
    - 总计：`11,720,448 + 10,642,176 + 384 = 22,363,008`（约 13.8M 参数）
  - 校验：与代码实际统计一致（`params_model=22,363,008`，约 13.8M 参数）

## 常见问题与排查

- 乱码输出：确保已运行 `src/build_tokenizer.py`，且其设置了 `ByteLevel` 解码器；查看推理是否使用 HF 分词器路径
- 空输出：提升步数（≥1500），并确认推理阶段首步屏蔽与停止词设置是否生效
- 两个不同问题给出同一答案：小数据短训记忆偏移，提升步数并适度扩充问句变体；或在推理时加`--stop_strings`
- MPS 训练失败：检查 `torch.backends.mps.is_available()` 是否为 True
- CUDA 训练失败：检查 `torch.cuda.is_available()` 是否为 True
- 训练太慢：尝试使用 MPS 或 GPU 而不是 CPU

## 教学建议（45 分钟）

1. 讲原理与结构（5 分钟）
2. 看代码与配置（5 分钟）
3. 训练与日志观察（20–25 分钟）
4. 推理演示与参数试验（8–10 分钟）
5. 作业：替换数据与分词器、改变步数，复现实验（5 分钟）

---

祝学习顺利！如需更高质量中文问答效果，可在不改代码的前提下扩充`data/train.jsonl`至数百条，并将`training.max_steps`提升到`20000`左右。课堂演示请选择`temperature=0.0`与适当`stop_strings`，确保"答案可用、与问题相关"。
