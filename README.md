# GPT Teacher — 从 0 到 1 在 CPU 上训练可推理可演示的小参数中文 GPT，实现中文 GPT 训练平民化，实现训练中文 GPT 自由

本项目面向课堂教学，[目标是让初学者用一台普通 CPU 电脑，在 45 分钟内从零跑通一个小参数的中文 GPT](https://mp.weixin.qq.com/s/gUgY_TIRSoEyzZ2YRVGB2w)：看清核心流程、跑通训练、得到“可用的中文回答”，并支持简单的推理演示。

## 项目收获

- 了解 GPT 的核心原理：分词 → 批处理 → 前向 → 损失 → 反向 → 优化 → 保存 → 推理（完整链路）
- 掌握高效小模型技术：RMSNorm、RoPE、权重共享、短序列、小词表、量化
- 学会仅用 CPU 训练：控制模型/数据规模、梯度累积、学习率预热与退火、禁用无关耗时特性
- 学会可用答案保障：目标对齐（忽略前缀）、推理首步约束与后处理、停止词、提示词规范化

## 代码结构

- `src/model.py`：GPT 核心（Embedding/自注意力/前馈/RMSNorm/RoPE/权重共享）
- `src/data.py`：分词器加载、指令和 LM 数据处理、批处理与目标构造
- `src/train.py`：CPU 训练主循环、评估、保存与动态量化、耗时统计
- `src/infer.py`：推理 CLI（温度、top-k、top-p、重复惩罚、停止词、输出清理）
- `src/build_tokenizer.py`：构建 HF ByteLevel BPE 中文分词器（带解码器，避免乱码）
- `config.yaml`：统一管理模型、训练、数据、分词器与保存路径
- `data/*.jsonl`：教学数据（`{"prompt": "...", "completion": "..."}`）

## 使用技术

- 模型结构：Decoder-only Transformer（因果自注意力）
- 归一化：RMSNorm（简洁高效）
- 位置编码：RoPE（相对位置，计算高效）
- 前馈：SiLU（现代 LLM 常用）
- 权重共享：词嵌入与输出层共享，降参数且表示一致
- 训练：AdamW、学习率线性预热+余弦退火、梯度裁剪
- 数据与分词：外置`jsonl`+HF ByteLevel BPE（显式设置 ByteLevel 解码器，避免乱码）
- 推理：温度/top-k/top-p 采样、重复惩罚、停止词、输出清理、提示规范化
- 量化：导出动态量化权重以加速 CPU 推理

## 如何仅用 CPU 训练

- 小模型+短序列：`n_layer=4, n_head=4, n_embd=256, seq_len=128`
- 小词表：HF ByteLevel BPE（带解码器）
- 梯度累积：`batch_size=16, micro_batch=4`（有效批 64）
- 学习率策略：线性预热 10%+余弦退火
- 资源控制：`torch.set_num_threads(os.cpu_count())`，DataLoader 禁用多进程（macOS 下更稳）
- 训练结束导出量化权重，演示更流畅

## GPT 训练是如何实现的

- 因果自注意力：下三角 mask 确保只看历史（`src/model.py:98`）
- Teacher Forcing：输入拼接“用户/prompt + 助手 + 答案”，仅对答案段计算损失（`src/data.py:24–31`）
- 目标对齐：忽略区间为`len(prefix)-1`，确保第一个答案 token 参与训练（`src/data.py:28–31`）
- 损失函数：`CrossEntropyLoss(ignore_index=-100)`（`src/train.py:50`）

## 如何保证“回答可用”

- 分词器解码器：HF BPE 设置 ByteLevel 解码器，彻底消除中文乱码（`src/build_tokenizer.py:6–8`）
- 目标对齐修复：忽略`len(prefix)-1`，避免答案错位（`src/data.py:28–31`）
- 推理约束：
  - 首步屏蔽`PAD/BOS/UNK`，前若干步屏蔽`EOS`，避免空答（`src/infer.py:41–50, 68–73`）
  - 开头清理：剔除首字符标点与空白（`src/infer.py:26–30, 82–83`）
  - 提示词规范化：去中文空格（`src/infer.py:35–37`）
  - 停止词/重复惩罚：控制尾部冗余与重复（`src/infer.py:68–73`）
- 训练耗时与评估：日志实时打印`eval loss`与`elapsed`，课时内可观测质量（`src/train.py:71–82`）

## 从 0 到 1：一步步跑通

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 构建中文分词器（HF ByteLevel BPE）

```bash
python -m src.build_tokenizer
```

### 3. 配置核对

- `config.yaml` 中：
  - `data.train_path: data/train.jsonl`
  - `data.val_path: data/val.jsonl`
  - `tokenizer.type: hf_tokenizers`
  - `tokenizer.path: tokenizer/tokenizer.json`
  - `training.max_steps: 1500–2000`（课堂机器允许的话）

### 4. 训练（CPU/GPU）

```bash
python -m src.train
```

- 观察日志：
  - 每`eval_interval`步打印`eval loss`与累计`elapsed Xs`
  - 结束后保存：`checkpoints/last.pt`、`checkpoints/quantized.pt`、`checkpoints/train_time.txt`
- 设备入参：
  - `--device auto|cpu|cuda|gpu`
  - 不传或 `auto`：优先使用 `cuda`，否则回退 `cpu`
  - `cpu`：强制使用 CPU
  - `cuda`/`gpu`：若本机不支持 CUDA 则报错
  - 示例：`python -m src.train --device gpu`
  

### 5. 推理验证（两条固定问题）

```bash
# Question1
python -m src.infer --prompt "什么是注意力机制？" --ckpt checkpoints/last.pt --temperature 0.0 --show_label

# Question2
python -m src.infer --prompt "解释蒸馏水与纯水区别？" --ckpt checkpoints/last.pt --temperature 0.0 --show_label
```

- 期望结果（示例）：
  - Q1：`注意力机制通过分配权重让模型关注关键位置，从而更好地理解序列中的关系。`
  - Q2：`蒸馏水是通过蒸馏获得的水，去除了大部分杂质；纯水是指杂质含量极低的水，制备方式可以是蒸馏、反渗透等。`
- 若出现“串联两段”或冗长，可加入停止词：

```bash
python -m src.infer --prompt "什么是注意力机制？" --stop_strings "。" "；" "\n" --temperature 0.0 --show_label
```

### 6.（可选）量化权重推理

```bash
python -m src.infer --prompt "什么是注意力机制？" --ckpt checkpoints/quantized.pt --temperature 0.0 --show_label
```

## 核心代码参考（行号）

- 因果 mask 与前向（`src/model.py:95–103`）
- RoPE 相对位置（`src/model.py:18–31`）
- 自注意力前向（`src/model.py:41–58`）
- 指令目标构造与忽略前缀（`src/data.py:24–31`）
- 训练主循环（`src/train.py:56–80`）
- 推理生成与采样管线（`src/infer.py:32–83`）
- ByteLevel 解码器设置（`src/build_tokenizer.py:6–8`）

> 说明：`src/model.py`核心约百行即可完整呈现 GPT 最小闭环；其余文件各自职责清晰，便于教学与修改。

## 核心代码解析（从架构 → 流程 → 方法级）

- 整体架构

  - `src/model.py`：定义 GPT 的模块化组件（RMSNorm、RoPE、自注意力、前馈、残差与权重共享）
  - `src/data.py`：将`{"prompt","completion"}`样本转换为模型输入与训练目标（teacher forcing）
  - `src/train.py`：CPU 训练主循环、评估与保存（含动态量化与耗时统计）
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
  - 残差块：`Block.forward`（`src/model.py:81–84`）注意力与前馈的残差堆叠，RMSNorm 在前
  - 模型前向：`GPT.forward`（`src/model.py:95–103`）嵌入 → 堆叠 Block→ 归一化 →LM Head（共享权重）

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
- `src/build_tokenizer.py`：27 行
- 合计（核心代码）：484 行
- 配置与辅助：`config.yaml`：23 行（不计入核心代码合计）

## 实测与分析（答案可用、与问题相关）

- 环境与数据：在 mac pro 2.6GHz 6c i7（16GB DDR4）上，使用 510 条训练集与 90 条验证集，CPU 训练 2000 步。
- 耗时与结果：总耗时约 19.78 分钟（≈1186.8s），得到“中文可读、与问题相关”的可用回答（固定问题 Q1/Q2 实测通过）。
- 配置要点：`n_layer=4, n_head=4, n_embd=256, seq_len=128`；HF ByteLevel BPE 分词器（设置 ByteLevel 解码器）；权重共享、RMSNorm、RoPE；AdamW+预热+余弦；梯度累积与裁剪。
- 效率核心原因（技术角度）：
  - 小模型与短序列降低注意力计算开销；权重共享减少参数与内存占用；RMSNorm 与 RoPE 计算简洁且稳定。
  - 仅 CPU 优化：限制 DataLoader 开销、设线程数；预热+退火提升早期收敛效率；梯度累积在小内存下获得有效大批次。
  - 可用性保障：目标对齐只对答案段计算损失；推理阶段首步/初期屏蔽无效与 `EOS`、提示规范化、输出清理与停止词，避免空答与乱码。

## 参数规模与计算

- 设定：嵌入维度 `d=256`、层数 `L=4`、词表大小 `V=824`（从 `tokenizer/tokenizer.json` 读取）。
- 总参数（LM Head 与嵌入权重共享）：
  - 公式：`Total = V*d + L*(12*d^2 + 11*d) + d`
  - 含义：
    - `V*d`：词嵌入（每个词一个长度为 `d` 的向量）
    - 每层约 `12*d^2 + 11*d`：注意力的线性与投影、前馈两层线性与偏置、两处 RMSNorm 权重合计
    - `+ d`：最终 RMSNorm 权重
  - 代入数字：
    - `V*d = 824*256 = 210,944`
    - `d^2 = 256*256 = 65,536`
    - 每层：`12*d^2 + 11*d = 12*65,536 + 11*256 = 786,432 + 2,816 = 789,248`
    - 所有层：`L*(...) = 4*789,248 = 3,156,992`
    - 总计：`210,944 + 3,156,992 + 256 = 3,368,192`
  - 校验：与代码实际统计一致（`params_model=3,368,192`）。

## 常见问题与排查

- 乱码输出：确保已运行 `src/build_tokenizer.py`，且其设置了 `ByteLevel` 解码器；查看推理是否使用 HF 分词器路径
- 空输出：提升步数（≥1500），并确认推理阶段首步屏蔽与停止词设置是否生效
- 两个不同问题给出同一答案：小数据短训记忆偏移，提升步数并适度扩充问句变体；或在推理时加`--stop_strings`

## 教学建议（45 分钟）

1. 讲原理与结构（5 分钟）
2. 看代码与配置（5 分钟）
3. 训练与日志观察（20–25 分钟）
4. 推理演示与参数试验（8–10 分钟）
5. 作业：替换数据与分词器、改变步数，复现实验（5 分钟）

---

祝学习顺利！如需更高质量中文问答效果，可在不改代码的前提下扩充`data/train.jsonl`至数百条，并将`training.max_steps`提升到`2000`左右。课堂演示请选择`temperature=0.0`与适当`stop_strings`，确保“答案可用、与问题相关”。

## 快速开始（两分钟上手）

- 安装依赖：`python -m pip install -r requirements.txt`
- 构建分词器：`python -m src.build_tokenizer`
- 训练（CPU）：`python -m src.train`
- 推理验证：
  - Q1：`python -m src.infer --prompt "什么是注意力机制？" --temperature 0.0 --show_label`
  - Q2：`python -m src.infer --prompt "解释蒸馏水与纯水区别？" --temperature 0.0 --show_label`

## Git LFS 使用说明（下载与使用权重）

- 目的：权重文件（如 `checkpoints/last.pt`, `checkpoints/quantized.pt`）由 Git LFS 管理，保证代码仓库轻量且权重传输高效。
- 安装 LFS：
  - macOS：`brew install git-lfs`
  - Ubuntu/Debian：`sudo apt install git-lfs`
- 初始化（首次在本机执行一次）：`git lfs install`
- 克隆并拉取权重：
  - `git clone <仓库地址>`
  - 进入仓库后执行：`git lfs pull`
- 验证：`git lfs ls-files` 显示被 LFS 管理的权重文件列表
- 推送说明：维护者在迁移到 LFS 后需用 `--force` 强推一次，其后正常 `git push`/`git pull` 即可；协作者如历史被重写，建议重新克隆或 `git fetch && git reset --hard origin/main`。

## 上传到 Hugging Face 分享模型

- 准备：
  - 安装 CLI：`pip install huggingface_hub`
  - 登录：`huggingface-cli login`
- 创建模型仓库（CLI）：
  - `huggingface-cli repo create GPT_teacher-3.37M-cn -y`
- 组织文件（建议）：
  - 复制到新目录：`checkpoints/last.pt`、`checkpoints/quantized.pt`、`tokenizer/tokenizer.json`、`config.yaml`、简要说明 `README.md`
- 推送到 Hub（用 Git LFS）：
  - `git init && git lfs install`
  - `git remote add origin https://huggingface.co/GPTcn/GPT_teacher-3.37M-cn`
  - `git add -A && git commit -m "Upload CPU GPT teacher checkpoints"`
  - `git push -u origin main`
- Python API（可选）：
  - 参见 `huggingface_hub` 的 `upload_file`/`upload_folder` 接口，适合脚本化批量上传。

## Hugging Face 使用指引（下载并在本项目推理）

- 仓库地址：`https://huggingface.co/GPTcn/GPT_teacher-3.37M-cn`
- 方式一：Git LFS 克隆
  - `git lfs install`
  - `git clone https://huggingface.co/GPTcn/GPT_teacher-3.37M-cn`
  - 将下载的 `last.pt`（或`quantized.pt`）放到本项目 `checkpoints/` 目录
  - 推理：
    - Q1：`python -m src.infer --prompt "什么是注意力机制？" --ckpt checkpoints/last.pt --temperature 0.0 --show_label`
    - Q2：`python -m src.infer --prompt "解释蒸馏水与纯水区别？" --ckpt checkpoints/last.pt --temperature 0.0 --show_label`
- 方式二：Python API 下载
  - 安装：`python -m pip install huggingface_hub`
  - 示例：
    - `from huggingface_hub import hf_hub_download`
    - `pt = hf_hub_download(repo_id='GPTcn/GPT_teacher-3.37M-cn', filename='last.pt')`
    - `tok = hf_hub_download(repo_id='GPTcn/GPT_teacher-3.37M-cn', filename='tokenizer.json')`
    - `cfg = hf_hub_download(repo_id='GPTcn/GPT_teacher-3.37M-cn', filename='config.yaml')`
  - 使用方法：将 `pt` 拷贝到 `checkpoints/last.pt`，`tok` 到 `tokenizer/tokenizer.json`，`cfg` 到项目根；随后按上面的推理命令运行。

