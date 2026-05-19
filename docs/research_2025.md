# 小模型训练最新经验总结（2025-2026）

> 调研时间：2026-05，为本项目 GPT Teacher 3.37M 的优化提供参考。

## 架构层面

| 技术 | 来源 | 说明 |
|------|------|------|
| **GQA (Grouped Query Attention)** | Llama3, Gemma2 | KV head 数量少于 Q head，减少推理 KV Cache 体积，几乎不影响效果 |
| **SwiGLU** | Llama 系列 | 前馈层 `gate=silu(W_gate*x); out=W_down(gate * W_up(x))`，效果优于单独 SiLU |
| **Pre-RMSNorm + RoPE** | 业界标准 | 已是标配，无需改动 |
| **Flash Attention** | PyTorch 2.0+ | 内置 `scaled_dot_product_attention`，训练速度提升约30% |

**本项目已采用**：GQA(n_kv_head=2)、SwiGLU、RMSNorm、RoPE、权重共享、Flash Attention。

## 训练配方

| 要点 | 说明 |
|------|------|
| **数据质量 > 数据量** | Microsoft Phi 核心发现：少量"教科书级"数据胜过大量低质量数据 |
| **学习率** | 小模型+小数据可用较高 lr（1e-3 ~ 3e-3），配合 warmup + cosine annealing |
| **Epoch 控制** | 200-500条数据，5-10 epoch 为宜，避免 20+ epoch 导致过拟合 |
| **Warmup** | 总步数的 5-10% |
| **Early stopping** | patience 设为 10-15 |
| **重复惩罚** | 小模型推理时必须有 repetition_penalty ≥ 1.5，否则容易陷入重复循环 |

## 数据策略

| 要点 | 说明 |
|------|------|
| **变体增强要多样化** | 不要简单复制 20 遍（= 严重过拟合），用前缀变化、同义改写 |
| **适度复制** | 控制在 3 倍以内 |
| **验证集独立** | 验证集数据不参与训练集增强 |
| **答案风格差异化** | 同一主题的多个问题，答案措辞尽量不同，帮助模型区分 |
| **词表覆盖** | 分词器要用实际训练数据训练，词表大小 1500-4000 为宜 |

## 教育项目标杆

| 项目 | Stars | 关键做法 |
|------|-------|----------|
| **nanochat** (Karpathy) | 53.5k | 单刻度盘 `--depth`，从文本到 ChatGPT 风格 UI 的完整管道 |
| **nanoGPT** (Karpathy) | 58k | Shakespeare 数据 3 分钟出结果 |
| **LLMs-from-scratch** (Raschka) | 95k | Jupyter notebook 每步有输出 |
| **llm.c** (Karpathy) | 30k | 纯 C 实现，单元测试验证数值正确性 |
| **picoGPT** | 3.5k | 60 行 NumPy 代码实现 GPT 推理 |

## 关键教训

1. **即时 gratification 不可妥协** — 用户必须几分钟内看到结果
2. **单命令管道是杀手级功能** — 不要让用户组装零件
3. **客观指标 > 主观评估** — 验收测试 pass/fail
4. **一个刻度盘 > 二十个参数** — 初学者不需要选超参
5. **对话式 UI 是终极闭环** — 训练完直接对话比看 loss 曲线更有说服力
6. **重复惩罚必须对 greedy 生效** — temperature=0 时也需要 repetition_penalty

## 本项目的优化路径

1. 分词器：用训练数据重建，vocab 750 → 1509+，编码质量大幅提升
2. 架构：加入 GQA + SwiGLU，同参数量下效果更好
3. 数据：原始问答从 16 条增加到 30 条，增强方式从"复制20遍"改为"前缀变体+适度复制"
4. 推理：修复 repetition_penalty 在 greedy 模式下不生效的 bug
5. 配置：max_steps 600 → 2000，lr 6e-4 → 1e-3，warmup 60 → 100
