# 大模型蒸馏小模型：路径路线图

> 2026-11-22 立项。首要目标：调研业界成功经验 → 在本仓验证 → 把「大模型蒸馏训练小模型」的可复制路径确定下来。
> 轻量级、敏捷、高效：小规模先跑通，再放大。本仓第 9 篇"三连败"（不可能三角）是背景而非本轮目标。

## 业界三条路径（2026 共识）

| 路径 | 机制 | 成本 | 适用场景 | 本仓验证点 |
|---|---|---|---|---|
| **A 数据级** | teacher 生成语料 → student 做 SFT | 最低（teacher 只推理一次） | 词表不匹配、预算紧、冷启动 | 阶段 2 |
| **B logits 级 off-policy** | teacher/student 同步前向，KL 对齐输出分布 | 中（训练每步都要 teacher 前向） | 词表可对齐或可投影，追求分布级监督 | 阶段 3 |
| **C on-policy（GKD）** | student 采样轨迹，teacher 对 student 的 token 逐位反向 KL | 最高（student 生成 + teacher 打分循环） | 效果上限最高，小模型能力迁移的主流方向 | v2 展望，不在本轮 |

效果序 C > B > A，成本序同向（HuggingFace H4 在 SmolLM/Llama/Qwen 上的对比，Qwen3 大规模验证）。
本轮走通 A、B，C 留接口。

### 本仓特殊约束与对策

第 9 篇不可能三角（容量 / 词表大小 / tokenizer 效率）的解法是**中等词表**：

- student 自建 2K-5K BPE（语料由 teacher 生成）——汉字 1-2 token，治 byte tokenizer 信息密度低的根因（方向二败因）
- embedding 占比可控（3K × 256 ≈ 0.77M，4 层 transformer 本体占大头）——治 151K 词表吃光参数的根因（方向三败因）
- 代价：与 teacher 词表不再相同 → 路径 B 需要**词表聚合投影**（teacher 分布按 token 字符串重切分聚合到 student 词表，Universal Logit Loss 思路）

### 业界参考

- [Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)（HF H4）：on-policy 蒸馏优于 RL 的实证，SmolLM/Llama/Qwen 跨家族压缩
- [Solving Without Stopping: On-Policy Distillation at Small Scale](https://arxiv.org/abs/2609.37326)（2026）：小规模下 on-policy 迁移了什么
- [SOD: Step-wise On-policy Distillation](https://github.com/chrisliu298/awesome-on-policy-distillation)：按 student-teacher 分歧度逐步重加权，端侧小模型智能体
- [Universal Logit Distillation Loss](https://arxiv.org/abs/2501.09723)：词表不匹配的通用 logit 损失——路径 B 投影的理论依据
- [logit distillation might not be always helpful](https://seungjuhan.me/logit-distillation-pretraining)：logit 蒸馏提计算效率、不提数据效率的实证——预期管理
- [On-Device LLMs: State of the Union 2026](https://v-chandra.github.io/on-device-llms) / [端侧 LLM 压缩综述 (Nature 2026)](https://arxiv.org/abs/2403.06664)：端侧部署全景，KD+量化是标配组合
- 语音端侧参照系：Whisper/Conformer 级 teacher → 端侧小 student + KD + 量化（管线结构与文本同构）；[Kakugo](https://arxiv.org/html/2601.14051v1)（2026）低资源语言蒸馏进端侧模型

## 落地计划与本仓映射

```
阶段 0  修 distill 已知 bug（KD masked KL / teacher 落盘 / infer 硬编码）   ✅ 2eb8d41
阶段 1  本路线图                                                            ✅ 本文件
阶段 2  路径 A 最小闭环：teacher 生成 ~500 主题语料 → 中等词表 → SFT → OOD 评测
阶段 3  路径 B：词表聚合投影（vocab_bridge，单测先行）→ KD → A/B/纯SFT 三组对比
阶段 4  实证数据回填本表 → 决策树定稿 + distill/README 更新（第 11 篇素材）
```

### 决策树（阶段 4 定稿，先立骨架）

- teacher/student 词表不同且不想做投影 → **A**
- 有 teacher 前向算力、要分布级监督 → **B**
- 追求能力上限、有 student 采样循环的工程预算 → **C**
- 端侧部署：以上任一路径 + 量化（本项目已有 int8 导出链路）

## 实验记录（阶段 2/3 回填）

### 路径 A：数据级蒸馏（2026-11-22 实测，MPS）

| 环节 | 参数/结果 |
|---|---|
| Teacher | Qwen2.5-1.5B-Instruct（bf16, MPS，hf-mirror 拉取） |
| 语料 | 两段式生成：10 领域发散 421 主题 → 逐题一句话作答 → 406 唯一问答（答案 4-59 字，均值 26） |
| 划分 | 训练 2842 行（7 前缀变体）/ 验证 406 行（同题留出前缀「简单说说」）/ OOD 15 问（held-out 主题） |
| Student tokenizer | 4096 BPE（语料训出满 4096），**0.50 token/字反超 Qwen 的 0.62**（领域语料红利） |
| Student | 4.0M 参数（4L/4H/256d GQA，与主线同构），SFT 6.4min，val CE 0.31（step 4950） |
| 域内评测 | 30 题精确命中 27 / 部分 2 / 未中 1（90%） |
| OOD 评测 | **15/15 唯一答案，塌缩 7%**（第 9 篇基线 1/15 唯一、100% 塌缩）；teacher 判卷 1/15 |

结论：数据级蒸馏的「不可能三角」两根因都被中等词表治了——student 记住了
全部教师语料且不塌缩；真·泛化（答对新主题）1/15，是 4M 参数的诚实上限，
交给路径 B 验证 KL 监督能否改善。

### 路径 A 踩坑（选型口径）

1. **held-out 主题不能当验证集**：3M student 在其上 val loss 卡 7.0（均匀基线
   8.3），早停在 550 步误杀记忆阶段，best.pt 域内也全是碎句。对「拟合语料」
   的目标，选型信号应是「已知问题的表面变体答得好不好」→ val 改同题留出
   前缀后 val 7.0→0.31、域内 90%。
2. **1.5B 阅卷员会放水**：把碎句「将使用，通过在」判对；判卷 prompt 须加
   「既语义通顺又正确」双条件。
3. **teacher 主题输出形态不稳**：纯行/编号/JSON/Python repr/围栏混出，
   解析统一为「提取以？结尾的干净片段」策略（6 形态单测锁定）。

### 路径 B：词表桥接蒸馏（进行中）

- 精确匹配 3079/4096（75%）；前缀聚合后**词表覆盖 83%**
- nearest 位置对齐下 completion 内对齐率 ~100%（诊断口径须除 CE 有效位，非填充全长）
- MPS 后端 index_add_/scatter_add_ 在真实 teacher 前向输出上必现
  "Placeholder storage" 错（合成最小复现不出）→ gather 后 CPU 执行兜底
- （训练结果与 A/B 对比表待回填）
