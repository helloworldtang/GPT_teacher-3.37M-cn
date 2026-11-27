# TODO

> 更新于 2026-11-27（销掉两条 2eb8d41 已修的旧账）。改代码红线见 CLAUDE.md（56 测试 + evaluate 6/6 + 示例命令 + 金样本回归）。

## 🧪 项目迭代

- [x] distill 蒸馏路径实验（2026-11-22 收官，全记录见 docs/distill-roadmap.md）
  - 中等词表(4096 BPE)实证 0.50 token/字反超 Qwen，不可能三角的 tokenizer 约束解除
  - 路径 A 数据级：域内 90% + OOD 零塌缩（第 9 篇基线全塌缩）——业界主流（DeepSeek-R1 同配方）
  - 路径 B 词表桥接 KD：负结果 ×2 轮，适用边界定稿（同词表预训练压缩才用 logit KD）
  - CoT 对照：4M 尺度纯负担（容量门槛），R1 配方需 student ≥1.5B
  - 系列第 11 篇素材齐备（含路径 B 负结果与容量门槛两个稀缺素材）
- [ ] 真实端侧小模型立项（终点二）：底座 Qwen2.5-0.5B/1.5B + 云端大 teacher 垂直 CoT 语料，
  实施方案见 [docs/ondevice-plan.md](docs/ondevice-plan.md)（迁移清单原件在 docs/distill-roadmap.md 决策树末节）
  - 阶段 0 任务分解（2026-11-27 展开）：
    - [ ] 用户拍板① 垂直领域（判据三条全过：端侧离线价值 / 答案短可自动判卷 / teacher 稳定产出）
    - [ ] 用户拍板② 端侧预算（目标设备 + 内存上限 + 首 token 延迟 → 反推底座尺寸与量化档位）
    - [ ] 评测集先行：域内 30 题 + OOD 15 题 + 判卷 prompt，定稿进 git 锁 md5 不许事后改（Claude 建、用户审）
    - [ ] 新项目落位：独立新仓拷四件套（生成管线 / 同题留出前缀 val 口径 / eval_ood / 指纹规范），不建跨仓 import
    - [ ] 冒烟脚本预铺：阶段 1 领域模板改造 + 30 主题试跑，等拍板即点火

## 🏭 半导体垂域蒸馏验证（独立新项目，2026-11-27 立项登记）

场景：半导体工厂工艺/设备/良率/处置日常问答，"问专家"→"问模型"（小时级→秒级）。
形态：9B 快车道 + 答不稳自动转 125B，部署成本 <125B 的 1/10，不占 125B 推理资源。

- [ ] 子命题① 语料质量：base-125B + 验证闸（证据链/自一致性）产语料过 rubric ≥6.0
  （此前 R1 3.73 / R2 3.69 已证伪不可上线）——**唯一阻塞点：等资料有无核查结论**
- [ ] 子命题② 知识注入：9B 垂域质量追平 125B 默认 prompt、接近严 prompt 98.8%
  （留出集 + bailian_val 对测）——环境已就绪（内网 9B 端点、域外基线 70%）
- [ ] 子命题③ 泛化代价：全参+量足下通用域外双基准掉多少——re-pilot 已预验证
  （LoRA 洗泛化 + 路由解兜底，五重证伪）
- [ ] 新项目落位预检：dataset-common 唯一事实源 + 评测泄漏审计（1288→1146 教训）；
  置信度路由阈值校准（低质答案识别的 precision/recall）
- [x] 落位路径定案 v2（2026-11-27，B→A 回切）：`github/ft/` **整体已搬** `~/workspace/semi/`
  （同卷 rename；四仓完好、相对 symlink 零断、hook 复测过；kev-lab 随行，其 `kev/.venv` 已重铸收尾
  ——`uv sync --frozen --inexact --extra serve` 只换 kev 自身链接，extras/手装包零卸载，
  /tmp 跨目录 import 复验全通过；runs/experiments 历史快照按"历史不动"原则保留旧路径）——新仓现址 `~/workspace/semi/qwen-distill-9b`
  （git @ 7597794 + pre-push 公网拦截 hook + eval/LOCK.json 尺子锁定）
  - **遗留**：① 搬前挂着的 node 会话需按新路径重开；② 三兄弟仓（dataset-common/qwen-flash/qwen3.5-9b）
    补装同款 hook + 四仓评估内网 gitlab 备份 remote（现全部无 remote，2.8G+ 数据单副本）
  - 后续任务清单跟到 `../../semi/qwen-distill-9b/TODO.md`（点火序列 + 等待期可做项）
- 与 GPT_teacher 仓的边界：只带管线口径四件套 + 三坑预检清单（拷出不用 import）；
  vocab_bridge / build_tokenizer / MPS 训练环不带；公司语料与评测集不进本仓（公网）

### 接续序列（2026-11-27 登记，下会话按此点火）

- [ ] **资料核查结论回来 → 定路 A/B**（路 A 证据链锚定 / 路 B 自一致性验证；
  分叉标准与资料需求说明见 re-pilot/docs/资料需求说明-半导体知识注入.md）
- [ ] **语料工厂点火**：改写引擎（参数化 30-100 问法/事实）+ 验证器（路 A 数值逐字比对 /
  路 B 三次严 prompt 一致性）两模块先写——**两路通用，不阻塞在资料结论上，可先行动工**
- [ ] **云机**：等 RAFT 训练空出 → SSH 验证（公钥 kev-finetune-cloud-2026-11 已授权）
  → 9B 训练环境准备（单卡 48G LoRA 先行，语料过万上全参）
- 环境就绪参照（均已实测）：内网 125B teacher 10.0.2.13:8001（113 tok/s，oQ6e/oQ8e）；
  9B 推理端点（域外基线 70%）；关思考链 `chat_template_kwargs.enable_thinking=false`

## 🔧 代码优化（2026-09-16 全库扫描，留作 v3.2.0 候选）

已核实未修项，按优先级排：

- [x] ~~`distill/infer.py:17` — TEACHER_MODEL 硬编码~~ — `2eb8d41`（2026-11-22）已修：
  teacher 名随 checkpoint save dict 落盘，infer 默认读 checkpoint 内记录（`--teacher` 可覆盖）
- [ ] `core/infer.py:289-291` — 重复惩罚是标量循环（每 token 最多 32 次独立 kernel dispatch），
  可向量化为 unique 索引批量除；注意现实现对重复出现的 token 会除多次，向量化须保持该语义，
  改完跑金样本回归验证
- [ ] `core/model.py:33-62` — rope cos/sin 表每层每次前向重算，生成循环 launch-bound；
  预计算 register_buffer 估算省 10-30%。公式不变则数值逐位相同，原闭式公式务必留注释（教学）
- [x] ~~`distill/train.py:128-135` — KL 对全部位置做全词表 softmax~~ — `2eb8d41`（2026-11-22）已修：
  `distill_loss` 先取 `targets != -100` 有效位再算 KL（全 -100 时返回 0，防 0×nan 传染）
- [ ] 重复代码收敛：`distill/infer.py:55-67` 是第 5 份 GPT 重建拷贝（收敛到
  `load_model_and_tokenizer`）；`train/train.py:310-351` tqdm/print 双分支 6 处约 20 行
  （一个 log() 闭包）；distill/train.py:142-154 的 evaluate 与 train/train.py:145-169 近乎逐行重复
- [ ] `pyproject.toml` — transformers 仅 distill/ 用、huggingface_hub 仅 push_to_hf.py 用，
  可移 optional extra（`[distill]`/`[hf]`），纯训练安装轻一大截；README 命令需同步
- [ ] `train/web_demo.py:473-690` — gr.Blocks 模块级构造，import 即建 UI；
  把 EXAMPLE_QUESTIONS / build_multi_turn_prompt 挪到无 gradio 依赖模块，
  回归套件与测试收集不再拖全 UI（引用方含 docker_entrypoint.py:6）

不建议动（教学取舍，扫描结论存档）：pad 固定 seq_len、KV cache 每 token cat、
generate 双 overload、train() 250 行长函数——均为教材本体，克制优先。
