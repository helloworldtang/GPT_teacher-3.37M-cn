# TODO

> 更新于 2026-11-22。改代码红线见 CLAUDE.md（56 测试 + evaluate 6/6 + 示例命令 + 金样本回归）。

## 🧪 项目迭代

- [ ] distill 新方向：中等词表 BPE tokenizer 实验（5K-10K，降低 embedding 占比）
  - 依据：第 9 篇蒸馏三连败的"不可能三角"结论；从 student 架构和 tokenizer 下手
  - 安全网已就位（2026-08-19 三件套）：
    - `train/scripts/regression.py` 金样本回归（27 条，量化"换 tokenizer 改变多少行为"；
      重建基线用 `--update`，当前基线锚定 best.pt md5=32b2f46db7ec，2026-11-22 词表 751→770 后重建）
    - checkpoint 版本指纹（git hash + 数据 md5 + tokenizer md5 随权重落盘）
    - 加载时 tokenizer 一致性校验（词表漂移显式警告）
  - 前置加固已完成（2026-08-20）：prepare_data 已参数化（`--seq-len`/`--tokens-per-char` 联动 config.yml，换 tokenizer 时长度上限不再静默失效）
  - 实验产出可写系列第 11 篇

## 🔧 代码优化（2026-09-16 全库扫描，留作 v3.2.0 候选）

已核实未修项，按优先级排：

- [ ] `distill/infer.py:17` — TEACHER_MODEL 硬编码 `uer/gpt2-chinese-cluecorpussmall`，
  与 `distill/train.py:298` 训练默认 teacher（DeepSeek-R1-Distill-Qwen-1.5B）不匹配：
  默认参数训出的 checkpoint 推理时 `assert tok.vocab_size == vocab_size` 必炸。
  修法：save dict（distill/train.py:260-274）顺手存 teacher 名，infer 侧读出
- [ ] `core/infer.py:289-291` — 重复惩罚是标量循环（每 token 最多 32 次独立 kernel dispatch），
  可向量化为 unique 索引批量除；注意现实现对重复出现的 token 会除多次，向量化须保持该语义，
  改完跑金样本回归验证
- [ ] `core/model.py:33-62` — rope cos/sin 表每层每次前向重算，生成循环 launch-bound；
  预计算 register_buffer 估算省 10-30%。公式不变则数值逐位相同，原闭式公式务必留注释（教学）
- [ ] `distill/train.py:128-135` — 蒸馏 KL 对全部位置（含 -100 的 prompt/pad）做全词表
  softmax，违背 76 行"只对 completion 计算 loss"的意图；只在 `targets != -100` 位置算
  （改变 KD 训练语义，需重训，正好配合上面 tokenizer 实验）
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
