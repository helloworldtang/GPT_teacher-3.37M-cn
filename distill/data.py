"""知识蒸馏数据工具。

提供三类功能：
1. teacher 语料生成 — 两段式：teacher 发散主题 → 逐题作答（路径 A 数据级蒸馏的语料源）
2. teacher 数据增强 — 用指令微调模型（Qwen/DeepSeek）生成训练答案
3. 主题扩展 — 生成更多主题的 Q&A 数据，测试模型容量

用法：
    uv run python -m distill.data --method generate --teacher Qwen/Qwen2.5-1.5B-Instruct --topics 500
    uv run python -m distill.data --method augment --teacher Qwen/Qwen2.5-1.5B-Instruct
    uv run python -m distill.data --method expand --topics 30
"""

import argparse
import json
import random

# 原始 12 个主题
ORIGINAL_TOPICS = [
    ("什么是因果掩码？", "下三角矩阵，把未来位置权重设为负无穷，确保模型只看到历史信息。"),
    ("RMSNorm 和 LayerNorm 有什么区别？", "RMSNorm 只做均方根缩放不减均值，比 LayerNorm 更简洁高效。"),
    ("15 乘以 6 等于多少？", "15 乘以 2 是 30，30 乘以 3 是 90，所以等于 90。"),
    ("RoPE 是什么？", "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"),
    ("125 除以 5 是多少？", "100 除以 5 是 20，25 除以 5 是 5，20 加 5 等于 25。"),
    ("什么是 Python？", "一种简洁易读的高级编程语言，以丰富的第三方库生态而闻名。"),
    ("太阳系有哪些行星？", "八大行星：水星、金星、地球、火星、木星、土星、天王星、海王星。"),
    ("权重共享有什么好处？", "嵌入层和输出层共用参数，减少模型体积并保持输入输出语义一致。"),
    ("什么是机器学习？", "让计算机通过数据自动学习规律，而不需要显式编写每一条规则的技术。"),
    ("什么是注意力机制？", "注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。"),
    ("蒸馏水和纯水有什么区别？", "蒸馏水通过蒸馏冷凝制得，纯水杂质极低。"),
    ("你是谁？", "我是一个基于 Transformer 的小型 GPT 教学演示模型。"),
]

# 扩展主题
EXTRA_ML_TOPICS = [
    ("什么是深度学习？", "用多层神经网络从数据中自动提取特征并进行学习的机器学习方法。"),
    ("什么是 Transformer？", "一种基于自注意力机制的神经网络架构，能并行处理序列数据。"),
    ("什么是 token？", "文本的最小处理单元，一个 token 可以是一个字、一个词或一个子词。"),
    ("什么是 GPT？", "基于 Transformer 解码器的生成式预训练模型，通过预测下一个 token 来生成文本。"),
    ("什么是神经网络？", "模拟生物神经元的计算模型，通过层与层之间的连接和权重来处理信息。"),
    ("什么是过拟合？", "模型在训练数据上表现好但在新数据上表现差，因为学到了噪声而非规律。"),
    ("什么是梯度下降？", "一种优化算法，沿损失函数梯度反方向更新参数，逐步逼近最优解。"),
    ("什么是反向传播？", "利用链式法则从输出层向输入层逐层计算梯度，用于训练神经网络。"),
    ("什么是损失函数？", "衡量模型预测值和真实值之间差距的函数，训练目标就是最小化它。"),
    ("什么是自然语言处理？", "让计算机理解、生成和处理人类自然语言的技术领域。"),
]

EXTRA_GENERAL_TOPICS = [
    ("CPU 和 GPU 有什么区别？", "CPU 擅长复杂逻辑运算，GPU 擅长大规模并行计算。"),
    ("什么是算法？", "解决特定问题的一组有限步骤，是计算机程序的核心逻辑。"),
    ("光速是多少？", "真空中光速约为每秒 30 万公里，是宇宙中信息传播的速度上限。"),
    ("什么是 DNA？", "脱氧核糖核酸，携带生物遗传信息的双螺旋结构分子。"),
    ("地球到月球有多远？", "约 38 万公里，光从地球到月球只需约 1.3 秒。"),
    ("什么是操作系统？", "管理计算机硬件和软件资源的系统软件，如 Windows、Linux、macOS。"),
    ("什么是互联网？", "由全球计算机通过标准协议互连而成的网络，实现信息共享和通信。"),
    ("圆周率是多少？", "圆的周长与直径之比，约等于 3.14159，是一个无理数。"),
]

PREFIXES = ["", "你知道", "请问", "帮我解释下", "你能告诉我", "能不能说说", "讲一下", "简单说说"]
REPEATS = 4

# teacher 发散主题的领域分布（教学演示口径：知识问答，避开时效/争议话题）
CATEGORIES = [
    "机器学习与人工智能",
    "计算机与编程",
    "数学与逻辑",
    "物理与天文",
    "化学与生物",
    "地理与自然",
    "历史与文化",
    "生活常识",
    "语言与文学",
    "健康与安全",
]


def generate_topic_dataset(topics: list[tuple[str, str]], path: str) -> list[dict[str, str]]:
    """从主题列表生成 JSONL 数据（前缀变体 × 重复次数，seed=42 打乱）。

    Args:
        topics: (问题, 答案) 列表。
        path: 输出 jsonl 路径。

    Returns:
        生成的问答对列表。
    """
    data = []
    for question, answer in topics:
        for pf in PREFIXES:
            for _ in range(REPEATS):
                data.append({"prompt": pf + question, "completion": answer})
    random.seed(42)
    random.shuffle(data)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    unique = len({item["completion"] for item in data})
    print(f"{path}: {len(data)} 条 ({unique} 唯一答案)")
    return data


def expand(topics_count: int) -> None:
    """按主题数生成扩展数据集（>22 时追加通用主题）。

    Args:
        topics_count: 主题数量。
    """
    if topics_count <= 12:
        topics = ORIGINAL_TOPICS[:topics_count]
    elif topics_count <= 22:
        topics = ORIGINAL_TOPICS + EXTRA_ML_TOPICS[: topics_count - 12]
    else:
        topics = ORIGINAL_TOPICS + EXTRA_ML_TOPICS + EXTRA_GENERAL_TOPICS[: topics_count - 22]

    path = f"distill/data/train_{topics_count}.jsonl"
    generate_topic_dataset(topics, path)
    print(f"生成 {topics_count} 主题数据 → {path}")


def augment(teacher_name: str) -> None:
    """用 teacher 模型逐题生成答案，构造增强训练数据。

    Args:
        teacher_name: HuggingFace teacher 模型名。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(teacher_name)
    model = AutoModelForCausalLM.from_pretrained(teacher_name)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"Teacher: {teacher_name} ({device})")

    # 提取唯一问题
    seen = set()
    questions = []
    with open("train/data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["prompt"]
            for pf in PREFIXES:
                if q.startswith(pf):
                    q = q[len(pf) :]
                    break
            q = q.strip()
            if q not in seen:
                seen.add(q)
                questions.append((q, obj["completion"]))

    print(f"唯一问题: {len(questions)} 个")

    results = []
    for q, orig in questions:
        messages = [
            {"role": "system", "content": "你是一个简洁的知识助手。回答控制在30字以内。"},
            {"role": "user", "content": q},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=80, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tok.eos_token_id
            )
        gen_ids = output[0][inputs["input_ids"].shape[1] :]
        answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
        for end in ["\n", "。"]:
            idx = answer.find(end, 1)
            if idx > 0:
                answer = answer[: idx + 1]
                break
        results.append({"prompt": q, "original": orig, "teacher": answer})
        print(f"Q: {q}")
        print(f"  原始: {orig}")
        print(f"  Teacher: {answer}")
        print()

    # 保存
    data = []
    for item in results:
        for pf in PREFIXES:
            for _ in range(REPEATS):
                data.append({"prompt": pf + item["prompt"], "completion": item["teacher"]})
    random.seed(42)
    random.shuffle(data)
    out_path = "distill/data/train_teacher_augmented.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n生成 {len(data)} 条 → {out_path}")


def _parse_topic_list(raw: str) -> list[str]:
    """从 teacher 输出解析主题问题列表。

    1.5B teacher 输出形态不稳定：纯行、编号、JSON 数组、Python repr
    （{'question': '…'}）、markdown 围栏都会出现。统一策略：剥围栏后，
    用正则提取所有"不含引号/括号/换行的、以？结尾的片段"，
    长度过滤 + 去重——对各种壳一视同仁。
    """
    import re

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)

    topics: list[str] = []
    for m in re.finditer(r"[^\n\"'“”{}()\[\]]{2,}？", text):
        q = m.group(0).strip()
        q = q.lstrip(" -0123456789.、？")
        if len(q) < 5 or q in topics:
            continue
        topics.append(q)
    return topics


def generate(
    teacher_name: str,
    topics_count: int,
    holdout: int,
    val_topics: int,
    repeats: int,
) -> None:
    """两段式语料生成：teacher 发散主题 → 逐题作答 → 前缀增强落盘。

    产出（均在 distill/data/ 下）：
    - generated_qa.jsonl   全量 (prompt, completion) 唯一对，含主题来源，供人工抽检
    - train_generated.jsonl 训练集（前缀变体 × repeats，seed=42 打乱）
    - val_generated.jsonl   验证集（从训练主题中划出，非行级切分，防同题泄漏）
    - ood_questions.json    训练从未见过的主题问题，供 eval_ood.py 用

    Args:
        teacher_name: teacher 模型名（HuggingFace id）。
        topics_count: 生成的主题总数（不含 holdout/val 划走的部分）。
        holdout: OOD 评测保留的主题数（不进训练）。
        val_topics: 验证集主题数（不进训练，但同分布）。
        repeats: 每条 (前缀, 问答) 的重复次数。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(teacher_name)
    # transformers 4.48 用 torch_dtype；bf16 在 M 系 MPS 上比 fp32 快约一倍
    model = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.bfloat16)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"Teacher: {teacher_name} ({device})")

    def chat(question: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": "你是一个简洁的知识助手。"},
            {"role": "user", "content": question},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = output[0][inputs["input_ids"].shape[1] :]
        decoded: str = tok.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()

    # ── 第一段：按领域发散主题 ──
    per_category = max(1, topics_count // len(CATEGORIES))
    seen = {q for q, _ in ORIGINAL_TOPICS}
    topics: list[str] = []
    for cat in CATEGORIES:
        raw = chat(
            f"请列出{per_category}个适合教学演示的中文知识问答主题，领域：{cat}。"
            "要求：每行一个问题，以？结尾，覆盖该领域不同的知识点，不要重复；"
            "不要编号、不要 JSON、不要任何解释。",
            max_new_tokens=1600,
            temperature=0.9,
        )
        got = _parse_topic_list(raw)
        for q in got:
            if q not in seen:
                seen.add(q)
                topics.append(q)
        print(f"  {cat}: +{len(got)} (累计 {len(topics)})")
    random.seed(42)
    random.shuffle(topics)
    if len(topics) < topics_count:
        print(f"⚠ 主题数 {len(topics)} 未达目标 {topics_count}（teacher 输出上限），继续")

    ood_set = topics[:holdout]
    val_set = topics[holdout : holdout + val_topics]
    train_set = topics[holdout + val_topics :]
    print(f"主题划分: 训练 {len(train_set)} / 验证 {len(val_set)} / OOD {len(ood_set)}")

    # ── 第二段：逐题作答（一句话 ≤40 字，低温保准确）──
    qa_pairs: list[dict[str, str]] = []
    for i, q in enumerate(train_set + val_set):
        answer = ""
        for _ in range(2):  # 空/超长答案重试一次
            answer = chat(
                f"用一句不超过40个字的完整句子准确回答：{q}\n要求：直接给答案，不要列表、编号、引号或解释。",
                max_new_tokens=80,
                temperature=0.3,
            )
            # 只取第一行；一句话要求下句号截断无损（列表型输出已被 prompt 禁掉）
            answer = answer.split("\n")[0].strip().strip('"“”')
            if 4 <= len(answer) <= 60:
                break
        if not (4 <= len(answer) <= 60):
            print(f"  ✗ 跳过（答案长度不合规）: {q} -> {answer[:30]}")
            continue
        qa_pairs.append({"prompt": q, "completion": answer})
        if (i + 1) % 25 == 0:
            print(f"  作答进度: {i + 1}/{len(train_set) + len(val_set)}")

    with open("distill/data/generated_qa.jsonl", "w", encoding="utf-8") as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 验证集 = 同题留出前缀（最后一个前缀变体只进 val）：对"拟合语料"的
    # 数据级蒸馏，选型信号应是"已知问题的表面变体答得好不好"，而非
    # held-out 主题的 loss（后者 3M student 天然下不去，会误杀记忆阶段）。
    # 泛化能力另由 OOD 问题集单独评测。
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for item in qa_pairs:
        for pf in PREFIXES[:-1]:
            for _ in range(repeats):
                train_rows.append({"prompt": pf + item["prompt"], "completion": item["completion"]})
        val_rows.append({"prompt": PREFIXES[-1] + item["prompt"], "completion": item["completion"]})
    random.seed(42)
    random.shuffle(train_rows)

    with open("distill/data/train_generated.jsonl", "w", encoding="utf-8") as f:
        for item in train_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("distill/data/val_generated.jsonl", "w", encoding="utf-8") as f:
        for item in val_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("distill/data/ood_questions.json", "w", encoding="utf-8") as f:
        json.dump({"ood": ood_set, "val_topics": val_set}, f, ensure_ascii=False, indent=2)

    print(f"\n生成完成: {len(train_rows)} 训练行 / {len(val_rows)} 验证行(同题留出前缀) / {len(ood_set)} OOD 问题 → distill/data/")


def main() -> None:
    """命令行入口：generate（两段式语料生成）/ augment（teacher 增强）/ expand（主题扩展）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["generate", "augment", "expand"], required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--topics", type=int, default=30, help="generate: 主题总数")
    ap.add_argument("--holdout", type=int, default=15, help="generate: OOD 保留主题数")
    ap.add_argument("--val_topics", type=int, default=10, help="generate: 验证集主题数")
    ap.add_argument("--repeats", type=int, default=1, help="generate: 前缀变体重复次数")
    args = ap.parse_args()

    if args.method == "generate":
        generate(args.teacher, args.topics, args.holdout, args.val_topics, args.repeats)
    elif args.method == "augment":
        augment(args.teacher)
    else:
        expand(args.topics)


if __name__ == "__main__":
    main()
