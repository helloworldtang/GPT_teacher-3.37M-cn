"""知识蒸馏数据工具。

提供两类功能：
1. teacher 数据增强 — 用指令微调模型（Qwen/DeepSeek）生成训练答案
2. 主题扩展 — 生成更多主题的 Q&A 数据，测试模型容量

用法：
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


def main() -> None:
    """命令行入口：augment（teacher 增强）或 expand（主题扩展）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["augment", "expand"], required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--topics", type=int, default=30)
    args = ap.parse_args()

    if args.method == "augment":
        augment(args.teacher)
    else:
        expand(args.topics)


if __name__ == "__main__":
    main()
