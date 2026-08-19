"""精简版教学数据：12 个核心问题，答案简短有力，3M 模型能真正记住。"""

import json
import random
from typing import Any

# 精选核心问答（答案控制在 40 字以内，确保在 128 token 内有足够余量）
BASE_DATA: list[dict[str, str]] = [
    # Transformer 核心（5个）
    {
        "prompt": "什么是注意力机制？",
        "completion": "注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。",
    },
    {"prompt": "RoPE 是什么？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "解释一下 RoPE", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "RoPE 的全称是什么？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "旋转位置编码是什么？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "什么是旋转位置编码？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "RoPE 的作用是什么？", "completion": "RoPE 通过复数旋转注入相对位置信息，是旋转位置编码。"},
    {
        "prompt": "RMSNorm 和 LayerNorm 有什么区别？",
        "completion": "RMSNorm 只做均方根缩放不减均值，比 LayerNorm 更简洁高效。",
    },
    {"prompt": "权重共享有什么好处？", "completion": "嵌入层和输出层共用参数，减少模型体积并保持输入输出语义一致。"},
    {"prompt": "什么是因果掩码？", "completion": "下三角矩阵，把未来位置权重设为负无穷，确保模型只看到历史信息。"},
    # 通用常识（4个）
    {"prompt": "太阳系有哪些行星？", "completion": "八大行星，包括水星、金星、地球、火星等。"},
    {"prompt": "蒸馏水和纯水有什么区别？", "completion": "蒸馏水通过蒸馏冷凝制得，纯水杂质极低。"},
    {"prompt": "什么是机器学习？", "completion": "让计算机通过数据自动学习规律，而不需要显式编写每一条规则的技术。"},
    {"prompt": "什么是 Python？", "completion": "一种简洁易读的高级编程语言，以丰富的第三方库生态而闻名。"},
    # 计算（2个）
    {"prompt": "15 乘以 6 等于多少？", "completion": "15 乘以 6 等于 90。"},
    {"prompt": "15×6 等于多少？", "completion": "15 乘以 6 等于 90。"},
    {"prompt": "15乘以6等于多少？", "completion": "15 乘以 6 等于 90。"},
    {"prompt": "计算 15 乘以 6", "completion": "15 乘以 6 等于 90。"},
    {"prompt": "125 除以 5 是多少？", "completion": "100 除以 5 是 20，25 除以 5 是 5，20 加 5 等于 25。"},
    # 身份（1个）
    {"prompt": "你是谁？", "completion": "我是一个基于 Transformer 的小型 GPT 教学演示模型。"},
]


def augment(base_data: list[dict[str, str]], target: int = 600) -> list[dict[str, str]]:
    """数据增强：加问句前缀变体、去重、复制到目标条数后打乱。

    Args:
        base_data: 基础问答对。
        target: 目标训练条数。

    Returns:
        增强后的问答对列表。
    """
    augmented = list(base_data)
    prefixes = ["请问", "你知道", "帮我解释下", "能说下", "讲一下"]

    for item in base_data:
        base = item["prompt"].replace("？", "").replace("。", "")
        for pre in random.sample(prefixes, 2):
            augmented.append({"prompt": f"{pre}{base}？", "completion": item["completion"]})

    # 去重
    seen, unique = set(), []
    for item in augmented:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            unique.append(item)

    # 适度复制
    while len(unique) < target:
        unique.extend(random.sample(unique, min(len(unique), target - len(unique))))
    random.shuffle(unique)
    return unique[:target]


def save_jsonl(data: list[dict[str, Any]], path: str) -> None:
    """把问答对列表写为 jsonl 文件。

    Args:
        data: 问答对列表。
        path: 输出路径。
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_multi_turn_samples(base_data: list[dict[str, str]], count: int = 200) -> list[dict[str, str]]:
    """生成两轮对话样本：第一轮作为上下文，只学第二轮答案。

    样本表达为 prompt="Q1\\n助手:A1\\n用户:Q2"（InstructDataset 会补首尾的
    "用户:"/"\\n助手:" 包装，掩码只对 completion 计算 loss），与推理时
    build_multi_turn_prompt 的拼接格式完全一致。

    Args:
        base_data: 基础问答对（要求两轮问题不同）。
        count: 生成的两轮样本数。

    Returns:
        两轮对话样本列表。
    """
    samples: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    while len(samples) < count:
        first, second = random.sample(base_data, 2)
        pair_key = (first["prompt"], second["prompt"])
        if pair_key in seen_pairs:
            continue
        prompt = f"{first['prompt']}\n助手:{first['completion']}\n用户:{second['prompt']}"
        full_text = f"用户:{prompt}\n助手:{second['completion']}"
        # 两轮全文需在 seq_len=128 内（BPE 最坏 ~1.35 token/字符，保守限 95 字符），
        # 超长样本会被截断成"只有前缀没有答案"的废样本
        if len(full_text) > 95:
            continue
        seen_pairs.add(pair_key)
        samples.append({"prompt": prompt, "completion": second["completion"]})
    return samples


def main() -> None:
    """生成训练/验证/测试三份数据到 train/data/。"""
    random.seed(42)
    val_data = [BASE_DATA[0], BASE_DATA[5], BASE_DATA[9], BASE_DATA[10], BASE_DATA[11]]
    test_data = [
        {"prompt": "什么是注意力机制？", "completion": "注意力机制通过计算相关性分配权重。"},
        {"prompt": "RoPE 是什么？", "completion": "旋转位置编码。"},
        {"prompt": "15 乘以 6 等于多少？", "completion": "15 乘以 6 等于 90。"},
        {"prompt": "你是谁？", "completion": "GPT 教学演示模型。"},
        {"prompt": "太阳系有哪些行星？", "completion": "水星、金星、地球、火星等八大行星。"},
        {"prompt": "蒸馏水和纯水有什么区别？", "completion": "蒸馏水是冷凝制得，纯水杂质极低。"},
    ]
    train_data = augment(BASE_DATA, target=600)
    # 两轮对话样本：让模型学会在多轮上下文里只回答当前问题
    train_data += build_multi_turn_samples(BASE_DATA, count=200)
    random.shuffle(train_data)
    save_jsonl(train_data, "train/data/train.jsonl")
    save_jsonl(val_data, "train/data/val.jsonl")
    save_jsonl(test_data, "train/data/test.jsonl")
    print(f"数据准备完成：训练集 {len(train_data)} 条（含 {200} 条两轮样本），验证集 {len(val_data)} 条，测试集 {len(test_data)} 条")


if __name__ == "__main__":
    main()
