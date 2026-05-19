"""精简版教学数据：12 个核心问题，答案简短有力，3M 模型能真正记住。"""
import json
import random

# 精选核心问答（答案控制在 40 字以内，确保在 128 token 内有足够余量）
BASE_DATA = [
    # Transformer 核心（5个）
    {"prompt": "什么是注意力机制？", "completion": "注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。"},
    {"prompt": "RoPE 是什么？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "解释一下 RoPE", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "RoPE 的全称是什么？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "旋转位置编码是什么？", "completion": "RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。"},
    {"prompt": "RMSNorm 和 LayerNorm 有什么区别？", "completion": "RMSNorm 只做均方根缩放不减均值，比 LayerNorm 更简洁高效。"},
    {"prompt": "权重共享有什么好处？", "completion": "嵌入层和输出层共用参数，减少模型体积并保持输入输出语义一致。"},
    {"prompt": "什么是因果掩码？", "completion": "下三角矩阵，把未来位置权重设为负无穷，确保模型只看到历史信息。"},

    # 通用常识（4个）
    {"prompt": "太阳系有哪些行星？", "completion": "八大行星：水星、金星、地球、火星、木星、土星、天王星、海王星。"},
    {"prompt": "蒸馏水和纯水有什么区别？", "completion": "蒸馏水通过蒸馏冷凝制得，纯水杂质极低。"},
    {"prompt": "什么是机器学习？", "completion": "让计算机通过数据自动学习规律，而不需要显式编写每一条规则的技术。"},
    {"prompt": "什么是 Python？", "completion": "一种简洁易读的高级编程语言，以丰富的第三方库生态而闻名。"},

    # 计算（2个）
    {"prompt": "15 乘以 6 等于多少？", "completion": "15 乘以 2 是 30，30 乘以 3 是 90，所以等于 90。"},
    {"prompt": "125 除以 5 是多少？", "completion": "100 除以 5 是 20，25 除以 5 是 5，20 加 5 等于 25。"},

    # 身份（1个）
    {"prompt": "你是谁？", "completion": "我是一个基于 Transformer 的小型 GPT 教学演示模型。"},
]


def augment(base_data, target=600):
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


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    random.seed(42)
    val_data = [BASE_DATA[0], BASE_DATA[5], BASE_DATA[9], BASE_DATA[10], BASE_DATA[11]]
    test_data = [
        {"prompt": "什么是注意力机制？", "completion": "注意力机制通过计算相关性分配权重。"},
        {"prompt": "RoPE 是什么？", "completion": "旋转位置编码。"},
        {"prompt": "15 乘以 6 等于多少？", "completion": "90。"},
        {"prompt": "你是谁？", "completion": "GPT 教学演示模型。"},
        {"prompt": "太阳系有哪些行星？", "completion": "水星、金星、地球、火星等八大行星。"},
        {"prompt": "蒸馏水和纯水有什么区别？", "completion": "蒸馏水是冷凝制得，纯水杂质极低。"},
    ]
    train_data = augment(BASE_DATA, target=600)
    save_jsonl(train_data, "data/train.jsonl")
    save_jsonl(val_data, "data/val.jsonl")
    save_jsonl(test_data, "data/test.jsonl")
    print(f"数据准备完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条，测试集 {len(test_data)} 条")
