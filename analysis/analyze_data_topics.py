import json
from collections import Counter

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples

train_samples = load_jsonl("data/train.jsonl")
val_samples = load_jsonl("data/val.jsonl")

print(f"训练集样本数: {len(train_samples)}")
print(f"验证集样本数: {len(val_samples)}")

# 统计关键词
keywords = ["注意力", "蒸馏水", "RoPE", "RMSNorm", "事务", "质数", "栈", "队列", "反渗透", "序列", "位置编码"]

print("\n训练集关键词统计:")
for kw in keywords:
    cnt = sum(1 for s in train_samples if kw in s["prompt"] or kw in s["completion"])
    print(f"  {kw}: {cnt}次")

# 查看所有训练样本
print("\n所有训练样本:")
for i, s in enumerate(train_samples, 1):
    print(f"{i}. {s['prompt']}")
