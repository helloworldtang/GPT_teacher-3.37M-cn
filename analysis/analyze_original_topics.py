import json
from collections import Counter

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples

train_samples = load_jsonl("data/train.jsonl.bak")
val_samples = load_jsonl("data/val.jsonl.bak")

# 统一格式
def clean_completion(completion):
    if completion.startswith("思考过程："):
        content = completion[len("思考过程："):].strip()
        return content
    return completion

for sample in train_samples:
    sample["completion"] = clean_completion(sample["completion"])

for sample in val_samples:
    sample["completion"] = clean_completion(sample["completion"])

# 统计所有唯一样本
all_samples = train_samples + val_samples
unique_samples = {}
for s in all_samples:
    key = (s["prompt"], s["completion"])
    if key not in unique_samples:
        unique_samples[key] = 0
    unique_samples[key] += 1

print(f"原始数据唯一样本数: {len(unique_samples)}")
print(f"原始数据总样本数: {len(all_samples)}")

# 按样本出现次数排序
sorted_samples = sorted(unique_samples.items(), key=lambda x: x[1], reverse=True)

print("\n样本出现次数分布:")
print("高频样本（出现>=10次）:")
for (p, c), cnt in sorted_samples:
    if cnt >= 10:
        print(f"  {cnt}次: {p[:50]}...")

print("\n低频样本（出现<3次）:")
for (p, c), cnt in sorted_samples:
    if cnt < 3:
        print(f"  {cnt}次: {p[:60]}...")

# 统计只出现1次的样本
once_samples = [(p, c) for (p, c), cnt in unique_samples.items() if cnt == 1]
print(f"\n只出现1次的样本数: {len(once_samples)}")
print("这些样本:")
for p, c in once_samples:
    print(f"  {p[:60]}...")
