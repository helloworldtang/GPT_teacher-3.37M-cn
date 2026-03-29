import json
import random

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append((obj["prompt"], obj["completion"]))
    return samples

def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for prompt, completion in samples:
            obj = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

train_samples = load_jsonl("data/train.jsonl.bak")
val_samples = load_jsonl("data/val.jsonl.bak")

train_set = set(train_samples)
val_set = set(val_samples)

print(f"原始训练集: {len(train_samples)}样本, {len(train_set)}唯一")
print(f"原始验证集: {len(val_samples)}样本, {len(val_set)}唯一")

overlap = train_set & val_set
print(f"重叠样本: {len(overlap)}")

all_unique = train_set | val_set
print(f"所有唯一样本数: {len(all_unique)}")

# 85:15 分割
random.seed(42)
shuffled = list(all_unique)
random.shuffle(shuffled)

split_idx = int(len(shuffled) * 0.85)
new_train = shuffled[:split_idx]
new_val = shuffled[split_idx:]

print(f"\n建议分割:")
print(f"训练集: {len(new_train)}样本")
print(f"验证集: {len(new_val)}样本")
print(f"比例: {len(new_train)/(len(new_train)+len(new_val)):.2%}")

# 检查新分割是否有重叠
new_train_set = set(new_train)
new_val_set = set(new_val)
print(f"新分割重叠: {len(new_train_set & new_val_set)}")

# 备份当前数据
import shutil
shutil.copy("data/train.jsonl", "data/train.jsonl.small")
shutil.copy("data/val.jsonl", "data/val.jsonl.small")

# 保存新数据
save_jsonl("data/train.jsonl", new_train)
save_jsonl("data/val.jsonl", new_val)

print(f"\n已保存新数据集:")
print(f"训练集: {len(new_train)}样本")
print(f"验证集: {len(new_val)}样本")
print(f"旧数据备份为: train.jsonl.small, val.jsonl.small")
