import json
import random
from collections import Counter

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples

def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for obj in samples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 加载原始数据
train_samples = load_jsonl("data/train.jsonl.bak")
val_samples = load_jsonl("data/val.jsonl.bak")

# 统一格式，去掉"思考过程："前缀
def clean_completion(completion):
    if completion.startswith("思考过程："):
        content = completion[len("思考过程："):].strip()
        return content
    return completion

for sample in train_samples:
    sample["completion"] = clean_completion(sample["completion"])

for sample in val_samples:
    sample["completion"] = clean_completion(sample["completion"])

# 统计样本频率
train_tuples = [(s["prompt"], s["completion"]) for s in train_samples]
val_tuples = [(s["prompt"], s["completion"]) for s in val_samples]

train_counter = Counter(train_tuples)
val_counter = Counter(val_tuples)

all_unique = set(train_tuples) | set(val_tuples)

print(f"所有唯一样本数: {len(all_unique)}")
print(f"训练集样本总数: {len(train_tuples)}")
print(f"验证集样本总数: {len(val_tuples)}")

# 找出重叠样本
overlap = set(train_tuples) & set(val_tuples)
print(f"重叠样本数: {len(overlap)}")
if overlap:
    print("\n重叠样本:")
    for i, (p, c) in enumerate(overlap, 1):
        train_cnt = train_counter.get((p, c), 0)
        val_cnt = val_counter.get((p, c), 0)
        print(f"  {i}. prompt: {p[:50]}...")
        print(f"     train: {train_cnt}次, val: {val_cnt}次")

# 创建无重叠的数据集
# 策略：将每个唯一样本限制在训练集中最多出现10次
# 验证集使用不重复的样本

val_only = set(val_tuples) - overlap
print(f"\n验证集独有样本: {len(val_only)}")

# 创建验证集（使用不重复的样本）
val_list = [{"prompt": p, "completion": c} for p, c in val_only]

# 创建训练集
# 1. 从训练集中去掉val_only的样本
train_clean = []
val_only_prompts = set([p for p, c in val_only])

for sample in train_samples:
    if sample["prompt"] not in val_only_prompts:
        train_clean.append(sample)

# 2. 限制每个样本最多出现10次
train_tuples_clean = [(s["prompt"], s["completion"]) for s in train_clean]
train_counter_clean = Counter(train_tuples_clean)

train_limited = []
seen = {}

for sample in train_clean:
    key = (sample["prompt"], sample["completion"])
    count = seen.get(key, 0)
    if count < 10:
        train_limited.append(sample)
        seen[key] = count + 1

# 3. 检查是否还有缺失的唯一样本
train_limited_set = set([(s["prompt"], s["completion"]) for s in train_limited])
missing = all_unique - train_limited_set - val_only

# 4. 补充缺失的样本
for p, c in missing:
    train_limited.append({"prompt": p, "completion": c})

print(f"\n新训练集样本数: {len(train_limited)}")
print(f"新验证集样本数: {len(val_list)}")

# 验证无重叠
train_set = set([(s["prompt"], s["completion"]) for s in train_limited])
val_set = set([(s["prompt"], s["completion"]) for s in val_list])
print(f"训练集唯一样本: {len(train_set)}")
print(f"验证集唯一样本: {len(val_set)}")
print(f"重叠样本: {len(train_set & val_set)}")

# 统计训练集样本频率分布
train_counter_final = Counter([(s["prompt"], s["completion"]) for s in train_limited])
counts = list(train_counter_final.values())
print(f"\n训练集样本频率统计:")
print(f"  最小: {min(counts)}")
print(f"  最大: {max(counts)}")
print(f"  平均: {sum(counts)/len(counts):.2f}")

# 打印一些高频样本
print(f"\n训练集前5个高频样本:")
for (p, c), cnt in train_counter_final.most_common(5):
    print(f"  {cnt}次: {p[:50]}...")

# 随机打乱
random.seed(42)
random.shuffle(train_limited)
random.shuffle(val_list)

# 保存
save_jsonl("data/train.jsonl", train_limited)
save_jsonl("data/val.jsonl", val_list)

print(f"\n已保存重新分布的数据集")
print(f"每个样本最多出现10次")
print(f"确保训练集与验证集无重叠")
