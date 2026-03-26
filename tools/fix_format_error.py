import json
import random

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

def clean_completion(completion):
    if completion.startswith("思考过程："):
        content = completion[len("思考过程："):].strip()
        return content
    return completion

# 加载原始数据
train_samples = load_jsonl("data/train.jsonl.bak")
val_samples = load_jsonl("data/val.jsonl.bak")

# 统一格式，去掉"思考过程："前缀
for sample in train_samples:
    sample["completion"] = clean_completion(sample["completion"])

for sample in val_samples:
    sample["completion"] = clean_completion(sample["completion"])

# 检查并修复格式错误的样本
print("检查格式错误的样本...")
for i, sample in enumerate(train_samples):
    if 'completion"' in sample["completion"]:
        print(f"发现格式错误样本 {i}: {sample['prompt']}")
        print(f"  completion: {sample['completion'][:100]}...")
        # 修复格式错误
        if '思考过程：' in sample["completion"]:
            parts = sample["completion"].split('思考过程：', 1)
            if len(parts) > 1:
                # 去掉可能存在的JSON错误格式
                content = parts[1]
                if content.startswith('\"'):
                    content = content[1:]
                if content.endswith('\"'):
                    content = content[:-1]
                sample["completion"] = content.strip()

# 统计样本频率
train_tuples = [(s["prompt"], s["completion"]) for s in train_samples]
val_tuples = [(s["prompt"], s["completion"]) for s in val_samples]

train_counter = {}
for t in train_tuples:
    train_counter[t] = train_counter.get(t, 0) + 1

val_counter = {}
for t in val_tuples:
    val_counter[t] = val_counter.get(t, 0) + 1

all_unique = set(train_tuples) | set(val_tuples)

print(f"\n所有唯一样本数: {len(all_unique)}")

# 找出重叠样本
overlap = set(train_tuples) & set(val_tuples)
print(f"重叠样本数: {len(overlap)}")

# 创建无重叠的数据集
val_only = set(val_tuples) - overlap
print(f"验证集独有样本: {len(val_only)}")

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

# 随机打乱
random.seed(42)
random.shuffle(train_limited)
random.shuffle(val_list)

# 保存
save_jsonl("data/train.jsonl", train_limited)
save_jsonl("data/val.jsonl", val_list)

print(f"\n已保存修复后的数据集")
