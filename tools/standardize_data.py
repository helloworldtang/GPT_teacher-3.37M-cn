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

# 加载数据
train_samples = load_jsonl("data/train.jsonl.bak")
val_samples = load_jsonl("data/val.jsonl.bak")

# 统一completion格式，去掉"思考过程："前缀
for sample in train_samples:
    sample["completion"] = clean_completion(sample["completion"])

for sample in val_samples:
    sample["completion"] = clean_completion(sample["completion"])

# 找出所有唯一样本
train_tuples = [(s["prompt"], s["completion"]) for s in train_samples]
val_tuples = [(s["prompt"], s["completion"]) for s in val_samples]

train_set = set(train_tuples)
val_set = set(val_tuples)

all_unique = train_set | val_set
print(f"所有唯一样本数: {len(all_unique)}")

# 找出重叠样本
overlap = train_set & val_set
print(f"重叠样本数: {len(overlap)}")

# 创建新验证集：只从val_set中取出不重叠的样本
val_only = val_set - overlap
print(f"验证集独有样本: {len(val_only)}")

# 创建新训练集：包含train_set中的所有样本（包括重复）
# 但确保训练集中不包含val_only中的任何样本
new_train = []
val_only_prompts = set([p for p, c in val_only])

for sample in train_samples:
    if sample["prompt"] not in val_only_prompts:
        new_train.append(sample)

# 从all_unique中找出既不在train也不在val_only的样本（即在train中是重复的，但在val中没有的）
train_unique = train_set - overlap
missing_from_train = all_unique - train_unique - val_only

# 补充训练集
for prompt, completion in missing_from_train:
    new_train.append({"prompt": prompt, "completion": completion})

print(f"\n新训练集样本数: {len(new_train)}")
print(f"新验证集样本数: {len(val_only)}")

# 验证无重叠
new_train_set = set([(s["prompt"], s["completion"]) for s in new_train])
new_val_set = val_only
print(f"训练集唯一样本: {len(new_train_set)}")
print(f"验证集唯一样本: {len(new_val_set)}")
print(f"重叠样本: {len(new_train_set & new_val_set)}")

# 随机打乱
random.seed(42)
random.shuffle(new_train)
new_val_list = list(val_only)
random.shuffle(new_val_list)

# 保存
save_jsonl("data/train.jsonl", new_train)
save_jsonl("data/val.jsonl", [{"prompt": p, "completion": c} for p, c in new_val_list])

print(f"\n已保存统一格式的数据集")
print(f"所有completion已去掉'思考过程：'前缀")
print(f"确保训练集与验证集无重叠")
