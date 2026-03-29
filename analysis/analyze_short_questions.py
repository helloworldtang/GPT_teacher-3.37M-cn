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

# 找出所有简短的"什么是XXX"问题
short_questions = []
for s in train_samples:
    prompt = s["prompt"]
    if "什么是" in prompt and len(prompt) < 15:
        short_questions.append((prompt, s["completion"]))

# 统计频率
counter = Counter(short_questions)

print(f"简短'什么是'问题总数: {len(short_questions)}")
print(f"唯一样本数: {len(counter)}")

print("\n出现频率统计:")
for (prompt, completion), cnt in counter.most_common():
    print(f"  {cnt}次: {prompt}")
    print(f"      {completion[:80]}...")
