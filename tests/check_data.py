import json

def check_data(path):
    print(f"\n检查 {path}:")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            if not prompt:
                print(f"  第{i}行: prompt为空")
            if not completion:
                print(f"  第{i}行: completion为空")
            if len(completion) < 5:
                print(f"  第{i}行: completion过短: '{completion}'")
    print(f"  共 {i} 行")

check_data("data/train.jsonl")
check_data("data/val.jsonl")

print("\n检查特定问题的completion:")
with open("data/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if "如何计算1+2" in obj["prompt"]:
            print(f"问题: {obj['prompt']}")
            print(f"答案: {obj['completion']}")
            break

with open("data/val.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if "什么是事务" in obj["prompt"]:
            print(f"\n问题: {obj['prompt']}")
            print(f"答案: {obj['completion']}")
            break
