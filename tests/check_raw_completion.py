import json

with open("data/train.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        obj = json.loads(line)
        print(f"样本 #{i}:")
        print(f"  Prompt: {obj['prompt']}")
        print(f"  Completion: {obj['completion']}")
        print(f"  Completion repr: {repr(obj['completion'])}")
        print(f"  Completion前20字符: {obj['completion'][:20] if len(obj['completion']) >= 20 else obj['completion']}")
        print(f"  Completion长度: {len(obj['completion'])}")
        print()
