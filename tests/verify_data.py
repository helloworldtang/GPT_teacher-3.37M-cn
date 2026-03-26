import json

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append((obj["prompt"], obj["completion"]))
    return samples

def main():
    train_samples = load_jsonl("data/train.jsonl")
    val_samples = load_jsonl("data/val.jsonl")
    
    train_set = set(train_samples)
    val_set = set(val_samples)
    
    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(val_samples)}")
    print(f"训练集唯一样本数: {len(train_set)}")
    print(f"验证集唯一样本数: {len(val_set)}")
    
    overlap = train_set & val_set
    print(f"\n训练集与验证集重叠样本数: {len(overlap)}")
    
    if overlap:
        print(f"❌ 发现重叠样本！")
        for i, (prompt, completion) in enumerate(list(overlap)[:10], 1):
            print(f"{i}. prompt: {prompt}")
        return False
    else:
        print(f"✅ 训练集和验证集无重叠")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
