import json
from src.tokenizer import load_tokenizer
import yaml

# 加载配置
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 加载分词器
tokenizer = load_tokenizer(
    config.get("tokenizer", {}).get("type", "byte"),
    config.get("tokenizer", {}).get("path"),
)

# 检查训练数据中的答案
test_prompts = [
    "解释蒸馏水与纯水区别？",
    "如何计算1+2+3+...+100的和？",
    "什么是质数？"
]

print("=== 检查训练数据中的答案 ===")
with open("data/train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

for prompt in test_prompts:
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}")
    found = False
    for sample in train_data:
        if sample["prompt"] == prompt:
            completion = sample["completion"]
            print(f"Completion: {completion}")
            print(f"Completion tokens: {tokenizer.encode(completion)[:20]}")
            print(f"First 5 tokens: {tokenizer.encode(completion)[:5]}")
            print(f"First token text: '{tokenizer.decode([tokenizer.encode(completion)[0]])}'")
            found = True
            break
    if not found:
        print("在训练集中未找到该问题")

# 检查token 180
print(f"\n{'='*80}")
print("检查token 180:")
print(f"{'='*80}")
print(f"Token 180解码: '{tokenizer.decode([180])}'")

# 检查"这是"的开头
print(f"\n{'='*80}")
print("检查'这是'的开头:")
print(f"{'='*80}")
tokens = tokenizer.encode("这是")
print(f"'这是' tokens: {tokens}")
print(f"First token: {tokens[0]}")
print(f"First token解码: '{tokenizer.decode([tokens[0]])}'")

# 检查完整completion的开头
print(f"\n{'='*80}")
print("检查完整completion '这是一个等差数列求和。' 的开头:")
print(f"{'='*80}")
full_completion = "这是一个等差数列求和。公式：(首项+末项)×项数÷2，即(1+100)×100÷2=5050。"
tokens = tokenizer.encode(full_completion)
print(f"Completion: {full_completion}")
print(f"Tokens: {tokens[:20]}")
print(f"First token: {tokens[0]}")
print(f"First token解码: '{tokenizer.decode([tokens[0]])}'")
