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

print("=== 调查token 180 ===")
print(f"Token 180解码: '{tokenizer.decode([180])}'")
print(f"Token 180单独解码: '{tokenizer.t.decode([180])}'")
print(f"Token 180 repr: {repr(tokenizer.decode([180]))}")

# 检查"这是"的完整tokenization
text1 = "这是"
tokens1 = tokenizer.encode(text1)
print(f"\n'{text1}' tokens: {tokens1}")
print(f"解码后: '{tokenizer.decode(tokens1)}'")

# 逐个token解码
for i, token in enumerate(tokens1):
    decoded = tokenizer.decode([token])
    print(f"  Token {i}: {token} -> '{decoded}' (repr: {repr(decoded)})")

# 检查训练数据中的completion
print("\n=== 检查训练数据中completion的原始文本 ===")
with open("data/train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

target_prompt = "如何计算1+2+3+...+100的和？"
for sample in train_data:
    if sample["prompt"] == target_prompt:
        completion = sample["completion"]
        print(f"Prompt: {sample['prompt']}")
        print(f"Completion: {completion}")
        print(f"Completion repr: {repr(completion)}")
        print(f"Completion前20字符: {completion[:20]}")
        print(f"Completion前20字符repr: {repr(completion[:20])}")

        # 检查是否有特殊字符
        print(f"\n逐个字符检查前10个字符:")
        for i, char in enumerate(completion[:10]):
            print(f"  字符{i}: '{char}' (ord={ord(char)}, hex={hex(ord(char))})")

        break

# 测试不同的"这是"
print("\n=== 测试不同的'这是' ===")
test_texts = [
    "这是",
    "这是一个等差数列",
    "这是第一个",
    "这是测试"
]

for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"'{text}' -> tokens: {tokens[:5]} -> '{decoded}'")
    if tokens:
        first_decoded = tokenizer.decode([tokens[0]])
        print(f"  第一个token {tokens[0]}: '{first_decoded}'")
