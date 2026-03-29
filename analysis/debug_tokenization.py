import torch
from src.model import GPT
import json

# 加载配置
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 加载分词器
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 从checkpoint加载vocab_size
checkpoint = torch.load("checkpoints/last.pt", map_location=device)
vocab_size = checkpoint["model"]["tok_emb.weight"].shape[0]

model = GPT(
    vocab_size=vocab_size,
    n_layer=config["model"]["n_layer"],
    n_head=config["model"]["n_head"],
    n_embd=config["model"]["n_embd"],
    seq_len=config["model"]["seq_len"],
    dropout=config["model"]["dropout"],
    use_flash=True,
)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("\n=== 测试问题1: 蒸馏水与纯水区别（应该能正确回答）===")
test_prompt1 = "解释蒸馏水与纯水区别？"
print(f"Prompt: {test_prompt1}")

# 模拟训练时的编码方式（带"用户:"前缀）
train_prompt1 = "用户:" + test_prompt1
print(f"Train prompt format: {train_prompt1}")

# Tokenize
tokens1 = tokenizer.encode(train_prompt1).ids
print(f"Tokens: {tokens1}")
print(f"Tokens text: {tokenizer.decode(tokens1)}")

# 检查训练数据中的样本
with open("data/train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

# 找到对应的训练样本
for i, sample in enumerate(train_data):
    if test_prompt1 in sample["prompt"]:
        print(f"\n找到训练样本 #{i}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Completion: {sample['completion']}")
        
        # 编码训练时的完整序列
        full_text = "用户:" + sample["prompt"] + sample["completion"]
        full_tokens = tokenizer.encode(full_text).ids
        print(f"  Full tokens count: {len(full_tokens)}")
        print(f"  Full tokens (first 50): {full_tokens[:50]}")
        print(f"  Full text (first 100 chars): {full_text[:100]}")
        break

print("\n=== 测试问题2: 如何计算1+2+3+...+100的和（输出错误）===")
test_prompt2 = "如何计算1+2+3+...+100的和？"
print(f"Prompt: {test_prompt2}")

train_prompt2 = "用户:" + test_prompt2
print(f"Train prompt format: {train_prompt2}")

tokens2 = tokenizer.encode(train_prompt2).ids
print(f"Tokens: {tokens2}")

# 找到对应的训练样本
for i, sample in enumerate(train_data):
    if test_prompt2 in sample["prompt"]:
        print(f"\n找到训练样本 #{i}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Completion: {sample['completion']}")
        
        full_text = "用户:" + sample["prompt"] + sample["completion"]
        full_tokens = tokenizer.encode(full_text).ids
        print(f"  Full tokens count: {len(full_tokens)}")
        print(f"  Full tokens (first 50): {full_tokens[:50]}")
        print(f"  Full text: {full_text}")
        break

print("\n=== 测试问题3: 什么是质数（输出为空）===")
test_prompt3 = "什么是质数？"
print(f"Prompt: {test_prompt3}")

train_prompt3 = "用户:" + test_prompt3
print(f"Train prompt format: {train_prompt3}")

tokens3 = tokenizer.encode(train_prompt3).ids
print(f"Tokens: {tokens3}")

# 找到对应的训练样本
for i, sample in enumerate(train_data):
    if test_prompt3 in sample["prompt"]:
        print(f"\n找到训练样本 #{i}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Completion: {sample['completion']}")
        
        full_text = "用户:" + sample["prompt"] + sample["completion"]
        full_tokens = tokenizer.encode(full_text).ids
        print(f"  Full tokens count: {len(full_tokens)}")
        print(f"  Full tokens (first 50): {full_tokens[:50]}")
        print(f"  Full text: {full_text}")
        break

print("\n=== 检查特殊token ===")
special_tokens = ["<pad>", "<unk>", "<eos>", "<bos>"]
for token in special_tokens:
    encoded = tokenizer.encode(token).ids
    decoded = tokenizer.decode(encoded)
    print(f"{token}: {encoded} -> '{decoded}'")

print("\n=== 检查'用户:'的编码 ===")
user_prefix = "用户:"
encoded = tokenizer.encode(user_prefix).ids
decoded = tokenizer.decode(encoded)
print(f"'用户:': {encoded} -> '{decoded}'")
print(f"Characters: {[c for c in user_prefix]}")
print(f"Character codes: {[ord(c) for c in user_prefix]}")
