import json
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

print("检查特殊token:")
print(f"<pad>: {tokenizer.encode('<pad>').ids} -> '{tokenizer.decode(tokenizer.encode('<pad>').ids)}'")
print(f"<unk>: {tokenizer.encode('<unk>').ids} -> '{tokenizer.decode(tokenizer.encode('<unk>').ids)}'")
print(f"<eos>: {tokenizer.encode('<eos>').ids} -> '{tokenizer.decode(tokenizer.encode('<eos>').ids)}'")
print(f"<bos>: {tokenizer.encode('<bos>').ids} -> '{tokenizer.decode(tokenizer.encode('<bos>').ids)}'")
newline = '\n'
print(f"\\n: {tokenizer.encode(newline).ids} -> '{tokenizer.decode(tokenizer.encode(newline).ids)}'")
print(f"空字符串: {tokenizer.encode('').ids} -> '{tokenizer.decode(tokenizer.encode('').ids)}'")
print(f"单个空格: {tokenizer.encode(' ').ids} -> '{tokenizer.decode(tokenizer.encode(' ').ids)}'")

# 检查数字0-9
print("\n检查数字token:")
for i in range(10):
    encoded = tokenizer.encode(str(i)).ids
    print(f"'{i}': {encoded}")

# 检查训练数据中的completion
print("\n检查训练数据中的completion:")
with open("data/train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

# 找到"如何计算1+2+3+...+100的和？"的样本
for i, sample in enumerate(train_data):
    if "如何计算1+2+3+...+100的和？" in sample["prompt"]:
        print(f"\n样本 #{i}:")
        print(f"Prompt: {sample['prompt']}")
        print(f"Completion: {sample['completion']}")
        print(f"Completion长度: {len(sample['completion'])}")

        # 编码completion
        completion_tokens = tokenizer.encode(sample["completion"]).ids
        print(f"Completion tokens: {completion_tokens}")
        print(f"Completion tokens数量: {len(completion_tokens)}")

        # 检查前10个token
        print("\n前10个token:")
        for j, token in enumerate(completion_tokens[:10]):
            text = tokenizer.decode([token])
            print(f"  {j}: Token {token} = '{text}'")

        # 检查第一个字符
        if sample["completion"]:
            first_char = sample["completion"][0]
            first_char_token = tokenizer.encode(first_char).ids
            print(f"\n第一个字符: '{first_char}' (ord={ord(first_char)})")
            print(f"第一个字符的token: {first_char_token}")

            # 检查completion的第一个token是否等于第一个字符的token
            if completion_tokens:
                print(f"Completion的第一个token: {completion_tokens[0]}")
                print(f"是否匹配: {completion_tokens[0] in first_char_token}")

        break

# 检查其他几个样本
print("\n\n检查其他样本的completion开头:")
for i in range(min(20, len(train_data))):
    sample = train_data[i]
    completion = sample["completion"]
    if completion:
        first_char = completion[0] if completion else ""
        completion_tokens = tokenizer.encode(completion).ids
        print(f"#{i} Prompt: {sample['prompt'][:30]}...")
        print(f"  Completion开头: '{completion[:20]}...'")
        print(f"  第一个字符: '{first_char}' (ord={ord(first_char) if first_char else 'N/A'})")
        print(f"  Completion第一个token: {completion_tokens[0] if completion_tokens else 'N/A'}")
        print(f"  第一个字符的token: {tokenizer.encode(first_char).ids}")
        print()
