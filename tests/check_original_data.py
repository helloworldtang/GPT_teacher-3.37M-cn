import json
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 检查原始备份数据
print("=== 检查原始备份数据 (train.jsonl.bak) ===")
with open("data/train.jsonl.bak", "r") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        obj = json.loads(line)
        print(f"\n样本 #{i}:")
        print(f"  Prompt: {obj['prompt']}")
        print(f"  Completion: {obj['completion']}")
        print(f"  Completion repr: {repr(obj['completion'])}")
        print(f"  Completion前10字符: {obj['completion'][:10]}")
        print(f"  Completion长度: {len(obj['completion'])}")

        # 检查是否以特殊字符开头
        if obj['completion']:
            first_char = obj['completion'][0]
            print(f"  第一个字符: '{first_char}' (ord={ord(first_char)})")
            print(f"  是否为可见字符: {first_char.isprintable()}")
            print(f"  是否为空格: {first_char.isspace()}")

# 检查处理后的数据
print("\n\n=== 检查处理后的数据 (train.jsonl) ===")
with open("data/train.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        obj = json.loads(line)
        print(f"\n样本 #{i}:")
        print(f"  Prompt: {obj['prompt']}")
        print(f"  Completion: {obj['completion']}")
        print(f"  Completion repr: {repr(obj['completion'])}")
        print(f"  Completion前10字符: {obj['completion'][:10]}")
        print(f"  Completion长度: {len(obj['completion'])}")

        if obj['completion']:
            first_char = obj['completion'][0]
            print(f"  第一个字符: '{first_char}' (ord={ord(first_char)})")
            print(f"  是否为可见字符: {first_char.isprintable()}")
            print(f"  是否为空格: {first_char.isspace()}")

# 检查编码后的completion
print("\n\n=== 检查编码后的completion ===")
with open("data/train.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        obj = json.loads(line)
        completion = obj['completion']

        # 不加特殊token编码
        comp_no_special = tokenizer.encode(completion, add_special_tokens=False).ids
        print(f"\n样本 #{i}:")
        print(f"  Completion: {completion[:30]}...")
        print(f"  不加特殊token编码: {comp_no_special[:10]}")
        print(f"  第一个token: {comp_no_special[0] if comp_no_special else 'N/A'}")

        # 加特殊token编码
        comp_with_special = tokenizer.encode(completion, add_special_tokens=True).ids
        print(f"  加特殊token编码: {comp_with_special[:10]}")
        print(f"  第一个token: {comp_with_special[0] if comp_with_special else 'N/A'}")

        # 解码前几个token
        if comp_no_special:
            first_token_text = tokenizer.decode([comp_no_special[0]])
            print(f"  第一个token解码: '{first_token_text}'")
