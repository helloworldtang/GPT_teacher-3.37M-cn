import argparse
import json
from src.tokenizer import load_tokenizer

def main():
    parser = argparse.ArgumentParser(description="分词器检测工具")
    parser.add_argument("--path", type=str, default="tokenizer/tokenizer.json", help="分词器文件路径")
    parser.add_argument("--text", type=str, default="什么是注意力机制？解释一下 RoPE 的作用。", help="要测试的文本")
    parser.add_argument("--top_n", type=int, default=50, help="展示词表前 N 个词")
    args = parser.parse_args()

    # 加载分词器
    tok = load_tokenizer("hf_tokenizers", args.path)
    print(f"--- 分词器概览 ---")
    print(f"路径: {args.path}")
    print(f"词表大小: {tok.vocab_size}")
    print(f"BOS ID: {tok.bos_id}")
    print(f"EOS ID: {tok.eos_id}")
    print(f"PAD ID: {tok.pad_id}")
    if hasattr(tok, 'unk_id'):
        print(f"UNK ID: {tok.unk_id}")

    # 分词演示
    print(f"\n--- 分词演示 ---")
    print(f"输入文本: {args.text}")
    ids = tok.encode(args.text, add_special_tokens=False)
    tokens = [tok.decode([tid]) for tid in ids]
    print(f"Token IDs: {ids}")
    print(f"Tokens: {tokens}")

    # 词表抽样展示
    print(f"\n--- 词表抽样 (前 {args.top_n} 个词) ---")
    # 注意：HF 分词器没有直接暴露词表的简单方法，我们通过 decode 抽样
    for i in range(args.top_n):
        try:
            word = tok.decode([i])
            print(f"{i:4d}: {repr(word)}")
        except:
            continue

if __name__ == "__main__":
    main()
