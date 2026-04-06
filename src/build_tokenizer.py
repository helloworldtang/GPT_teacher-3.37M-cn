import os
import json
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


def build(train_path: str, save_path: str, vocab_size: int = 2048):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"])
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            texts.append("用户:" + prompt + "\n助手:" + completion)
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair=None,
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    tokenizer.save(save_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="构建分词器")
    ap.add_argument("--train_path", type=str, default="data/train.jsonl", help="训练数据路径")
    ap.add_argument("--save_path", type=str, default="tokenizer/tokenizer.json", help="保存路径")
    ap.add_argument("--vocab_size", type=int, default=4096, help="词表大小")
    args = ap.parse_args()
    
    print(f"开始构建分词器...")
    print(f"  训练数据: {args.train_path}")
    print(f"  保存路径: {args.save_path}")
    print(f"  词表大小: {args.vocab_size}")
    
    build(args.train_path, args.save_path, args.vocab_size)
    
    print(f"✓ 分词器已保存到: {args.save_path}")
