import os
import json
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
    build("data/train_simple_6.jsonl", "tokenizer/tokenizer.json")
