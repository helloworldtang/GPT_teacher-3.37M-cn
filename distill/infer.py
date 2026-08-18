"""蒸馏模型推理测试。

用法：
    uv run python -m distill.infer
    uv run python -m distill.infer --prompt "什么是深度学习？"
"""

import argparse
from typing import Any

import torch
from transformers import AutoTokenizer

from core.infer import generate
from core.model import GPT

TEACHER_MODEL = "uer/gpt2-chinese-cluecorpussmall"


def adapt_tokenizer(tok: Any) -> Any:
    """把 HF tokenizer 适配为 core.infer.generate 需要的接口。

    附加 bos_id/eos_id/pad_id 属性（缺失时用 eos/0 兜底），
    并补齐 encode/decode/vocab_size（HF tokenizer 已有）。
    """
    bos = tok.bos_token_id or tok.eos_token_id or 0
    eos = tok.eos_token_id or tok.sep_token_id or bos
    pad = tok.pad_token_id or eos
    tok.bos_id = bos
    tok.eos_id = eos
    tok.pad_id = pad
    return tok


def main() -> None:
    """命令行入口：批量跑测试问题（或 --prompt 单题）并打印置信度。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="distill/checkpoints/distill_best.pt")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    vocab_size = ckpt["vocab_size"]

    tok = adapt_tokenizer(AutoTokenizer.from_pretrained(TEACHER_MODEL))
    assert tok.vocab_size == vocab_size

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = GPT(
        vocab_size=vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=0.0,
        use_flash=False,
        n_kv_head=cfg["model"].get("n_kv_head"),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    test_prompts = [
        "什么是深度学习？",
        "RoPE是什么？",
        "15乘以6等于多少？",
        "什么是token？",
        "CPU和GPU有什么区别？",
        "太阳系有几大行星？",
    ]

    prompts = [args.prompt] if args.prompt else test_prompts
    stop_strings = ["\n", "；"]

    print(f"设备: {device} | 温度: {args.temperature} | top_k: {args.top_k}")
    print("=" * 50)
    for p in prompts:
        result = generate(
            model,
            tok,
            p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_strings=stop_strings,
            device=device,
            return_confidence=True,
        )
        conf = result["avg_confidence"] * 100
        print(f"Q: {p}")
        print(f"A: {result['text']}")
        print(f"置信度: {conf:.1f}%")
        print("-" * 50)


if __name__ == "__main__":
    main()
