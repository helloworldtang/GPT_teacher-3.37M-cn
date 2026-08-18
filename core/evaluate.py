"""验收测试：读取 test.jsonl，跑推理对比期望答案，输出 pass/fail 汇总。"""

import argparse
import json
import os
from typing import Any

import torch

from core.infer import generate
from core.model import GPT
from core.tokenizer import TokenizerLike, load_tokenizer


def load_model(ckpt_path: str = "train/checkpoints/best.pt") -> tuple[GPT, TokenizerLike, torch.device]:
    """从 checkpoint 恢复模型与分词器，自动选择可用设备。

    Args:
        ckpt_path: checkpoint 路径。

    Returns:
        (模型, 分词器, 设备)，模型已 eval 并移至设备。

    Raises:
        SystemExit: checkpoint 不存在。
    """
    if not os.path.exists(ckpt_path):
        print(f"错误：未找到模型文件 {ckpt_path}")
        print("请先运行训练：uv run python -m train.train")
        raise SystemExit(1)
    checkpoint: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = checkpoint["cfg"]
    tok = load_tokenizer(
        cfg.get("tokenizer", {}).get("type", "byte"),
        cfg.get("tokenizer", {}).get("path"),
    )
    model = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=0.0,
        n_kv_head=cfg["model"].get("n_kv_head"),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    return model, tok, device


def run_test(
    model: GPT, tok: TokenizerLike, device: torch.device, test_path: str = "train/data/test.jsonl"
) -> tuple[int, int]:
    """跑测试集并打印 pass/fail 汇总。

    判定规则：从期望答案提取 2 字片段，多数（>=1/2）出现在输出中即 PASS。

    Args:
        model: GPT 模型。
        tok: 分词器。
        device: 推理设备。
        test_path: test.jsonl 路径。

    Returns:
        (通过数, 总题数)。
    """
    with open(test_path, encoding="utf-8") as f:
        tests = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'=' * 50}")
    print(f"验收测试：共 {len(tests)} 题")
    print(f"{'=' * 50}")

    passed = 0
    for i, item in enumerate(tests, 1):
        prompt = item["prompt"]
        expected = item["completion"]

        response = generate(
            model,
            tok,
            prompt,
            max_new_tokens=40,
            temperature=0.0,
            repetition_penalty=2.0,
            stop_strings=["用户:", "\n用户", "。", "；"],
            device=device,
        )

        # 关键词匹配：从期望答案提取 2-4 字片段，检查是否多数出现在输出中
        clean_exp = expected.replace("。", "").replace("，", "").replace("？", "").replace("、", "").strip()
        clean_resp = response.replace("。", "").replace("，", "").replace("？", "").replace("、", "").strip()
        chunks = [clean_exp[j : j + 2] for j in range(0, len(clean_exp), 2) if len(clean_exp[j : j + 2]) >= 2]
        if not chunks:
            chunks = [clean_exp] if clean_exp else []
        matched_count = sum(1 for c in chunks if c in clean_resp)
        matched = len(chunks) > 0 and matched_count >= max(1, len(chunks) // 2)

        status = "PASS" if matched else "FAIL"
        if matched:
            passed += 1

        print(f"\n[{status}] Q{i}: {prompt}")
        print(f"  期望: {expected}")
        print(f"  实际: {response}")

    total = len(tests)
    rate = passed / total * 100 if total > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"结果: {passed}/{total} 通过 ({rate:.0f}%)")
    if rate >= 80:
        print("验收通过！模型已学会回答测试集中的大部分问题。")
    elif rate >= 50:
        print("部分通过。可以尝试增加训练步数或扩充数据来提升效果。")
    else:
        print("验收未通过。建议检查数据质量、增加训练步数或调整配置。")
    print(f"{'=' * 50}")
    return passed, total


def main() -> None:
    """命令行验收入口。"""
    ap = argparse.ArgumentParser(description="验收测试")
    ap.add_argument("--ckpt", default="train/checkpoints/best.pt", help="模型路径")
    ap.add_argument("--test", default="train/data/test.jsonl", help="测试集路径")
    args = ap.parse_args()

    model, tok, device = load_model(args.ckpt)
    run_test(model, tok, device, args.test)


if __name__ == "__main__":
    main()
