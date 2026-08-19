"""金样本回归套件：量化"本次改动改变了多少模型行为"。

三组用例（与生产路径同参：temperature=0 / top_k=50 / rep_penalty=1.5 / stop 同 web_demo）：
- single: 8 道单轮问题（Web Demo 快捷问题全集）
- multi:  4 例两轮对话（多轮拼接路径）
- ood:    15 道训练集外问题（幻觉行为监测）

用法：
    uv run python train/scripts/regression.py            # 与基线比对，输出行为变化报告
    uv run python train/scripts/regression.py --update   # 重建基线（行为变更是有意为之后使用）

基线文件 train/data/regression_baseline.json 记录每条用例输出的 md5；
比对只判"变没变"，不判对错——换 tokenizer/重训/改解码逻辑后跑一次，
即可拿到"多少条答案发生了变化"的量化数字。
"""

import argparse
import hashlib
import json
from typing import Any

from core.evaluate import load_model
from core.infer import generate
from train.web_demo import EXAMPLE_QUESTIONS, build_multi_turn_prompt

BASELINE_PATH = "train/data/regression_baseline.json"
STOP_STRINGS = ["用户:", "\n用户", "。", "；"]

# 训练集外问题（幻觉行为监测：正常应塌缩到高频先验，行为变化说明分布变了）
OOD_QUESTIONS = [
    "什么是量子计算？",
    "法国的首都是哪里？",
    "怎么做红烧肉？",
    "什么是区块链？",
    "地球为什么是圆的？",
    "光合作用是什么？",
    "什么是通货膨胀？",
    "如何学好英语？",
    "火星上有生命吗？",
    "什么是相对论？",
    "天为什么是蓝的？",
    "1 加 1 等于几？",
    "asdfghjkl",
    "What is deep learning?",
    "什么是新型冠状病毒？",
]

MULTI_TURN_CASES = [
    ("什么是注意力机制？", "RoPE 是什么？"),
    ("什么是机器学习？", "太阳系有哪些行星？"),
    ("你是谁？", "15 乘以 6 等于多少？"),
    ("什么是注意力机制？", "蒸馏水和纯水有什么区别？"),
]


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def run_all() -> dict[str, str]:
    """跑全部金样本，返回 {用例名: 输出 md5}。"""
    model, tok, device = load_model("train/checkpoints/best.pt")

    def gen(prompt: str) -> str:
        return generate(
            model,
            tok,
            prompt,
            max_new_tokens=128,
            temperature=0.0,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            stop_strings=STOP_STRINGS,
            device=device,
        )

    results: dict[str, str] = {}
    for q in EXAMPLE_QUESTIONS:
        results[f"single:{q}"] = _md5(gen(q))
    for q1, q2 in MULTI_TURN_CASES:
        a1 = gen(q1)
        results[f"multi:{q1}|{q2}"] = _md5(gen(build_multi_turn_prompt([(q1, a1)], q2)))
    for q in OOD_QUESTIONS:
        results[f"ood:{q}"] = _md5(gen(q))
    return results


def load_baseline() -> dict[str, Any] | None:
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            baseline: dict[str, Any] = json.load(f)
        return baseline
    except FileNotFoundError:
        return None


def main() -> None:
    """命令行入口：与基线比对或重建基线。"""
    ap = argparse.ArgumentParser(description="金样本回归套件")
    ap.add_argument("--update", action="store_true", help="重建基线（有意变更行为后使用）")
    args = ap.parse_args()

    print("跑金样本回归（single 8 + multi 4 + ood 15）...")
    results = run_all()

    if args.update:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"✓ 基线已重建: {len(results)} 条 → {BASELINE_PATH}")
        return

    baseline = load_baseline()
    if baseline is None:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"✓ 首次运行，已建立基线: {len(results)} 条 → {BASELINE_PATH}")
        print("  再次运行将输出行为变化报告。")
        return

    changed, added = [], []
    for name, h in results.items():
        if name not in baseline:
            added.append(name)
        elif baseline[name] != h:
            changed.append(name)

    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"行为变化报告: {len(changed)}/{total} 条输出改变, {len(added)} 条新增")
    if changed:
        print("改变的用例:")
        for name in changed:
            print(f"  ✗ {name}")
    if added:
        print("基线中不存在的用例（用 --update 纳入）:")
        for name in added:
            print(f"  + {name}")
    if not changed and not added:
        print("✓ 与基线完全一致，零行为变化")
    print(f"{'=' * 50}")
    raise SystemExit(1 if changed else 0)


if __name__ == "__main__":
    main()
