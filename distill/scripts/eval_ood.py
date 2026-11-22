"""OOD 泛化评测：student 在训练从未见过的主题上答题，量泛化与塌缩。

指标：
- 多样性: 唯一答案数 / 总题数（第 9 篇基线为 1/15，即全部塌缩到同一句）
- 塌缩占比: 出现次数最多的答案占全部答案的比例
- 答对率(--judge): teacher 对 (问题, student 答案) 逐题判定是否正确

用法：
    uv run python distill/scripts/eval_ood.py
    uv run python distill/scripts/eval_ood.py --ckpt distill/checkpoints/best.pt --judge
"""

import argparse
import json
from collections import Counter

from core.infer import generate, load_model_and_tokenizer

OOD_PATH = "distill/data/ood_questions.json"
STOP_STRINGS = ["\n", "；", "！"]


def judge_with_teacher(questions: list[str], answers: list[str], teacher: str) -> list[bool]:
    """用 teacher 逐题判定 student 答案是否正确（0/1，MPS/CPU）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(teacher)
    model = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=torch.bfloat16)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    verdicts: list[bool] = []
    for q, a in zip(questions, answers, strict=True):
        messages = [
            {"role": "system", "content": "你是阅卷员。只输出“对”或“错”，不要任何解释。"},
            {"role": "user", "content": f"问题：{q}\n学生回答：{a}\n这个回答是否正确？"},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=4, temperature=0.0, do_sample=False, pad_token_id=tok.eos_token_id
            )
        gen_ids = output[0][inputs["input_ids"].shape[1] :]
        verdict: str = tok.decode(gen_ids, skip_special_tokens=True).strip()
        verdicts.append(verdict.startswith("对"))
    return verdicts


def main() -> None:
    """命令行入口：加载 student → OOD 答题 → 指标汇总（可选 teacher 判卷）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="distill/checkpoints/best.pt")
    ap.add_argument("--judge", action="store_true", help="加载 teacher 判卷（答对率指标）")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="distill/data/ood_results.json")
    args = ap.parse_args()

    with open(OOD_PATH, encoding="utf-8") as f:
        ood = json.load(f)["ood"]

    model, tok, _, device = load_model_and_tokenizer(args.ckpt, device="cpu")

    answers: list[str] = []
    for q in ood:
        answers.append(
            generate(
                model,
                tok,
                q,
                max_new_tokens=64,
                temperature=0.0,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.5,
                stop_strings=STOP_STRINGS,
                device=device,
            )
        )

    unique = len(set(answers))
    top, top_count = Counter(answers).most_common(1)[0]
    print(f"OOD {len(ood)} 题 | 唯一答案 {unique} | 塌缩占比 {top_count / len(ood):.0%}（“{top[:20]}…”）")

    verdicts: list[bool] | None = None
    if args.judge:
        verdicts = judge_with_teacher(ood, answers, args.teacher)
        acc = sum(verdicts) / len(verdicts)
        print(f"teacher 判卷答对率: {sum(verdicts)}/{len(verdicts)} ({acc:.0%})")

    print("=" * 50)
    for i, (q, a) in enumerate(zip(ood, answers, strict=True)):
        mark = "" if verdicts is None else ("✓ " if verdicts[i] else "✗ ")
        print(f"{mark}Q: {q}")
        print(f"  A: {a}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "questions": ood,
                "answers": answers,
                "unique": unique,
                "collapse_share": top_count / len(ood),
                "verdicts": verdicts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
