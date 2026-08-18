"""知识蒸馏训练。

支持两种模式：
1. 标准训练 — 用 core 模型 + 原始 byte tokenizer
2. logits 级蒸馏 — teacher/student 共享 tokenizer，KL 散度对齐输出分布

用法：
    # 标准训练（复用 train 模块）
    uv run python -m distill.train --config train/config.yml

    # logits 级蒸馏
    uv run python -m distill.train --kd --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from core.model import GPT
from core.utils import ensure_dir, set_seed

# ── 数据集 ──────────────────────────────────────────


class KDDataset(Dataset[tuple[Tensor, Tensor]]):
    """蒸馏训练数据集，用 teacher 的 tokenizer 编码。

    编码后截断到 seq_len，prompt 部分的 target 置 -100，
    不足 seq_len 时用 pad/-100 补齐。

    Attributes:
        samples: (输入 id 序列, 目标 id 序列) 列表。
    """

    def __init__(self, path: str, tok: PreTrainedTokenizerBase, seq_len: int, vocab_size: int) -> None:
        """加载 jsonl 并用 teacher tokenizer 编码。

        Args:
            path: 训练数据 jsonl 路径。
            tok: teacher 分词器。
            seq_len: 最大序列长度。
            vocab_size: 词表上限（超出 vocab 的 token 被过滤）。
        """
        self.samples: list[tuple[list[int], list[int]]] = []
        bos = tok.bos_token_id or tok.eos_token_id or 0
        eos = tok.eos_token_id or tok.sep_token_id or bos

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj["prompt"]
                completion = obj["completion"]
                text = prompt + completion
                ids = list(tok.encode(text))
                ids = [v for v in ids if v < vocab_size]
                ids = [bos] + ids + [eos]
                if len(ids) > seq_len:
                    ids = ids[:seq_len]

                prompt_ids = list(tok.encode(prompt))
                prompt_ids = [v for v in prompt_ids if v < vocab_size]
                prompt_ids = [bos] + prompt_ids
                prompt_len = min(len(prompt_ids), seq_len)

                # mask 掉 prompt 部分的 target，只对 completion 计算 loss
                mask_len = min(prompt_len - 1, len(ids) - 1)
                y = [-100] * mask_len + ids[mask_len + 1 :]
                y = y[: len(ids) - 1]

                x = ids[:-1][:seq_len]
                y = y[:seq_len]

                # 补齐
                pad_id = tok.pad_token_id or eos
                while len(x) < seq_len:
                    x.append(pad_id)
                    y.append(-100)

                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """把一批 (x, y) 张量对堆叠为 batch 张量。"""
    xs, ys = zip(*batch, strict=True)
    return torch.stack(xs), torch.stack(ys)


# ── 损失函数 ──────────────────────────────────────────


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    alpha: float,
    temperature: float,
) -> Tensor:
    """蒸馏损失 = α × CE(student, 硬标签) + (1-α) × KL(student, teacher) × T²。

    Args:
        student_logits: student 模型输出 logits。
        teacher_logits: teacher 模型输出 logits。
        targets: 硬标签（-100 忽略）。
        alpha: CE loss 权重（1-alpha 为 KL 权重）。
        temperature: 蒸馏温度。

    Returns:
        组合损失标量。
    """
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature**2)
    return alpha * ce + (1 - alpha) * kl


# ── 评估 ──────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader[Any], loss_fn: nn.Module, device: torch.device) -> float:
    """在验证集上计算平均 loss（结束后恢复训练态）。"""
    model.eval()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


# ── 主训练流程 ────────────────────────────────────────


def train_kd(args: argparse.Namespace) -> None:
    """Logits 级知识蒸馏训练：teacher/student 共享 tokenizer，KL 对齐输出分布。"""
    set_seed(42)

    # 加载 teacher
    print(f"加载 teacher: {args.teacher}")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher)
    teacher_model.eval()

    # 关键：用 model.config.vocab_size，不是 tokenizer.vocab_size
    vocab_size = teacher_model.config.vocab_size
    print(f"Teacher vocab: {vocab_size} (tokenizer 报告: {teacher_tok.vocab_size})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    teacher_model.to(device)
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Teacher: {teacher_params / 1e6:.1f}M 参数")

    # 构建 student
    model = GPT(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        dropout=0.1,
        use_flash=True,
        n_kv_head=args.n_kv_head,
    )
    model.to(device)
    student_params = sum(p.numel() for p in model.parameters())
    print(f"Student: {student_params / 1e6:.1f}M 参数, vocab={vocab_size}")

    # 数据
    train_ds = KDDataset(args.train_data, teacher_tok, args.seq_len, vocab_size)
    val_ds = KDDataset(args.val_data, teacher_tok, args.seq_len, vocab_size)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False, collate_fn=collate_fn)
    print(f"  数据: {len(train_ds)} 条训练, {len(val_ds)} 条验证")
    print(f"设备: {device} | α={args.alpha} | T={args.temperature}")
    print("=" * 50)

    optimizer = torch.optim.AdamW(  # type: ignore[attr-defined]
        model.parameters(), lr=args.lr, weight_decay=0.1
    )
    ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    ensure_dir(args.save_dir)

    best_val_loss = float("inf")
    patience = 0
    max_patience = args.patience
    step = 0
    accum = 0
    start = time.time()

    while step < args.max_steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # student forward
            student_logits, _ = model(xb)

            # teacher forward
            with torch.no_grad():
                teacher_out = teacher_model(xb)
                teacher_logits = teacher_out.logits

            loss = distillation_loss(
                student_logits,
                teacher_logits,
                yb,
                args.alpha,
                args.temperature,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            accum += 1

            if accum >= args.accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                accum = 0

                if step % 10 == 0:
                    elapsed = time.time() - start
                    print(f"step {step} loss {loss.item():.4f} elapsed {elapsed:.0f}s")

                if step % args.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, ce_fn, device)
                    elapsed = time.time() - start
                    print(f"  eval loss {val_loss:.4f} elapsed {elapsed:.0f}s")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "cfg": {
                                    "model": {
                                        "n_layer": args.n_layer,
                                        "n_head": args.n_head,
                                        "n_embd": args.n_embd,
                                        "n_kv_head": args.n_kv_head,
                                        "seq_len": args.seq_len,
                                        "dropout": 0.1,
                                    }
                                },
                                "vocab_size": vocab_size,
                            },
                            os.path.join(args.save_dir, "distill_best.pt"),
                        )
                        print(f"  → 新最佳 (step {step})")
                    else:
                        patience += 1
                        print(f"  → 未改善 ({patience}/{max_patience})")
                        if patience >= max_patience:
                            print(f"早停 (step {step})")
                            break

            if step >= args.max_steps:
                break

    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print(f"训练完成: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"最佳验证损失: {best_val_loss:.4f}")


def main() -> None:
    """命令行入口：--kd 走蒸馏训练，否则委托给 train 模块标准训练。"""
    ap = argparse.ArgumentParser(description="知识蒸馏训练")
    ap.add_argument("--kd", action="store_true", help="启用 logits 级蒸馏模式")
    ap.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--train_data", default="train/data/train.jsonl")
    ap.add_argument("--val_data", default="train/data/val.jsonl")
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--n_kv_head", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--accum_steps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5, help="CE loss 权重 (1-alpha 为 KL 权重)")
    ap.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    ap.add_argument("--patience", type=int, default=15, help="早停 patience")
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--save_dir", default="distill/checkpoints")
    args = ap.parse_args()

    if args.kd:
        train_kd(args)
    else:
        from train.train import train

        train(config_path="train/config.yml")


if __name__ == "__main__":
    main()
