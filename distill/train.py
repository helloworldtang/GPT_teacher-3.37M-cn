"""知识蒸馏训练。

支持三种模式：
1. 标准训练 — 用 core 模块 + config 指定的 tokenizer/数据
2. logits 级蒸馏（共享词表） — teacher/student 同 tokenizer，KL 对齐输出分布
3. logits 级蒸馏（词表桥接） — student 自建中等词表，teacher 分布经
   vocab_bridge 精确匹配投影 + 字符偏移位置对齐后做 KL

用法：
    # 标准训练（复用 train 模块，如路径 A 的 SFT）
    uv run python -m distill.train --config distill/config.yml

    # logits 级蒸馏（共享词表）
    uv run python -m distill.train --kd --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

    # logits 级蒸馏（词表桥接，路径 B）
    uv run python -m distill.train --kd --bridge --teacher Qwen/Qwen2.5-1.5B-Instruct \
        --student_tokenizer distill/tokenizer.json --train_data distill/data/train_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, TypedDict

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
    # CE 项带 ignore_index：全 batch 均为 -100 时返回 nan，
    # 权重为 0 也不该让 nan 传染（0 × nan = nan），故 alpha=0 时跳过计算。
    if alpha > 0:
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        ce_term = alpha * ce
    else:
        ce_term = student_logits.new_zeros(())
    # KL 只在 targets != -100 的位置计算：prompt/pad 位置 teacher 的条件分布
    # 与"下一个 completion token"无关，对齐它们是噪声信号。
    valid = targets.view(-1) != -100
    if valid.any():
        s = student_logits.reshape(-1, student_logits.size(-1))[valid]
        t = teacher_logits.reshape(-1, teacher_logits.size(-1))[valid]
        student_soft = F.log_softmax(s / temperature, dim=-1)
        teacher_soft = F.softmax(t / temperature, dim=-1)
        # batchmean 在 [N_valid, V] 上即"每个有效 token 的平均 KL"
        kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature**2)
    else:
        kl = student_logits.new_zeros(())
    return ce_term + (1 - alpha) * kl


# ── 路径 B：词表桥接蒸馏 ──────────────────────────────


class KDDualSample(TypedDict):
    """双编码样本的字段。"""

    s_x: list[int]  # student 输入序列（定长 seq_len，pad 补齐）
    s_y: list[int]  # student 目标序列（prompt/pad 位置 -100）
    align: list[int]  # 逐 student 位置对齐的 teacher 位置（无对齐 -1）
    t_ids: list[int]  # teacher 输入序列（变长）


class KDDualDataset(Dataset[KDDualSample]):
    """双编码蒸馏数据集：同一文本分别用 student/teacher tokenizer 编码。

    KL 监督只在「student 位置 i 的字符结尾偏移与某 teacher 位置 j 精确相等」
    处可用（align[i] = j，否则 -1）；CE 仍覆盖全部 completion 位置。

    Attributes:
        samples: 双编码样本列表。
    """

    def __init__(
        self,
        path: str,
        student_tok: Any,  # tokenizers.Tokenizer
        teacher_tok: PreTrainedTokenizerBase,
        seq_len: int,
    ) -> None:
        """加载 jsonl 并双编码。

        Args:
            path: 训练数据 jsonl 路径（prompt/completion 对）。
            student_tok: student 侧 tokenizers.Tokenizer（需 offsets）。
            teacher_tok: teacher 侧 HF fast tokenizer。
            seq_len: student 序列长度上限。
        """
        self.samples: list[KDDualSample] = []
        bos = student_tok.token_to_id("<bos>")
        eos = student_tok.token_to_id("<eos>")
        pad = student_tok.token_to_id("<pad>")
        assert bos is not None and eos is not None and pad is not None, "student tokenizer 缺少特殊 token"

        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj["prompt"]
                completion = obj["completion"]
                text = prompt + completion

                enc_s = student_tok.encode(text, add_special_tokens=False)
                s_body = enc_s.ids[: seq_len - 2]
                s_offsets = enc_s.offsets[: seq_len - 2]
                s_ids = [bos, *s_body, eos]
                # student 输入位置 i 的上下文结尾偏移（BOS 结尾记 0）
                s_ends = [0, *(end for _, end in s_offsets), len(text)][: len(s_ids)]

                # teacher 侧编码（带字符偏移）
                t_enc = teacher_tok(text, add_special_tokens=False, return_offsets_mapping=True)
                t_ids = t_enc["input_ids"]
                t_ends = [end for _, end in t_enc["offset_mapping"]]
                t_pos = {end: j for j, end in enumerate(t_ends)}

                # CE mask：目标 token 属于 completion（起始偏移越过 prompt）
                starts = [0, *(st for st, _ in s_offsets), len(text)][: len(s_ids)]
                y: list[int] = []
                for i in range(len(s_ids) - 1):
                    y.append(s_ids[i + 1] if starts[i + 1] >= len(prompt) else -100)

                s_x = s_ids[: len(y)]
                align = [t_pos.get(e, -1) for e in s_ends[: len(y)]]
                while len(s_x) < seq_len:
                    s_x.append(pad)
                    y.append(-100)
                    align.append(-1)

                self.samples.append(KDDualSample(s_x=s_x, s_y=y, align=align, t_ids=list(t_ids)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> KDDualSample:
        return self.samples[idx]


def collate_dual(
    batch: list[KDDualSample],
    teacher_pad: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """堆叠双编码 batch：student 三序列定长，teacher 序列 pad 到批内最长。"""
    bs = len(batch)
    t_max = max(len(b["t_ids"]) for b in batch)

    s_x = torch.tensor([b["s_x"] for b in batch], dtype=torch.long)
    s_y = torch.tensor([b["s_y"] for b in batch], dtype=torch.long)
    align = torch.tensor([b["align"] for b in batch], dtype=torch.long)
    t_ids = torch.full((bs, t_max), teacher_pad, dtype=torch.long)
    for i, b in enumerate(batch):
        t_ids[i, : len(b["t_ids"])] = torch.tensor(b["t_ids"], dtype=torch.long)
    return s_x, s_y, align, t_ids


def bridge_distillation_loss(
    student_logits: Tensor,
    teacher_proj_log: Tensor,
    targets: Tensor,
    aligned_mask: Tensor,
    alpha: float,
    temperature: float,
) -> Tensor:
    """词表桥接蒸馏损失：CE（全部 completion 位置）+ KL（仅对齐位置）。

    Args:
        student_logits: student 输出 logits [B, S, V_s]。
        teacher_proj_log: 已投影到 student 词表并对齐 gather 过的 teacher
            log 分布 [B, S, V_s]（未对齐位置应为全零，配合 aligned_mask）。
        targets: student 硬标签（-100 忽略）。
        aligned_mask: [B, S] bool，位置既有 CE 有效又与 teacher 对齐。
        alpha: CE 权重。
        temperature: 蒸馏温度。

    Returns:
        组合损失标量。
    """
    if alpha > 0:
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        ce_term = alpha * ce
    else:
        ce_term = student_logits.new_zeros(())

    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    kl_point = (teacher_proj_log.exp() * (teacher_proj_log - s_log)).sum(dim=-1)  # [B, S]
    kl = (kl_point * aligned_mask).sum() / aligned_mask.sum() if aligned_mask.any() else student_logits.new_zeros(())
    return ce_term + (1 - alpha) * kl * (temperature**2)


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
    early_stop_triggered = False

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
                                "teacher": args.teacher,
                            },
                            os.path.join(args.save_dir, "distill_best.pt"),
                        )
                        print(f"  → 新最佳 (step {step})")
                    else:
                        patience += 1
                        print(f"  → 未改善 ({patience}/{max_patience})")
                        if patience >= max_patience:
                            print(f"早停 (step {step})")
                            early_stop_triggered = True
                            break

            if step >= args.max_steps or early_stop_triggered:
                break

    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print(f"训练完成: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"最佳验证损失: {best_val_loss:.4f}")


def train_kd_bridge(args: argparse.Namespace) -> None:
    """词表桥接蒸馏：student 自建词表 + teacher 分布投影对齐（路径 B）。"""
    from tokenizers import Tokenizer

    from core.tokenizer import HfTokenizer
    from distill.vocab_bridge import build_exact_pairs, project_distribution

    set_seed(42)

    # student 侧：自建中等词表
    student_hf = HfTokenizer(Tokenizer.from_file(args.student_tokenizer))
    vocab_size = student_hf.vocab_size
    student_raw = Tokenizer.from_file(args.student_tokenizer)

    # teacher 侧
    print(f"加载 teacher: {args.teacher}")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16)
    teacher_model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    teacher_model.to(device)
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 词表桥接（一次性构建）
    pairs_t, pairs_s = build_exact_pairs(student_hf, teacher_tok, vocab_size, teacher_tok.vocab_size)
    print(f"词表桥接: {pairs_t.numel()} 对精确匹配 / student 词表 {vocab_size}")
    teacher_pad = teacher_tok.pad_token_id or teacher_tok.eos_token_id or 0

    # student 模型（结构与主线一致，保证与路径 A 可比）
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
    print(f"Student: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数, vocab={vocab_size}")

    train_ds = KDDualDataset(args.train_data, student_raw, teacher_tok, args.seq_len)
    val_ds = KDDualDataset(args.val_data, student_raw, teacher_tok, args.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch,
        shuffle=True,
        collate_fn=lambda b: collate_dual(b, teacher_pad),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_batch,
        shuffle=False,
        collate_fn=lambda b: collate_dual(b, teacher_pad),
    )
    print(f"数据: {len(train_ds)} 训练 / {len(val_ds)} 验证 | α={args.alpha} | T={args.temperature}")
    print("=" * 50)

    optimizer = torch.optim.AdamW(  # type: ignore[attr-defined]
        model.parameters(), lr=args.lr, weight_decay=0.1
    )
    ensure_dir(args.save_dir)

    best_val_loss = float("inf")
    patience = 0
    step = 0
    accum = 0
    start = time.time()
    align_rate_smoothed = 0.0
    coverage_smoothed = 0.0
    early_stop = False

    while step < args.max_steps and not early_stop:
        for s_x, s_y, align, t_ids in train_loader:
            s_x, s_y, align, t_ids = s_x.to(device), s_y.to(device), align.to(device), t_ids.to(device)

            student_logits, _ = model(s_x)

            with torch.no_grad():
                t_logits = teacher_model(t_ids).logits.float()
                t_probs = F.softmax(t_logits / args.temperature, dim=-1)
                proj, coverage = project_distribution(t_probs, pairs_t.to(device), pairs_s.to(device), vocab_size)
                # 把 teacher 位置分布 gather 到 student 位置（未对齐位置清零）
                idx = align.clamp_min(0).unsqueeze(-1).expand(-1, -1, vocab_size)
                proj_at = proj.gather(1, idx) * (align >= 0).unsqueeze(-1)
                teacher_log = proj_at.clamp_min(1e-9).log()

            aligned_mask = (align >= 0) & (s_y != -100)
            loss = bridge_distillation_loss(
                student_logits, teacher_log, s_y, aligned_mask, args.alpha, args.temperature
            )

            # 对齐率/词表覆盖率的滑动诊断（走通路径的关键观测）
            align_rate_smoothed = 0.98 * align_rate_smoothed + 0.02 * aligned_mask.float().mean().item()
            ce_mask = s_y != -100
            cov_at = coverage.gather(1, align.clamp_min(0)) * (align >= 0).float()
            cov_mean = (cov_at * ce_mask.float()).sum() / ce_mask.float().sum().clamp_min(1)
            coverage_smoothed = 0.98 * coverage_smoothed + 0.02 * cov_mean.item()

            loss.backward()  # type: ignore[no-untyped-call]
            accum += 1
            if accum >= args.accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                accum = 0

                if step % 10 == 0:
                    print(
                        f"step {step} loss {loss.item():.4f} "
                        f"align {align_rate_smoothed:.0%} cov {coverage_smoothed:.0%} "
                        f"elapsed {time.time() - start:.0f}s"
                    )

                if step % args.eval_interval == 0:
                    model.eval()
                    total, count = 0.0, 0
                    with torch.no_grad():
                        for s_xv, s_yv, _alignv, _t_idsv in val_loader:
                            s_xv, s_yv = s_xv.to(device), s_yv.to(device)
                            logits, _ = model(s_xv)
                            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), s_yv.view(-1), ignore_index=-100)
                            total += ce.item()
                            count += 1
                    model.train()
                    val_loss = total / max(1, count)
                    print(f"  eval ce {val_loss:.4f} (align {align_rate_smoothed:.0%} cov {coverage_smoothed:.0%})")

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
                                    },
                                    "tokenizer": {"path": args.student_tokenizer, "type": "hf_tokenizers"},
                                },
                                "vocab_size": vocab_size,
                                "teacher": args.teacher,
                                "mode": "kd_bridge",
                            },
                            os.path.join(args.save_dir, "distill_bridge_best.pt"),
                        )
                        print(f"  → 新最佳 (step {step})")
                    else:
                        patience += 1
                        print(f"  → 未改善 ({patience}/{args.patience})")
                        if patience >= args.patience:
                            print(f"早停 (step {step})")
                            early_stop = True
                            break

            if step >= args.max_steps or early_stop:
                break

    print(f"\n{'=' * 50}")
    print(f"训练完成: {(time.time() - start) / 60:.1f}min | 最佳验证 CE: {best_val_loss:.4f}")
    print(f"诊断: 对齐率 {align_rate_smoothed:.0%} | 词表覆盖率 {coverage_smoothed:.0%}")


def main() -> None:
    """命令行入口：--kd --bridge 走词表桥接蒸馏，--kd 走共享词表蒸馏，否则标准训练。"""
    ap = argparse.ArgumentParser(description="知识蒸馏训练")
    ap.add_argument("--kd", action="store_true", help="启用 logits 级蒸馏模式")
    ap.add_argument("--bridge", action="store_true", help="词表桥接模式（student 自建词表）")
    ap.add_argument(
        "--student_tokenizer", default="distill/tokenizer.json", help="bridge 模式的 student tokenizer 路径"
    )
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

    if args.kd and args.bridge:
        train_kd_bridge(args)
    elif args.kd:
        train_kd(args)
    else:
        from train.train import train

        train(config_path="train/config.yml")


if __name__ == "__main__":
    main()
