"""词表桥接：teacher（Qwen 151K）输出分布聚合投影到 student（自建 BPE）词表。

teacher/student 词表不同时的 logits 级蒸馏需要两件桥：

1. **词表投影** — 双方都是 byte-level BPE，token 字符串经各自 byte 解码后落在
   同一个字符串空间。按"解码字符串精确相等"建配对，teacher 分布 scatter_add
   聚合到 student 词表维度，未匹配质量丢弃后重归一（coverage 指标监控丢弃比例）。
   这是最保守的对齐：多字 student token 若不在 teacher 词表中则拿不到监督信号，
   但绝不引入错位信号。
2. **位置对齐** — 两个 tokenizer 对同一段文本的切分边界不同。KD 只在
   "student 边界恰有 teacher 边界重合（字符偏移相等）"的位置计算 KL，
   其余位置只算 CE。宁缺毋滥：teacher 多看一个字符等于把答案漏给损失函数。

用法见 core/tests/test_vocab_bridge.py（合成数据锁定数学性质）与
distill/train.py 的 KD 训练路径。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from core.tokenizer import TokenizerLike


def build_exact_pairs(
    student_tok: TokenizerLike,
    teacher_tok: Any,
    student_vocab: int,
    teacher_vocab: int,
) -> tuple[Tensor, Tensor]:
    """按解码字符串精确相等构建 teacher→student 词表配对。

    Args:
        student_tok: student 分词器（TokenizerLike 接口）。
        teacher_tok: teacher 分词器（HF PreTrainedTokenizerFast）。
        student_vocab: student 词表大小。
        teacher_vocab: teacher 词表大小。

    Returns:
        (teacher_ids, student_ids) 两个等长 LongTensor：
        teacher_probs[..., teacher_ids] 沿最后一维取出后按 student_ids
        scatter_add 即得投影分布。
    """
    # student 侧：词表小，逐 id decode 建字符串索引
    s_str_to_id: dict[str, int] = {}
    for sid in range(student_vocab):
        text = student_tok.decode([sid]).strip("\x00")
        if text and text not in s_str_to_id:
            s_str_to_id[text] = sid

    # teacher 侧：151K 逐 id decode（一次性，秒级），建字符串 → id 列表
    t_pairs_t: list[int] = []
    t_pairs_s: list[int] = []
    for tid in range(teacher_vocab):
        text = teacher_tok.decode([tid])
        if not text:
            continue
        matched = s_str_to_id.get(text)
        if matched is not None:
            t_pairs_t.append(tid)
            t_pairs_s.append(matched)

    return torch.tensor(t_pairs_t, dtype=torch.long), torch.tensor(t_pairs_s, dtype=torch.long)


def project_distribution(
    teacher_probs: Tensor,
    pairs_t: Tensor,
    pairs_s: Tensor,
    student_vocab: int,
) -> tuple[Tensor, Tensor]:
    """把 teacher 概率分布聚合投影到 student 词表维度。

    Args:
        teacher_probs: 任意形状（..., V_teacher）的概率分布（已 softmax）。
        pairs_t / pairs_s: build_exact_pairs 的配对。
        student_vocab: student 词表大小。

    Returns:
        (投影分布 (..., V_student)，已对匹配质量重归一；
         coverage (...,)：匹配质量占 teacher 总质量的比例）。
    """
    gathered = teacher_probs[..., pairs_t]  # (..., K)
    coverage = gathered.sum(dim=-1)  # teacher 概率和为 1，匹配量即占比
    proj = torch.zeros(*teacher_probs.shape[:-1], student_vocab, dtype=teacher_probs.dtype)
    proj.scatter_add_(-1, pairs_s.expand(*teacher_probs.shape[:-1], pairs_s.size(0)), gathered)
    proj = proj / coverage.clamp_min(1e-8).unsqueeze(-1)
    return proj, coverage


def align_by_offsets(student_ends: list[int], teacher_ends: list[int]) -> Tensor:
    """按字符偏移结尾对齐 student/teacher 位置。

    对每个 student 位置 i（上下文截至 student_ends[i]），找 teacher 中
    结尾偏移恰好相等的位置；找不到则为 -1（该位置不参与 KL）。
    两侧 ends 均须严格递增（byte-level 切分的自然性质）。

    Args:
        student_ends: student 各 token 的结尾字符偏移。
        teacher_ends: teacher 各 token 的结尾字符偏移。

    Returns:
        LongTensor [len(student_ends)]，元素为 teacher 位置或 -1。
    """
    t_pos = {end: j for j, end in enumerate(teacher_ends)}
    return torch.tensor([t_pos.get(e, -1) for e in student_ends], dtype=torch.long)
