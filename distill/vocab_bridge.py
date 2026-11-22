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
    mode: str = "prefix",
) -> tuple[Tensor, Tensor]:
    """构建 teacher→student 词表配对。

    两种模式：
    - exact：解码字符串精确相等才配对（最保守，覆盖率低）。
    - prefix：teacher token 字符串取「词表内最长 student 前缀」配对——
      teacher 的多字 token 质量归给它的 student 前缀。事件 "文本以 t 开头"
      蕴含 "文本以 s 开头"，且每个 teacher token 只归一处，配对保持互斥，
      投影分布仍是合法概率（P(s) 是真值的下界）。

    Args:
        student_tok: student 分词器（TokenizerLike 接口）。
        teacher_tok: teacher 分词器（HF PreTrainedTokenizerFast）。
        student_vocab: student 词表大小。
        teacher_vocab: teacher 词表大小。
        mode: "exact" 或 "prefix"。

    Returns:
        (teacher_ids, student_ids) 两个等长 LongTensor：
        teacher_probs[..., teacher_ids] 沿最后一维取出后按 student_ids
        聚合即得投影分布。
    """
    # student 侧：词表小，逐 id decode 建字符串索引
    s_str_to_id: dict[str, int] = {}
    for sid in range(student_vocab):
        text = student_tok.decode([sid]).strip("\x00")
        if text and text not in s_str_to_id:
            s_str_to_id[text] = sid

    # teacher 侧：151K 逐 id decode（一次性，秒级）
    t_pairs_t: list[int] = []
    t_pairs_s: list[int] = []
    for tid in range(teacher_vocab):
        text = teacher_tok.decode([tid])
        if not text:
            continue
        if mode == "exact":
            matched = s_str_to_id.get(text)
            if matched is not None:
                t_pairs_t.append(tid)
                t_pairs_s.append(matched)
            continue
        # prefix：找词表内最长前缀
        matched = None
        for end in range(len(text), 0, -1):
            matched = s_str_to_id.get(text[:end])
            if matched is not None:
                break
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
    # 用 index_add_ 而非 scatter_add_：MPS 的 scatter_add_ 对展开索引有
    # "Placeholder storage has not been allocated" bug，index_add_ 的
    # 1-D 索引 + 2D 源形态在 MPS 上稳定。
    batch = teacher_probs.shape[:-1]
    flat = gathered.reshape(-1, gathered.size(-1))  # (N, K)
    proj_flat = torch.zeros(flat.size(0), student_vocab, dtype=teacher_probs.dtype)
    try:
        proj_flat.index_add_(1, pairs_s, flat)
    except RuntimeError:
        # MPS 后端在大词表 softmax 输出上偶发 "Placeholder storage has not
        # been allocated"（合成最小复现不出，真实 teacher 前向后必现）。
        # gather 后张量已小（N×K），CPU 执行 index_add 再搬回，代价毫秒级。
        proj_flat.index_add_(1, pairs_s.cpu(), flat.cpu())
        proj_flat = proj_flat.to(flat.device)
    proj = (proj_flat / coverage.reshape(-1, 1).clamp_min(1e-8)).reshape(*batch, student_vocab)
    return proj, coverage


def align_by_offsets(student_ends: list[int], teacher_ends: list[int], mode: str = "nearest") -> Tensor:
    """按字符偏移结尾对齐 student/teacher 位置。

    两种模式：
    - exact：仅接受结尾偏移恰好相等（对齐率低）。
    - nearest：取「不超过 student 结尾的最近 teacher 位置」——teacher 的
      上下文是 student 上下文的前缀，只是看得更短，不泄漏目标 token。

    Args:
        student_ends: student 各 token 的结尾字符偏移。
        teacher_ends: teacher 各 token 的结尾字符偏移。
        mode: "exact" 或 "nearest"。

    Returns:
        LongTensor [len(student_ends)]，元素为 teacher 位置或 -1。
    """
    if mode == "exact":
        t_pos = {end: j for j, end in enumerate(teacher_ends)}
        return torch.tensor([t_pos.get(e, -1) for e in student_ends], dtype=torch.long)

    import bisect

    out: list[int] = []
    for e in student_ends:
        j = bisect.bisect_right(teacher_ends, e) - 1
        out.append(j if j >= 0 else -1)
    return torch.tensor(out, dtype=torch.long)
