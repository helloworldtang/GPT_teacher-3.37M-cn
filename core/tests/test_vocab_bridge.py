"""vocab_bridge 词表桥接测试。

合成数据锁定投影/对齐的数学性质（无网络依赖）；真实 tokenizer 的集成
sanity 仅在本机有 Qwen 缓存时运行（CI 无缓存自动跳过）。
"""

import pytest
import torch

from distill.vocab_bridge import align_by_offsets, build_exact_pairs, project_distribution


class FakeTok:
    """只实现 decode 的假分词器，测纯逻辑（其余协议属性给空值）。"""

    vocab_size: int = 0
    bos_id: int | None = None
    eos_id: int | None = None
    pad_id: int | None = None
    unk_id: int | None = None

    def __init__(self, id2str: dict[int, str]) -> None:
        self.id2str = id2str

    def decode(self, ids: list[int]) -> str:
        return self.id2str.get(ids[0], "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return []  # 纯逻辑测试用不到 encode


def test_build_exact_pairs_filters_unmatched() -> None:
    student = FakeTok({0: "你", 1: "好", 2: "机器"})
    teacher = FakeTok({5: "你", 7: "好", 9: "世", 11: "的"})
    pairs_t, pairs_s = build_exact_pairs(student, teacher, student_vocab=3, teacher_vocab=12, mode="exact")
    assert pairs_t.tolist() == [5, 7]
    assert pairs_s.tolist() == [0, 1]


def test_prefix_mode_longest_prefix_wins() -> None:
    # teacher "ABC" 的词表内最长前缀是 "AB"；"X" 无任何前缀在词表中
    student = FakeTok({0: "A", 1: "AB", 2: "B"})
    teacher = FakeTok({5: "ABC", 7: "B", 9: "X"})
    pairs_t, pairs_s = build_exact_pairs(student, teacher, student_vocab=3, teacher_vocab=10)
    assert pairs_t.tolist() == [5, 7]
    assert pairs_s.tolist() == [1, 2]


def test_project_renormalizes_and_reports_coverage() -> None:
    # teacher: id5=0.5(你) id7=0.3(好) id9=0.2(世)；student 只有 你/好
    pairs_t = torch.tensor([5, 7])
    pairs_s = torch.tensor([0, 1])
    probs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2])
    proj, coverage = project_distribution(probs, pairs_t, pairs_s, student_vocab=3)
    assert pytest.approx(coverage.item()) == 0.8
    # 匹配质量 0.8 重归一到 1.0：你=0.625 好=0.375 机器=0
    assert pytest.approx(proj.tolist(), abs=1e-6) == [0.625, 0.375, 0.0]


def test_project_batch_shape() -> None:
    pairs_t = torch.tensor([0, 1])
    pairs_s = torch.tensor([0, 1])
    probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    proj, coverage = project_distribution(probs, pairs_t, pairs_s, student_vocab=2)
    assert proj.shape == (2, 3, 2)
    assert coverage.shape == (2, 3)
    assert torch.allclose(proj.sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_duplicate_teacher_strings_aggregate() -> None:
    # teacher 两个 id 解码同字符串：质量应聚合到同一 student id
    student = FakeTok({0: "你"})
    teacher = FakeTok({3: "你", 8: "你"})
    pairs_t, pairs_s = build_exact_pairs(student, teacher, student_vocab=1, teacher_vocab=9)
    probs = torch.tensor([0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.6])
    proj, coverage = project_distribution(probs, pairs_t, pairs_s, student_vocab=1)
    assert coverage.item() == 1.0
    assert proj[0].item() == 1.0


def test_align_by_offsets_exact_match_only() -> None:
    mapping = align_by_offsets(student_ends=[1, 2, 4], teacher_ends=[2, 3, 4], mode="exact")
    assert mapping.tolist() == [-1, 0, 2]


def test_align_by_offsets_nearest_no_future_leak() -> None:
    # nearest：取不超过 student 结尾的最近 teacher 位置；student 1 前无 teacher 边界仍为 -1
    mapping = align_by_offsets(student_ends=[1, 3, 4], teacher_ends=[2], mode="nearest")
    assert mapping.tolist() == [-1, 0, 0]


def test_align_by_offsets_no_match() -> None:
    mapping = align_by_offsets(student_ends=[1, 3], teacher_ends=[2], mode="exact")
    assert mapping.tolist() == [-1, -1]


def test_integration_qwen_projection() -> None:
    """真实 tokenizer sanity：teacher 单点分布经投影后落在同字符串 student token 上。"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers 不可用")

    try:
        teacher = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("本机无 Qwen tokenizer 缓存")

    import json
    import os
    import tempfile

    from tokenizers import Tokenizer

    from core.build_tokenizer import build
    from core.tokenizer import HfTokenizer

    # 迷你 student tokenizer：用仓库自带主题语料子集训练
    from distill.data import EXTRA_GENERAL_TOPICS, EXTRA_ML_TOPICS, ORIGINAL_TOPICS

    rows = ORIGINAL_TOPICS + EXTRA_ML_TOPICS + EXTRA_GENERAL_TOPICS
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "train.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            for q, a in rows:
                f.write(json.dumps({"prompt": q, "completion": a}, ensure_ascii=False) + "\n")
        tok_path = os.path.join(tmp, "tokenizer.json")
        build(corpus, tok_path, vocab_size=1024)

        student = HfTokenizer(Tokenizer.from_file(tok_path))
        pairs_t, pairs_s = build_exact_pairs(student, teacher, student.vocab_size, teacher.vocab_size)

        # 词表重叠必须显著存在，否则投影路线不成立
        assert pairs_t.numel() > student.vocab_size // 3

        # teacher 在"的"上放一个尖峰分布，投影后 argmax 应解码为"的"
        tid = teacher.encode("的", add_special_tokens=False)[0]
        probs = torch.full((teacher.vocab_size,), 1e-9)
        probs[tid] = 1.0
        proj, coverage = project_distribution(probs, pairs_t, pairs_s, student.vocab_size)
        assert coverage.item() > 0.99
        best = int(proj.argmax())
        assert student.decode([best]) == "的"
