import json
import torch
from src.data import InstructDataset, collate
from src.tokenizer import ByteTokenizer


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestInstructDataset:
    def test_length(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [
            {"prompt": "问题1", "completion": "答案1"},
            {"prompt": "问题2", "completion": "答案2"},
        ])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=128)
        assert len(ds) == 2

    def test_prefix_has_bos(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [{"prompt": "你好", "completion": "世界"}])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=128)
        ids, tar = ds[0]
        assert ids[0] == 256  # bos_id

    def test_target_prefix_masked(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [{"prompt": "你好", "completion": "世界"}])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=128)
        ids, tar = ds[0]
        # 前缀部分 target 应为 -100
        prefix_len = len([256] + tok.encode("用户:你好\n助手:", add_special_tokens=False))
        assert all(t == -100 for t in tar[:prefix_len - 1])

    def test_truncation(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [{"prompt": "很长的内容" * 20, "completion": "答案" * 20}])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=16)
        ids, tar = ds[0]
        assert len(ids) <= 16
        assert len(tar) <= 16


class TestCollate:
    def test_padding_shape(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [
            {"prompt": "短", "completion": "答"},
            {"prompt": "比较长的问题", "completion": "比较长的答案"},
        ])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=64)
        batch = [ds[0], ds[1]]
        x, y = collate(batch, seq_len=64, pad_id=258)
        assert x.shape == (2, 64)
        assert y.shape == (2, 64)

    def test_padding_values(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [{"prompt": "短", "completion": "答"}])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=64)
        x, y = collate([ds[0]], seq_len=64, pad_id=258)
        # 末尾应该是 pad_id
        assert x[0, -1] == 258
        assert y[0, -1] == -100

    def test_output_dtype(self, tmp_path):
        f = tmp_path / "test.jsonl"
        _write_jsonl(str(f), [{"prompt": "q", "completion": "a"}])
        tok = ByteTokenizer()
        ds = InstructDataset(str(f), tok, seq_len=32)
        x, y = collate([ds[0]], seq_len=32, pad_id=258)
        assert x.dtype == torch.long
        assert y.dtype == torch.long
