"""checkpoint 指纹与 tokenizer 一致性校验的测试。"""

from core.infer import warn_tokenizer_mismatch
from core.utils import file_md5


class TestFileMd5:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        digest = file_md5(str(f))
        assert digest is not None
        assert digest == file_md5(str(f))  # 确定性
        assert len(digest) == 12

    def test_missing_file_returns_none(self):
        assert file_md5("/nonexistent/path") is None

    def test_none_path_returns_none(self):
        assert file_md5(None) is None


class TestWarnTokenizerMismatch:
    def _ckpt(self, fp, tok_path="train/tokenizer.json"):
        return {"cfg": {"tokenizer": {"path": tok_path}}, "fingerprint": fp}

    def test_no_fingerprint_silent(self, capsys):
        warn_tokenizer_mismatch({"cfg": {}})
        assert capsys.readouterr().out == ""

    def test_match_silent(self, capsys):
        ckpt = self._ckpt({"tokenizer_md5": file_md5("train/tokenizer.json")})
        warn_tokenizer_mismatch(ckpt)
        assert capsys.readouterr().out == ""

    def test_mismatch_warns(self, capsys):
        ckpt = self._ckpt({"tokenizer_md5": "000000000000"})
        warn_tokenizer_mismatch(ckpt)
        out = capsys.readouterr().out
        assert "词表已漂移" in out and "000000000000" in out

    def test_missing_tokenizer_file_warns(self, capsys):
        ckpt = self._ckpt({"tokenizer_md5": "abc123"}, tok_path="/nonexistent/tok.json")
        warn_tokenizer_mismatch(ckpt)
        assert "警告" in capsys.readouterr().out
