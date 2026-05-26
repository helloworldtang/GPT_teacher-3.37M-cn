import torch
from src.infer import _trim_leading_punct, _is_punct_token, generate
from src.tokenizer import ByteTokenizer


class TestTrimLeadingPunct:
    def test_spaces(self):
        assert _trim_leading_punct("  hello") == "hello"

    def test_chinese_punct(self):
        assert _trim_leading_punct("，。！回答") == "回答"

    def test_no_punct(self):
        assert _trim_leading_punct("answer") == "answer"

    def test_empty(self):
        assert _trim_leading_punct("") == ""

    def test_all_punct(self):
        assert _trim_leading_punct("。。。") == ""


class TestIsPunctToken:
    def test_punct_returns_true(self):
        tok = ByteTokenizer()
        # ASCII 标点是单字节，可以直接判断
        assert _is_punct_token(tok, ord(",")) is True
        assert _is_punct_token(tok, ord("!")) is True
        assert _is_punct_token(tok, ord("?")) is True

    def test_letter_returns_false(self):
        tok = ByteTokenizer()
        assert _is_punct_token(tok, ord("a")) is False


class TestGenerate:
    def test_returns_string(self, small_model):
        tok = ByteTokenizer()
        result = generate(small_model, tok, "测试", max_new_tokens=5, temperature=0.0, device="cpu")
        assert isinstance(result, str)

    def test_return_confidence(self, small_model):
        tok = ByteTokenizer()
        result = generate(small_model, tok, "测试", max_new_tokens=5, temperature=0.0, device="cpu", return_confidence=True)
        assert isinstance(result, dict)
        assert "text" in result
        assert "avg_confidence" in result
        assert "token_probs" in result
