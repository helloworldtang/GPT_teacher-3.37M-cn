from src.tokenizer import ByteTokenizer, load_tokenizer


class TestByteTokenizerEncode:
    def test_ascii(self):
        tok = ByteTokenizer()
        assert tok.encode("hello") == [104, 101, 108, 108, 111]

    def test_chinese(self):
        tok = ByteTokenizer()
        ids = tok.encode("你好")
        assert len(ids) == 6  # 每个中文字符 3 个 UTF-8 字节

    def test_empty_string(self):
        tok = ByteTokenizer()
        assert tok.encode("") == []

    def test_special_tokens(self):
        tok = ByteTokenizer()
        ids = tok.encode("hi", add_special_tokens=True)
        assert ids[0] == 256  # bos
        assert ids[-1] == 257  # eos


class TestByteTokenizerDecode:
    def test_roundtrip_ascii(self):
        tok = ByteTokenizer()
        assert tok.decode(tok.encode("hello world")) == "hello world"

    def test_roundtrip_chinese(self):
        tok = ByteTokenizer()
        assert tok.decode(tok.encode("你好世界")) == "你好世界"

    def test_strips_special_tokens(self):
        tok = ByteTokenizer()
        assert tok.decode([256, 257, 258]) == ""


class TestByteTokenizerProperties:
    def test_vocab_size(self):
        tok = ByteTokenizer()
        assert tok.vocab_size == 259
        assert tok.bos_id == 256
        assert tok.eos_id == 257
        assert tok.pad_id == 258


class TestLoadTokenizer:
    def test_default_returns_byte(self):
        tok = load_tokenizer()
        assert isinstance(tok, ByteTokenizer)

    def test_bad_path_returns_byte(self):
        tok = load_tokenizer(kind="hf_tokenizers", path="/nonexistent/path")
        assert isinstance(tok, ByteTokenizer)
