"""分词器：byte-level 与 HuggingFace tokenizers（BPE）两种实现。"""

import os
from typing import Protocol

from tokenizers import Tokenizer


class TokenizerLike(Protocol):
    """分词器统一接口：编码/解码与特殊 token id。"""

    vocab_size: int
    bos_id: int | None
    eos_id: int | None
    pad_id: int | None
    unk_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Byte-level 分词器：每个 UTF-8 字节一个 token，词表 259。

    每个汉字占 3 个 token，信息密度低，但零依赖、绝对可逆。
    """

    def __init__(self) -> None:
        self.vocab_size: int = 259
        self.bos_id: int | None = 256
        self.eos_id: int | None = 257
        self.pad_id: int | None = 258
        self.unk_id: int | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """把文本编码为 byte-level token id 列表。

        Args:
            text: 待编码文本。
            add_special_tokens: 是否在首尾添加 BOS/EOS。

        Returns:
            token id 列表。
        """
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            assert self.bos_id is not None and self.eos_id is not None, "ByteTokenizer 缺少 BOS/EOS"
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码为文本（过滤特殊 token，忽略无效字节）。

        Args:
            ids: token id 列表。

        Returns:
            解码后的文本。
        """
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")


class HfTokenizer:
    """HuggingFace tokenizers 的 BPE 分词器封装。

    Attributes:
        vocab_size: 词表大小。
        bos_id: BOS token id，词表中不存在时为 None。
        eos_id: EOS token id，词表中不存在时为 None。
        pad_id: PAD token id，词表中不存在时为 None。
        unk_id: UNK token id，词表中不存在时为 None。
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        """初始化封装。

        Args:
            tokenizer: 已加载的 HF Tokenizer 实例。
        """
        self._tok = tokenizer
        self.vocab_size: int = tokenizer.get_vocab_size()
        self.bos_id: int | None = tokenizer.token_to_id("<bos>")
        self.eos_id: int | None = tokenizer.token_to_id("<eos>")
        self.pad_id: int | None = tokenizer.token_to_id("<pad>")
        self.unk_id: int | None = tokenizer.token_to_id("<unk>")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """把文本编码为 BPE token id 列表。

        Args:
            text: 待编码文本。
            add_special_tokens: 是否添加特殊 token。

        Returns:
            token id 列表。
        """
        ids: list[int] = self._tok.encode(text, add_special_tokens=add_special_tokens).ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码为文本。

        Args:
            ids: token id 列表。

        Returns:
            解码后的文本。
        """
        text: str = self._tok.decode(ids)
        return text


def load_tokenizer(kind: str = "byte", path: str | None = None) -> TokenizerLike:
    """加载分词器。

    Args:
        kind: 分词器类型，"byte" 或 "hf_tokenizers"。
        path: hf_tokenizers 的 tokenizer.json 路径。

    Returns:
        实现统一接口的分词器；hf_tokenizers 加载失败时回退 ByteTokenizer。
    """
    if kind == "hf_tokenizers" and path and os.path.exists(path):
        try:
            return HfTokenizer(Tokenizer.from_file(path))
        except Exception:
            pass
    return ByteTokenizer()
