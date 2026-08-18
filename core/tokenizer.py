import os
from typing import List, Optional

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 259
        self.bos_id = 256
        self.eos_id = 257
        self.pad_id = 258
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        b = text.encode("utf-8")
        ids = list(b)
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids
    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i < 256]
        return bytes(ids).decode("utf-8", errors="ignore")

def load_tokenizer(kind: str = "byte", path: Optional[str] = None):
    if kind == "hf_tokenizers" and path and os.path.exists(path):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(path)
            class _Tok:
                def __init__(self, t):
                    self.t = t
                    self.vocab_size = t.get_vocab_size()
                    self.bos_id = t.token_to_id("<bos>")
                    self.eos_id = t.token_to_id("<eos>")
                    self.pad_id = t.token_to_id("<pad>")
                def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                    return self.t.encode(text, add_special_tokens=add_special_tokens).ids
                def decode(self, ids: List[int]) -> str:
                    return self.t.decode(ids)
            return _Tok(tok)
        except Exception:
            pass
    return ByteTokenizer()
