import pytest

from core.model import GPT
from core.tokenizer import ByteTokenizer


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


@pytest.fixture
def small_model():
    return GPT(
        vocab_size=259,
        n_layer=2,
        n_head=4,
        n_embd=32,
        seq_len=16,
        dropout=0.0,
        use_flash=False,
        n_kv_head=2,
    )
