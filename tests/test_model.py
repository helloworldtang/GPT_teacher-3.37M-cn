import torch
from src.model import RMSNorm, rope, SelfAttention, MLP, Block, GPT


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(8)
        x = torch.randn(2, 3, 8)
        assert norm(x).shape == x.shape

    def test_normalization_effect(self):
        norm = RMSNorm(4)
        norm.weight.data.fill_(1.0)
        x = torch.randn(2, 4)
        out = norm(x)
        # 验证输出范数被归一化（相比输入大幅缩小）
        assert out.norm(dim=-1).max() < x.norm(dim=-1).max()

    def test_weight_scaling(self):
        norm = RMSNorm(4)
        norm.weight.data.fill_(2.0)
        x = torch.randn(1, 4)
        out_w1 = RMSNorm(4)(x)
        norm2 = RMSNorm(4)
        norm2.weight.data.fill_(2.0)
        out_w2 = norm2(x)
        # weight=2 的输出应该是 weight=1 的 2 倍
        assert torch.allclose(out_w2 / out_w1, torch.tensor(2.0), atol=0.1)


class TestRoPE:
    def test_output_shape(self):
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 4, 2, 8)
        q2, k2 = rope(q, k, seq_len=4, head_dim=8, device="cpu")
        assert q2.shape == q.shape
        assert k2.shape == k.shape

    def test_applies_rotation(self):
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 4, 2, 8)
        q2, k2 = rope(q, k, seq_len=4, head_dim=8, device="cpu")
        assert not torch.allclose(q, q2)
        assert not torch.allclose(k, k2)

    def test_offset(self):
        q = torch.randn(1, 1, 2, 8)
        k = torch.randn(1, 1, 2, 8)
        q1, k1 = rope(q, k, seq_len=1, head_dim=8, device="cpu", offset=0)
        q2, k2 = rope(q, k, seq_len=1, head_dim=8, device="cpu", offset=10)
        assert not torch.allclose(q1, q2)


class TestSelfAttention:
    def test_output_shape(self):
        attn = SelfAttention(32, n_head=4, dropout=0.0, use_flash=False, n_kv_head=2)
        x = torch.randn(1, 8, 32)
        out, cache = attn(x)
        assert out.shape == (1, 8, 32)
        assert cache is None  # training mode

    def test_gqa_kv_cache(self):
        attn = SelfAttention(32, n_head=4, dropout=0.0, use_flash=False, n_kv_head=2)
        attn.eval()
        x = torch.randn(1, 4, 32)
        _, cache = attn(x, kv_cache=(torch.randn(1, 0, 2, 8), torch.randn(1, 0, 2, 8)))
        assert cache is not None
        assert cache[0].shape[1] == 4

    def test_incremental_kv_cache(self):
        attn = SelfAttention(32, n_head=4, dropout=0.0, use_flash=False, n_kv_head=2)
        attn.eval()
        x1 = torch.randn(1, 4, 32)
        _, (k1, v1) = attn(x1)
        x2 = torch.randn(1, 2, 32)
        _, (k2, v2) = attn(x2, kv_cache=(k1, v1))
        assert k2.shape[1] == 6
        assert v2.shape[1] == 6


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(32, dropout=0.0)
        x = torch.randn(2, 10, 32)
        assert mlp(x).shape == (2, 10, 32)


class TestBlock:
    def test_output_shape(self):
        block = Block(32, n_head=4, dropout=0.0, use_flash=False, n_kv_head=2)
        x = torch.randn(1, 8, 32)
        out, _ = block(x)
        assert out.shape == (1, 8, 32)


class TestGPT:
    def test_forward_output_shape(self, small_model):
        idx = torch.randint(0, 259, (2, 8))
        logits, caches = small_model(idx)
        assert logits.shape == (2, 8, 259)
        assert len(caches) == 2

    def test_weight_tying(self, small_model):
        assert small_model.head.weight.data_ptr() == small_model.tok_emb.weight.data_ptr()

    def test_kv_cache_incremental(self, small_model):
        small_model.eval()
        idx = torch.randint(0, 259, (1, 8))
        logits1, caches = small_model(idx)
        next_tok = logits1[:, -1:, :].argmax(dim=-1)
        logits2, _ = small_model(next_tok, kv_caches=caches)
        assert logits2.shape == (1, 1, 259)
