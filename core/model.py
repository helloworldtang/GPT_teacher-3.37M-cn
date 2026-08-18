import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = 1e-6

    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True)
        n = n * (n.shape[-1] ** -0.5)
        return (x / (n + self.eps)) * self.weight


def rope(q, k, seq_len, head_dim, device, offset=0):
    half = head_dim // 2
    idx = torch.arange(half, device=device)
    pos = torch.arange(offset, offset + seq_len, device=device).unsqueeze(1)
    rates = torch.pow(10000, -2 * idx / head_dim)
    theta = pos * rates
    cos = torch.cos(theta).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(2)

    def apply(x):
        x1 = x[..., :half]
        x2 = x[..., half:half * 2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return apply(q), apply(k)


def flash_attention(q, k, v, dropout_p=0.0, is_causal=True):
    if hasattr(F, 'scaled_dot_product_attention'):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p if dropout_p > 0 else 0.0,
            is_causal=is_causal,
        )
    return standard_attention(q, k, v)


def standard_attention(q, k, v, mask=None):
    B, H, T, D = q.shape
    T_kv = k.shape[2]
    attn = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    elif T > 1:
        mask = torch.tril(torch.ones(T, T_kv, device=q.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return attn @ v


class SelfAttention(nn.Module):
    """GQA: n_kv_heads 可以少于 n_head，减少 KV Cache 体积。"""

    def __init__(self, d, n_head, dropout, use_flash=True, n_kv_head=None):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d // n_head
        self.n_rep = n_head // self.n_kv_head  # 每个 KV head 被几个 Q head 复用
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')

        self.wq = nn.Linear(d, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def _repeat_kv(self, x):
        """将 KV head 重复以匹配 Q head 数量: [B, S, H_kv, D] -> [B, S, H_q, D]"""
        if self.n_rep == 1:
            return x
        B, S, H_kv, D = x.shape
        return x[:, :, :, None, :].expand(B, S, H_kv, self.n_rep, D).reshape(B, S, H_kv * self.n_rep, D)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)

        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        q, k = rope(q, k, T, self.head_dim, x.device, offset=offset)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        new_kv_cache = (k.detach(), v.detach()) if not self.training else None

        # GQA: 扩展 KV heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_flash:
            y = flash_attention(q, k, v, self.drop.p, is_causal=(T > 1))
        else:
            y = standard_attention(q, k, v, mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.drop(y)
        return y, new_kv_cache


class MLP(nn.Module):
    """SwiGLU: gate = silu(W_gate * x), out = gate * (W_up * x), out = W_down(out)"""

    def __init__(self, d, dropout):
        super().__init__()
        # SwiGLU 需要三个线性层，但中间维度可以小一些补偿
        hidden = int(4 * d * 2 / 3)  # Llama 风格: 4d * 2/3 再对齐
        hidden = ((hidden + 63) // 64) * 64  # 对齐到 64 的倍数
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, d, n_head, dropout, use_flash=True, n_kv_head=None):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_head, dropout, use_flash, n_kv_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout)

    def forward(self, x, mask=None, kv_cache=None):
        attn_out, new_kv = self.attn(self.norm1(x), mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, seq_len, dropout,
                 use_flash=True, n_kv_head=None):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, use_flash, n_kv_head)
            for _ in range(n_layer)
        ])

        self.norm = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('w_down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_caches=None):
        B, T = idx.shape
        x = self.tok_emb(idx)

        new_kv_caches = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = blk(x, mask=None, kv_cache=kv)
            new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_kv_caches
