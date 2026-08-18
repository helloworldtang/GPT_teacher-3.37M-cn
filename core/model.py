"""Decoder-only Transformer（GQA + SwiGLU + RMSNorm + RoPE + 权重共享）。"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 每层注意力返回的 KV Cache：(k, v)，训练时为 None
KvCache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    """均方根归一化：比 LayerNorm 少一次均值计算。"""

    def __init__(self, d: int) -> None:
        """初始化归一化层。

        Args:
            d: 归一化的特征维度。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        n = x.norm(dim=-1, keepdim=True)
        n = n * (n.shape[-1] ** -0.5)
        out: Tensor = (x / (n + self.eps)) * self.weight
        return out


def rope(
    q: Tensor, k: Tensor, seq_len: int, head_dim: int, device: torch.device | str, offset: int = 0
) -> tuple[Tensor, Tensor]:
    """旋转位置编码（RoPE）：用复数旋转把相对位置注入 Q/K。

    Args:
        q: 查询向量，[B, T, H, D]。
        k: 键向量，[B, T, H_kv, D]。
        seq_len: 当前输入序列长度。
        head_dim: 每个头的维度。
        device: 张量所在设备。
        offset: KV Cache 场景下已缓存的长度（位置偏移）。

    Returns:
        旋转后的 (q, k)，形状不变。
    """
    half = head_dim // 2
    idx = torch.arange(half, device=device)
    pos = torch.arange(offset, offset + seq_len, device=device).unsqueeze(1)
    rates = torch.pow(10000, -2 * idx / head_dim)
    theta = pos * rates
    cos = torch.cos(theta).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(2)

    def apply(x: Tensor) -> Tensor:
        x1 = x[..., :half]
        x2 = x[..., half : half * 2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return apply(q), apply(k)


def flash_attention(q: Tensor, k: Tensor, v: Tensor, dropout_p: float = 0.0, is_causal: bool = True) -> Tensor:
    """Flash Attention（scaled_dot_product_attention），不可用时回退标准实现。

    Args:
        q: 查询向量，[B, H, T, D]。
        k: 键向量，[B, H, T_kv, D]。
        v: 值向量，[B, H, T_kv, D]。
        dropout_p: dropout 概率。
        is_causal: 是否使用因果掩码。

    Returns:
        注意力输出，[B, H, T, D]。
    """
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p if dropout_p > 0 else 0.0,
            is_causal=is_causal,
        )
    return standard_attention(q, k, v)


def standard_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """标准缩放点积注意力：手动构造因果掩码。

    Args:
        q: 查询向量，[B, H, T, D]。
        k: 键向量，[B, H, T_kv, D]。
        v: 值向量，[B, H, T_kv, D]。
        mask: 可选掩码，None 时按 T>1 自动构造下三角因果掩码。

    Returns:
        注意力输出，[B, H, T, D]。
    """
    B, H, T, D = q.shape
    T_kv = k.shape[2]
    attn = (q @ k.transpose(-2, -1)) * (D**-0.5)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    elif T > 1:
        causal = torch.tril(torch.ones(T, T_kv, device=q.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return attn @ v


class SelfAttention(nn.Module):
    """分组查询注意力（GQA）：n_kv_head 可以少于 n_head，减少 KV Cache 体积。"""

    def __init__(
        self, d: int, n_head: int, dropout: float, use_flash: bool = True, n_kv_head: int | None = None
    ) -> None:
        """初始化注意力层。

        Args:
            d: 输入特征维度。
            n_head: 查询头数。
            dropout: dropout 概率。
            use_flash: 是否使用 Flash Attention。
            n_kv_head: KV 头数，None 时与 n_head 相同（退化为 MHA）。
        """
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d // n_head
        self.n_rep = n_head // self.n_kv_head  # 每个 KV head 被几个 Q head 复用
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        self.wq = nn.Linear(d, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """将 KV head 重复以匹配 Q head 数量：[B, S, H_kv, D] -> [B, S, H_q, D]。

        Args:
            x: KV 张量。

        Returns:
            扩展后的 KV 张量。
        """
        if self.n_rep == 1:
            return x
        B, S, H_kv, D = x.shape
        return x[:, :, :, None, :].expand(B, S, H_kv, self.n_rep, D).reshape(B, S, H_kv * self.n_rep, D)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, kv_cache: KvCache | None = None
    ) -> tuple[Tensor, KvCache | None]:
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

        new_kv_cache: KvCache | None = (k.detach(), v.detach()) if not self.training else None

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
    """SwiGLU 前馈网络：out = W_down(silu(W_gate·x) * (W_up·x))。"""

    def __init__(self, d: int, dropout: float) -> None:
        """初始化前馈层。

        Args:
            d: 输入/输出特征维度。
            dropout: dropout 概率。
        """
        super().__init__()
        # SwiGLU 需要三个线性层，但中间维度可以小一些补偿
        hidden = int(4 * d * 2 / 3)  # Llama 风格: 4d * 2/3 再对齐
        hidden = ((hidden + 63) // 64) * 64  # 对齐到 64 的倍数
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
        return out


class Block(nn.Module):
    """Transformer 层：RMSNorm → 注意力 + 残差 → RMSNorm → SwiGLU + 残差。"""

    def __init__(
        self, d: int, n_head: int, dropout: float, use_flash: bool = True, n_kv_head: int | None = None
    ) -> None:
        """初始化 Transformer 层。

        Args:
            d: 特征维度。
            n_head: 查询头数。
            dropout: dropout 概率。
            use_flash: 是否使用 Flash Attention。
            n_kv_head: KV 头数，None 时与 n_head 相同。
        """
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_head, dropout, use_flash, n_kv_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, kv_cache: KvCache | None = None
    ) -> tuple[Tensor, KvCache | None]:
        attn_out, new_kv = self.attn(self.norm1(x), mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    """小参数中文 GPT：权重共享（LM head 复用词嵌入）+ KV Cache 推理。

    Example:
        >>> model = GPT(vocab_size=4096, n_layer=4, n_head=4, n_embd=256, seq_len=128, dropout=0.1)
        >>> logits, kv_caches = model(idx)
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        seq_len: int,
        dropout: float,
        use_flash: bool = True,
        n_kv_head: int | None = None,
    ) -> None:
        """初始化模型。

        Args:
            vocab_size: 词表大小。
            n_layer: Transformer 层数。
            n_head: 查询头数。
            n_embd: 嵌入维度。
            seq_len: 最大序列长度。
            dropout: dropout 概率。
            use_flash: 是否使用 Flash Attention。
            n_kv_head: KV 头数，None 时与 n_head 相同。
        """
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, use_flash, n_kv_head) for _ in range(n_layer)])

        self.norm = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """按 GPT-2 惯例初始化权重（std=0.02）。"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: Tensor, kv_caches: list[KvCache | None] | None = None
    ) -> tuple[Tensor, list[KvCache | None]]:
        """前向计算。

        Args:
            idx: 输入 token id，[B, T]。
            kv_caches: 每层的 KV Cache（增量推理），首步传 None。

        Returns:
            (logits, 新的每层 KV Cache 列表)，训练态下 KV Cache 为 None。
        """
        x = self.tok_emb(idx)

        new_kv_caches: list[KvCache | None] = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = blk(x, mask=None, kv_cache=kv)
            new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_kv_caches
