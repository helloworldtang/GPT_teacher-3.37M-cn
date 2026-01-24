import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = 1e-6
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True)
        n = n * (n.shape[-1] ** -0.5)
        return (x / (n + self.eps)) * self.weight

def rope(q, k, seq_len, head_dim, device):
    half = head_dim // 2
    idx = torch.arange(half, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    rates = torch.pow(10000, -2 * idx / head_dim)
    theta = pos * rates
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    def apply(x):
        x1 = x[..., :half]
        x2 = x[..., half:half*2]
        xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return xr
    return apply(q), apply(k)

class SelfAttention(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mask):
        B, T, C = x.shape
        h = self.n_head
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, h, self.head_dim).transpose(1, 2)
        k = k.view(B, T, h, self.head_dim).transpose(1, 2)
        v = v.view(B, T, h, self.head_dim).transpose(1, 2)
        q, k = rope(q, k, T, self.head_dim, x.device)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.drop(y)
        return y

class MLP(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_head, dropout)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout)
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        m = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, m)
        x = self.norm(x)
        logits = self.head(x)
        return logits
