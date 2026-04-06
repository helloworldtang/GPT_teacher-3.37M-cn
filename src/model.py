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


def flash_attention(q, k, v, dropout_p=0.0):
    """
    Flash Attention 实现
    使用 PyTorch 2.0+ 的 scaled_dot_product_attention
    
    参数:
        q: [B, H, T, D]
        k: [B, H, T, D]
        v: [B, H, T, D]
        dropout_p: dropout 概率
    
    返回:
        output: [B, H, T, D]
    """
    if hasattr(F, 'scaled_dot_product_attention'):
        # 使用 is_causal=True 自动应用 causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p if dropout_p > 0 else 0.0,
            is_causal=True
        )
        return out
    else:
        # 回退到标准注意力计算
        return standard_attention(q, k, v)


def standard_attention(q, k, v, mask=None):
    """
    标准注意力计算（回退方案）
    如果没有提供mask，自动创建因果mask
    """
    B, H, T, D = q.shape
    head_dim = D
    
    # 计算注意力分数
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    
    # 应用mask（如果没有提供，创建因果mask）
    if mask is None:
        mask = torch.tril(torch.ones(T, T, device=q.device)).unsqueeze(0).unsqueeze(0)
    attn = attn.masked_fill(mask == 0, float("-inf"))
    
    # softmax
    attn = F.softmax(attn, dim=-1)
    
    # 乘以 value
    out = attn @ v
    
    return out


class SelfAttention(nn.Module):
    def __init__(self, d, n_head, dropout, use_flash=True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d // n_head
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        h = self.n_head
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        
        q = q.view(B, T, h, self.head_dim).transpose(1, 2)
        k = k.view(B, T, h, self.head_dim).transpose(1, 2)
        v = v.view(B, T, h, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE
        q, k = rope(q, k, T, self.head_dim, x.device)
        
        # 使用 Flash Attention 或标准注意力
        if self.use_flash:
            y = flash_attention(q, k, v, self.drop.p)
        else:
            y = standard_attention(q, k, v, mask)
        
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
    def __init__(self, d, n_head, dropout, use_flash=True):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_head, dropout, use_flash)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout)
    
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, seq_len, dropout, use_flash=True):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, use_flash) for _ in range(n_layer)
        ])
        
        self.norm = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # 初始化权重
        self.apply(self._init_weights)
        # 对 GPT-2 风格的残差投影进行特殊初始化
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        # 创建因果 mask（上三角为 1，下三角为 0）
        m = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        
        for blk in self.blocks:
            x = blk(x, m)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
