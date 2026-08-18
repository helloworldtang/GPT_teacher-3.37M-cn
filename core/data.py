"""指令数据集：把 prompt/completion 对编码为带掩码的训练样本。"""

import json
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.tokenizer import TokenizerLike, load_tokenizer

# 一条训练样本：(输入 id 序列, 目标 id 序列)，目标中 prompt 部分被置为 -100
Sample = tuple[list[int], list[int]]


class InstructDataset(Dataset[Sample]):
    """指令微调数据集。

    每条样本构造为 "用户:{prompt}\\n助手:{completion}<eos>"，
    目标序列中 prompt 前缀部分置 -100（不参与 loss），
    超过 seq_len 时从尾部截断 completion。

    Attributes:
        samples: (输入序列, 目标序列) 列表。
        tok: 分词器。
        seq_len: 最大序列长度。
    """

    def __init__(self, path: str, tokenizer: TokenizerLike, seq_len: int) -> None:
        """从 jsonl 文件加载数据并编码。

        Args:
            path: jsonl 路径，每行一个 {"prompt": ..., "completion": ...} 对象。
            tokenizer: 分词器。
            seq_len: 最大序列长度。
        """
        self.samples: list[Sample] = []
        self.tok = tokenizer
        self.seq_len = seq_len
        assert self.tok.bos_id is not None and self.tok.eos_id is not None, "分词器缺少 BOS/EOS 特殊 token"
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                completion = obj.get("completion", "")
                prefix = [self.tok.bos_id, *self.tok.encode("用户:" + prompt + "\n助手:", add_special_tokens=False)]
                comp = self.tok.encode(completion, add_special_tokens=False)
                ids = prefix + comp + [self.tok.eos_id]

                if len(ids) > seq_len:
                    max_comp_len = seq_len - len(prefix) - 1
                    if max_comp_len > 0:
                        ids = prefix + comp[:max_comp_len] + [self.tok.eos_id]
                    else:
                        ids = prefix[: seq_len - 1] + [self.tok.eos_id]

                tar = ids[1:] + [self.tok.eos_id]
                ignore = min(max(0, len(prefix) - 1), len(tar))
                tar[:ignore] = [-100] * ignore
                self.samples.append((ids, tar))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate(batch: list[Sample], seq_len: int, pad_id: int) -> tuple[Tensor, Tensor]:
    """把一批样本 pad 到 seq_len 并堆叠为张量。

    Args:
        batch: 样本列表。
        seq_len: 目标序列长度。
        pad_id: 输入侧的填充 token id（目标侧固定用 -100）。

    Returns:
        (输入张量, 目标张量)，均为 [B, seq_len] 的 long 张量。
    """
    x = []
    y = []
    for a, b in batch:
        pa = a + [pad_id] * (seq_len - len(a))
        pb = b + [-100] * (seq_len - len(b))
        x.append(pa)
        y.append(pb)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def build_datasets(cfg: dict[str, Any]) -> tuple[TokenizerLike, InstructDataset, InstructDataset]:
    """按配置构建分词器与训练/验证数据集。

    Args:
        cfg: 完整配置字典（config.yml 加载结果）。

    Returns:
        (分词器, 训练集, 验证集)。

    Raises:
        ValueError: data.format 配置了不支持的值。
    """
    tok = load_tokenizer(
        cfg.get("tokenizer", {}).get("type", "byte"),
        cfg.get("tokenizer", {}).get("path"),
    )
    seq_len = cfg["model"]["seq_len"]
    fmt = cfg["data"].get("format", "instruct")
    if fmt != "instruct":
        raise ValueError(f"配置项data.format的值无效：{fmt}。仅支持配置为'instruct'，或不配置使用默认值'instruct'。")
    train_ds = InstructDataset(cfg["data"]["train_path"], tok, seq_len)
    val_ds = InstructDataset(cfg["data"]["val_path"], tok, seq_len)
    return tok, train_ds, val_ds
