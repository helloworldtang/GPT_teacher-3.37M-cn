import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import load_tokenizer

class InstructDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self.samples = []
        self.tok = tokenizer
        self.seq_len = seq_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                completion = obj.get("completion", "")
                prefix = self.tok.encode("用户:" + prompt + "\n助手:", add_special_tokens=True)
                comp = self.tok.encode(completion, add_special_tokens=False)
                ids = prefix + comp + [self.tok.eos_id]
                if len(ids) > seq_len:
                    ids = ids[:seq_len]
                tar = ids[1:] + [self.tok.eos_id]
                ignore = min(max(0, len(prefix) - 1), len(tar))
                tar[:ignore] = [-100] * ignore
                self.samples.append((ids, tar))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]



def collate(batch: List[Tuple[List[int], List[int]]], seq_len: int, pad_id: int):
    x = []
    y = []
    for a, b in batch:
        pa = a + [pad_id] * (seq_len - len(a))
        pb = b + [-100] * (seq_len - len(b))
        x.append(pa)
        y.append(pb)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_datasets(cfg):
    tok = load_tokenizer(
        cfg.get("tokenizer", {}).get("type", "byte"),
        cfg.get("tokenizer", {}).get("path"),
    )
    seq_len = cfg["model"]["seq_len"]
    fmt = cfg["data"].get("format", "instruct")
    if fmt != "instruct":
        raise ValueError(
            f"配置项data.format的值无效：{fmt}。仅支持配置为'instruct'，或不配置使用默认值'instruct'。"
        )
    train_ds = InstructDataset(cfg["data"]["train_path"], tok, seq_len)
    val_ds = InstructDataset(cfg["data"]["val_path"], tok, seq_len)
    return tok, train_ds, val_ds
