import os
import yaml
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import set_seed, ensure_dir, num_threads
from .data import build_datasets, collate
from .model import GPT

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device(cfg):
    d = (
        cfg.get("training", {}).get("device")
        if isinstance(cfg, dict)
        else None
    )
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if d == "cpu" or d is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")

def train():
    cfg = load_config("config.yaml")
    set_seed(cfg["training"]["seed"])
    torch.set_num_threads(num_threads())
    tok, train_ds, val_ds = build_datasets(cfg)
    seq_len = cfg["model"]["seq_len"]
    model = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=seq_len,
        dropout=cfg["model"]["dropout"],
    )
    device = get_device(cfg)
    model.to(device)
    bs = cfg["training"]["batch_size"]
    mb = cfg["training"]["micro_batch"]
    train_loader = DataLoader(train_ds, batch_size=mb, shuffle=True, num_workers=0, collate_fn=lambda b: collate(b, seq_len, tok.pad_id))
    val_loader = DataLoader(val_ds, batch_size=mb, shuffle=False, num_workers=0, collate_fn=lambda b: collate(b, seq_len, tok.pad_id))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    total_steps = cfg["training"]["max_steps"]
    warmup = cfg["training"]["warmup_steps"]
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    save_dir = cfg["training"]["save_dir"]
    ensure_dir(save_dir)
    step = 0
    accum = 0
    model.train()
    start_time = time.time()
    while step < total_steps:
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            accum += 1
            if accum == bs // mb:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                sched.step()
                step += 1
                accum = 0
                if step % 10 == 0:
                    print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}")
                if step % cfg["training"]["eval_interval"] == 0:
                    eval_loss = evaluate(model, val_loader, loss_fn, device)
                    elapsed = time.time() - start_time
                    print(f"eval loss {eval_loss:.4f} elapsed {elapsed:.1f}s")
                    torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(save_dir, "last.pt"))
                if step >= total_steps:
                    break
    torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(save_dir, "last.pt"))
    total_elapsed = time.time() - start_time
    with open(os.path.join(save_dir, "train_time.txt"), "w") as f:
        f.write(f"elapsed_seconds={total_elapsed:.2f}\n")
    qmodel = torch.quantization.quantize_dynamic(model.to("cpu"), {nn.Linear}, dtype=torch.qint8)
    torch.save({"model": qmodel.state_dict(), "cfg": cfg}, os.path.join(save_dir, "quantized.pt"))

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            total += loss.item()
            count += 1
    model.train()
    return total / max(1, count)

if __name__ == "__main__":
    train()
