from __future__ import annotations

import os
import yaml
import math
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import set_seed, ensure_dir, num_threads
from .data import build_datasets, collate
from .model import GPT


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(want: str | None = None):
    if want is None or want == "auto":
        # 优先级: CUDA > MPS > CPU
        if torch.cuda.is_available():
            print("🚀 使用CUDA GPU加速训练")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("🍎 使用M5 Neural Engine (MPS)加速训练")
            return torch.device("mps")
        else:
            print("🖥️ 使用CPU训练（未检测到GPU/MPS）")
            return torch.device("cpu")
    if want == "cpu":
        print("🖥️ 强制使用CPU训练")
        return torch.device("cpu")
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，但指定了--device cuda")
        print("🚀 使用CUDA GPU加速训练")
        return torch.device("cuda")
    if want == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS不可用，但指定了--device mps")
        print("🍎 使用M5 Neural Engine (MPS)加速训练")
        return torch.device("mps")
    raise RuntimeError(f"Unknown device option: {want}. Available options: auto, cpu, cuda, mps")


def train(device_arg: str | None = None, use_flash: bool = True, config_path: str | None = None):
    if config_path is None:
        config_path = "config.yaml"
    cfg = load_config(config_path)
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
        use_flash=use_flash,
    )
    device = get_device(device_arg)
    model.to(device)
    bs = cfg["training"]["batch_size"]
    mb = cfg["training"]["micro_batch"]
    train_loader = DataLoader(
        train_ds,
        batch_size=mb,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate(b, seq_len, tok.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=mb,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, seq_len, tok.pad_id),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
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
    
    early_stopping_patience = cfg["training"].get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0
    early_stop_triggered = False
    
    val_losses = []
    train_losses = []
    
    step = 0
    accum = 0
    model.train()
    start_time = time.time()
    while step < total_steps and not early_stop_triggered:
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
                
                train_losses.append(loss.item())
                
                if step % 10 == 0:
                    print(
                        f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}"
                    )

                if step % cfg["training"]["eval_interval"] == 0:
                    eval_loss = evaluate(model, val_loader, loss_fn, device)
                    elapsed = time.time() - start_time
                    print(f"eval loss {eval_loss:.4f} elapsed {elapsed:.1f}s")
                    
                    val_losses.append(eval_loss)
                    
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        best_step = step
                        patience_counter = 0
                        torch.save(
                            {"model": model.state_dict(), "cfg": cfg},
                            os.path.join(save_dir, "best.pt"),
                        )
                        print(f"  → 新的最佳模型保存 (step {step})")
                    else:
                        patience_counter += 1
                        print(f"  → 验证损失未改善 ({patience_counter}/{early_stopping_patience})")
                        if patience_counter >= early_stopping_patience:
                            print(f"\n=== 早停触发 (step {step}) ===")
                            print(f"最佳验证损失: {best_val_loss:.4f} (step {best_step})")
                            print(f"总训练时间: {elapsed:.1f}s")
                            torch.save(
                                {"model": model.state_dict(), "cfg": cfg},
                                os.path.join(save_dir, "last.pt"),
                            )
                            early_stop_triggered = True
                    
                    torch.save(
                        {"model": model.state_dict(), "cfg": cfg},
                        os.path.join(save_dir, "last.pt"),
                    )
            if step >= total_steps or early_stop_triggered:
                break
    torch.save(
        {"model": model.state_dict(), "cfg": cfg}, os.path.join(save_dir, "last.pt")
    )
    total_elapsed = time.time() - start_time
    print(f"✓ 训练完成，总用时: {total_elapsed:.1f}s")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yml", help="配置文件路径")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="训练设备: auto(自动检测), cpu, cuda(GPU), mps(Apple M芯片)")
    ap.add_argument("--no-flash", action="store_true", help="禁用 Flash Attention")
    args = ap.parse_args()
    train(args.device, use_flash=not args.no_flash, config_path=args.config)
