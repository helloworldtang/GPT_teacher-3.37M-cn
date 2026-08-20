"""训练主流程：数据 → 模型 → AdamW + warmup/cosine 训练 → 验收。"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from core.data import build_datasets, collate
from core.model import GPT
from core.utils import ensure_dir, file_md5, num_threads, set_seed


def _git_hash() -> str | None:
    """返回当前 git commit 短 hash，非 git 环境返回 None。"""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def build_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """构建 checkpoint 版本指纹：代码/数据/分词器的版本坐标。

    tokenizer、数据与模型权重必须配套演进（词表变化会使旧权重静默失效），
    指纹让每个 checkpoint 自带完整来源信息，支持审计与复算。

    Args:
        cfg: 训练配置。

    Returns:
        含 git hash、训练/验证数据 md5、tokenizer md5 的字典。
    """
    tok_path = cfg.get("tokenizer", {}).get("path")
    return {
        "git": _git_hash(),
        "train_data_md5": file_md5(cfg["data"]["train_path"]),
        "val_data_md5": file_md5(cfg["data"]["val_path"]),
        "tokenizer_md5": file_md5(tok_path),
    }


def load_config(path: str) -> dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        path: 配置文件路径。

    Returns:
        配置字典。
    """
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def get_device(want: str | None = None) -> torch.device:
    """解析训练设备。

    Args:
        want: "auto"/"cpu"/"cuda"/"mps"，None 等同 auto。

    Returns:
        目标设备。

    Raises:
        RuntimeError: 指定了不可用的设备或未知选项。
    """
    if want is None or want == "auto":
        # 优先级: CUDA > MPS > CPU
        if torch.cuda.is_available():
            print("🚀 使用CUDA GPU加速训练")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("🍎 使用Apple Neural Engine (MPS)加速训练")
            return torch.device("mps")
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
        print("🍎 使用Apple Neural Engine (MPS)加速训练")
        return torch.device("mps")
    raise RuntimeError(f"Unknown device option: {want}. Available options: auto, cpu, cuda, mps")


def plot_loss_curve(train_losses: list[float], val_losses: list[float], eval_interval: int, save_path: str) -> None:
    """绘制并保存训练/验证 loss 曲线。

    matplotlib 未安装时静默跳过。

    Args:
        train_losses: 每 step 的训练 loss。
        val_losses: 每次评估的验证 loss。
        eval_interval: 评估间隔（step 数）。
        save_path: 输出 png 路径。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  提示: 安装 matplotlib 可生成 loss 曲线图 (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    steps_train = list(range(1, len(train_losses) + 1))
    ax.plot(steps_train, train_losses, alpha=0.3, label="train loss")

    steps_val = list(range(eval_interval, eval_interval * (len(val_losses) + 1), eval_interval))
    ax.plot(steps_val, val_losses, "o-", label="val loss", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss 曲线已保存: {save_path}")


def evaluate(model: GPT, loader: DataLoader[Any], loss_fn: nn.Module, device: torch.device) -> float:
    """在验证集上计算平均 loss（结束后恢复训练态）。

    Args:
        model: 待评估模型。
        loader: 验证集 DataLoader。
        loss_fn: 损失函数。
        device: 计算设备。

    Returns:
        平均验证 loss。
    """
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            total += loss.item()
            count += 1
    model.train()
    return total / max(1, count)


def train(
    device_arg: str | None = None,
    use_flash: bool = True,
    config_path: str | None = None,
    max_steps_override: int | None = None,
    quantize: bool = False,
) -> None:
    """完整训练流程：早停、checkpoint、loss 曲线、量化导出与自动验收。

    Args:
        device_arg: 训练设备（None/auto/cpu/cuda/mps）。
        use_flash: 是否启用 Flash Attention。
        config_path: 配置文件路径，默认 train/config.yml。
        max_steps_override: 覆盖配置中的最大训练步数。
        quantize: 是否额外导出 int8 动态量化模型（需 fbgemm/qnnpack 引擎，
            macOS 不支持；产物 quantized.pt 目前无推理链路消费，仅教学演示用）。
    """
    if config_path is None:
        config_path = "train/config.yml"
    cfg = load_config(config_path)
    fp = build_fingerprint(cfg)
    print(f"版本指纹: git={fp['git']} data={fp['train_data_md5']} tokenizer={fp['tokenizer_md5']}")
    set_seed(cfg["training"]["seed"])
    torch.set_num_threads(num_threads())
    tok, train_ds, val_ds = build_datasets(cfg)
    seq_len: int = cfg["model"]["seq_len"]
    model = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=seq_len,
        dropout=cfg["model"]["dropout"],
        use_flash=use_flash,
        n_kv_head=cfg["model"].get("n_kv_head"),
    )
    device = get_device(device_arg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"设备: {device} | 序列长度: {seq_len} | 训练集: {len(train_ds)} 条")
    print(f"{'=' * 50}")

    assert tok.pad_id is not None, "分词器缺少 PAD 特殊 token"
    pad_id: int = tok.pad_id
    bs: int = cfg["training"]["batch_size"]
    mb: int = cfg["training"]["micro_batch"]
    train_loader = DataLoader(
        train_ds,
        batch_size=mb,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type != "cpu",
        collate_fn=lambda b: collate(b, seq_len, pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=mb,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type != "cpu",
        collate_fn=lambda b: collate(b, seq_len, pad_id),
    )
    opt = torch.optim.AdamW(  # type: ignore[attr-defined]
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps: int = max_steps_override if max_steps_override else cfg["training"]["max_steps"]
    warmup: int = cfg["training"]["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    save_dir: str = cfg["training"]["save_dir"]
    ensure_dir(save_dir)

    early_stopping_patience: int = cfg["training"].get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0
    early_stop_triggered = False
    eval_interval: int = cfg["training"]["eval_interval"]

    val_losses: list[float] = []
    train_losses: list[float] = []

    # 尝试导入 tqdm 用于进度条
    try:
        from tqdm import tqdm

        pbar = tqdm(
            total=total_steps,
            desc="训练",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}",
        )
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    step = 0
    accum = 0
    model.train()
    start_time = time.time()

    def save_ckpt(name: str) -> None:
        torch.save({"model": model.state_dict(), "cfg": cfg, "fingerprint": fp}, os.path.join(save_dir, name))

    while step < total_steps and not early_stop_triggered:
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
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

                loss_val = loss.item()
                train_losses.append(loss_val)

                if use_tqdm:
                    pbar.set_postfix_str(f"{loss_val:.4f}")
                    pbar.update(1)
                elif step % 10 == 0:
                    print(f"step {step} loss {loss_val:.4f} lr {sched.get_last_lr()[0]:.6f}")

                if step % eval_interval == 0:
                    eval_loss = evaluate(model, val_loader, loss_fn, device)
                    elapsed = time.time() - start_time

                    if use_tqdm:
                        pbar.write(f"eval loss {eval_loss:.4f} elapsed {elapsed:.1f}s")
                    else:
                        print(f"eval loss {eval_loss:.4f} elapsed {elapsed:.1f}s")

                    val_losses.append(eval_loss)

                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        best_step = step
                        patience_counter = 0
                        save_ckpt("best.pt")
                        if use_tqdm:
                            pbar.write(f"  → 新的最佳模型保存 (step {step})")
                        else:
                            print(f"  → 新的最佳模型保存 (step {step})")
                    else:
                        patience_counter += 1
                        msg = f"  → 验证损失未改善 ({patience_counter}/{early_stopping_patience})"
                        if use_tqdm:
                            pbar.write(msg)
                        else:
                            print(msg)
                        if patience_counter >= early_stopping_patience:
                            if use_tqdm:
                                pbar.write("\n=== 早停触发 (step {step}) ===")
                                pbar.write(f"最佳验证损失: {best_val_loss:.4f} (step {best_step})")
                                pbar.write(f"总训练时间: {elapsed:.1f}s")
                            else:
                                print(f"\n=== 早停触发 (step {step}) ===")
                                print(f"最佳验证损失: {best_val_loss:.4f} (step {best_step})")
                                print(f"总训练时间: {elapsed:.1f}s")
                            early_stop_triggered = True

                    save_ckpt("last.pt")
                    # 实时更新 loss 曲线
                    loss_curve_path = os.path.join(save_dir, "loss_curve.png")
                    plot_loss_curve(train_losses, val_losses, eval_interval, loss_curve_path)
            if step >= total_steps or early_stop_triggered:
                break

    if use_tqdm:
        pbar.close()

    save_ckpt("last.pt")
    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 50}")
    print(f"✓ 训练完成，总用时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    print(f"  最佳验证损失: {best_val_loss:.4f} (step {best_step})")

    # 保存 loss 曲线
    loss_curve_path = os.path.join(save_dir, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, eval_interval, loss_curve_path)

    # 保存训练耗时
    with open(os.path.join(save_dir, "train_time.txt"), "w") as f:
        f.write(f"{total_elapsed:.1f}\n")

    # 保存 loss 记录供后续分析
    loss_data = {"train_losses": train_losses, "val_losses": val_losses, "eval_interval": eval_interval}
    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
        json.dump(loss_data, f)

    # 导出量化模型：默认关闭（产物无消费链路，且 macOS 无量化引擎），
    # 需要体验动态量化时用 --quantize 显式开启（Linux x86/ARM 支持）
    if quantize:
        try:
            model.cpu()
            quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)  # type: ignore[attr-defined]
            torch.save(
                {"model": quantized.state_dict(), "cfg": cfg},
                os.path.join(save_dir, "quantized.pt"),
            )
            print("  量化模型已保存: train/checkpoints/quantized.pt")
        except Exception as e:
            print(f"  量化导出跳过（--quantize 已开启，但当前平台无动态量化引擎: {e}）")

    # 自动运行验收测试
    print(f"\n{'=' * 50}")
    print("开始验收测试...")
    print(f"{'=' * 50}")
    try:
        from core.evaluate import run_test
        from core.infer import load_model_and_tokenizer

        model.eval()
        # 加载 best 模型进行验收
        best_path = os.path.join(save_dir, "best.pt")
        if not os.path.exists(best_path):
            raise RuntimeError(f"未生成 best checkpoint（{best_path}），无法验收")
        eval_model, _, _, _ = load_model_and_tokenizer(best_path, device=device)
        passed, total = run_test(eval_model, tok, device)
        # 验收未达线（与 core.evaluate 的 80% 阈值一致）时以非零退出码结束：
        # 一键流程的成败必须可被脚本与 CI 感知，不能只在控制台打印。
        if total == 0 or passed / total < 0.8:
            raise SystemExit(f"验收未达线: {passed}/{total} 通过（需 ≥80%），退出码置 1")
    except SystemExit:
        raise
    except Exception as e:
        print(f"验收测试执行出错: {e}")
        print("可手动运行: uv run python -m core.evaluate")
        raise SystemExit(1) from e


def main() -> None:
    """命令行训练入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="train/config.yml", help="配置文件路径")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="训练设备: auto(自动检测), cpu, cuda(GPU), mps(Apple M芯片)",
    )
    ap.add_argument("--no-flash", action="store_true", help="禁用 Flash Attention")
    ap.add_argument("--quantize", action="store_true", help="额外导出 int8 动态量化模型（macOS 不支持）")
    ap.add_argument("--max_steps", type=int, default=None, help="覆盖配置文件中的训练步数")
    args = ap.parse_args()
    train(
        args.device,
        use_flash=not args.no_flash,
        config_path=args.config,
        max_steps_override=args.max_steps,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
