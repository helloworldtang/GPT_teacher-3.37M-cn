"""核心概念可视化：让抽象理论变成看得见的图。

用法：
    uv run python -m src.visualize          # 生成全部图到 docs/
    uv run python -m src.visualize --only causal    # 只看因果掩码
    uv run python -m src.visualize --only attention # 只看注意力权重
    uv run python -m src.visualize --only loss      # 只看 loss 曲线解读
"""
import argparse
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("请安装 matplotlib: pip install matplotlib")
    exit(1)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def plot_causal_mask(seq_len=16, save_dir="docs"):
    """可视化因果掩码：1 表示能看到，0 表示被遮住。"""
    os.makedirs(save_dir, exist_ok=True)
    mask = np.tril(np.ones((seq_len, seq_len)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：掩码矩阵
    ax = axes[0]
    ax.imshow(mask, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("Key 位置（被看的）")
    ax.set_ylabel("Query 位置（主动看的）")
    ax.set_title("因果掩码（Causal Mask）")
    # 标注几个位置
    ax.annotate("自己看自己 ✓", xy=(3, 3), fontsize=9, color="white", ha="center", va="center")
    ax.annotate("未来 ✗", xy=(12, 3), fontsize=9, color="red", ha="center", va="center",
                fontweight="bold")
    ax.annotate("历史 ✓", xy=(3, 8), fontsize=9, color="white", ha="center", va="center")

    # 右：文字解释
    ax2 = axes[1]
    ax2.axis("off")
    text = (
        "因果掩码是什么？\n\n"
        "一个下三角矩阵，确保每个位置只能\n"
        "「看到」自己和之前的词，不能偷看\n"
        "未来的词。\n\n"
        "为什么要这样？\n\n"
        "因为 GPT 是逐字生成的：写第 3 个\n"
        "字时，你只看到了第 1、2 个字，\n"
        "不可能看到第 4 个字。\n\n"
        "绿色 = 允许关注    红色 = 被遮住"
    )
    ax2.text(0.1, 0.95, text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="sans-serif",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "causal_mask.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  因果掩码图: {path}")


def plot_attention_weights(save_dir="docs"):
    """可视化注意力权重：模型在回答时关注了哪些词。"""
    os.makedirs(save_dir, exist_ok=True)

    # 模拟一个真实场景：模型处理 "用户:什么是注意力机制？\n助手:注意力机制通过..."
    tokens = list("用:什么是注意力机制？\n助手:注意力机制通过计算查询")
    n = len(tokens)

    # 模拟注意力模式：每个位置主要关注语义相关的位置
    attn = np.random.rand(n, n) * 0.1
    for i in range(n):
        # 因果掩码
        attn[i, i + 1:] = 0
        # 自注意力：同义词互相注意
        for j in range(i + 1):
            if tokens[i] == tokens[j] and i != j:
                attn[i, j] += 0.5
        # "注意力" 后面的词关注 "注意力" 本身
        if i > 8:
            attn[i, 6:9] += 0.3
        attn[i, i] += 0.2

    # 归一化
    for i in range(n):
        s = attn[i].sum()
        if s > 0:
            attn[i] /= s

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=8, rotation=90)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key（被关注的词）")
    ax.set_ylabel("Query（主动关注的词）")
    ax.set_title("自注意力权重：模型在处理每个词时，关注了哪些其他词")

    plt.colorbar(im, ax=ax, label="注意力权重", shrink=0.8)

    # 标注
    ax.annotate("「注意力」被强烈关注", xy=(7, 12), xytext=(14, 14),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

    plt.tight_layout()
    path = os.path.join(save_dir, "attention_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  注意力权重图: {path}")


def plot_loss_interpretation(save_dir="docs"):
    """解读 loss 曲线：每个阶段意味着什么。"""
    os.makedirs(save_dir, exist_ok=True)

    loss_path = os.path.join("checkpoints", "loss_history.json")
    if os.path.exists(loss_path):
        import json
        with open(loss_path) as f:
            data = json.load(f)
        train_losses = data["train_losses"]
        eval_interval = data["eval_interval"]
        val_losses = data["val_losses"]
    else:
        # 模拟一条典型 loss 曲线
        steps = 3000
        train_losses = [2.5 * np.exp(-i / 300) + 0.02 + np.random.rand() * 0.1 for i in range(steps)]
        eval_interval = 50
        val_steps = list(range(eval_interval, steps + 1, eval_interval))
        val_losses = [2.3 * np.exp(-s / 350) + 0.04 + np.random.rand() * 0.05 for s in val_steps]

    fig, ax = plt.subplots(figsize=(12, 6))

    steps_train = list(range(1, len(train_losses) + 1))
    ax.plot(steps_train, train_losses, alpha=0.3, label="训练 Loss", color="blue")
    ax.plot(steps_train, [sum(train_losses[max(0, i - 10):i + 1]) / min(i + 1, 10) for i in range(len(train_losses))],
            color="blue", linewidth=2, label="训练 Loss（平滑）")

    val_steps = list(range(eval_interval, eval_interval * (len(val_losses) + 1), eval_interval))
    ax.plot(val_steps, val_losses, "o-", label="验证 Loss", color="orange", markersize=3)

    # 阶段标注
    ax.axvspan(0, 100, alpha=0.1, color="red", label="Warmup 阶段")
    ax.axvspan(100, 1000, alpha=0.1, color="yellow", label="快速下降")
    ax.axvspan(1000, len(train_losses), alpha=0.1, color="green", label="精细收敛")

    # 标注关键点
    ax.annotate("模型开始「理解」数据模式\nloss 快速下降",
                xy=(300, 1.0), fontsize=9, color="darkblue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.annotate("收敛：模型已经\n学到了能学的全部",
                xy=(2000, val_losses[-1] if val_losses else 0.05), fontsize=9, color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    ax.set_xlabel("训练步数")
    ax.set_ylabel("Loss（越小越好）")
    ax.set_title("Loss 曲线解读：训练过程的「体检报告」")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_interpretation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss 曲线解读图: {path}")


def plot_model_structure(save_dir="docs"):
    """可视化模型结构：数据从输入到输出的流程。"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 绘制流程
    blocks = [
        (5, 9.0, "输入：\"什么是注意力机制？\"", "lightblue"),
        (5, 7.8, "分词器 → Token IDs", "lightyellow"),
        (5, 6.6, "词嵌入 (Embedding)", "lightgreen"),
        (5, 5.4, "×4 层 Transformer Block", "lightsalmon"),
        (5, 4.2, "RMSNorm 归一化", "lightyellow"),
        (5, 3.0, "LM Head → 预测下一个词的概率", "lightgreen"),
        (5, 1.8, "输出：\"注意力机制通过计算...\"", "lightblue"),
    ]

    for x, y, text, color in blocks:
        box = mpatches.FancyBboxPatch((x - 2.5, y - 0.4), 5, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor="gray")
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=11)

    # 箭头
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(5, blocks[i + 1][1] + 0.45),
                    xytext=(5, blocks[i][1] - 0.45),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    # Transformer Block 展开
    sub_blocks = [
        (8.5, 6.6, "RMSNorm", 0.6),
        (8.5, 5.9, "GQA 注意力", 0.6),
        (8.5, 5.2, "残差连接 +", 0.6),
        (8.5, 4.5, "RMSNorm", 0.6),
        (8.5, 3.8, "SwiGLU 前馈", 0.6),
    ]
    # 连接线
    ax.plot([7.3, 8.0], [5.4, 5.4], "k--", alpha=0.4)
    ax.text(7.7, 5.6, "展开", fontsize=9, color="gray")

    for x, y, text, w in sub_blocks:
        box = mpatches.FancyBboxPatch((x - w, y - 0.25), w * 2, 0.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor="mistyrose", edgecolor="gray", alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)

    for i in range(len(sub_blocks) - 1):
        ax.annotate("", xy=(8.5, sub_blocks[i + 1][1] + 0.3),
                    xytext=(8.5, sub_blocks[i][1] - 0.3),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    ax.set_title("GPT 模型结构：从输入到输出的完整流程", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "model_structure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  模型结构图: {path}")


def plot_real_attention(ckpt_path="checkpoints/best.pt", prompt="什么是注意力机制？", save_dir="docs"):
    """从训练好的模型中提取真实注意力权重并可视化。"""
    import torch
    from src.model import GPT, rope as apply_rope
    from src.tokenizer import load_tokenizer

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"  跳过真实注意力可视化：未找到模型 {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    tok = load_tokenizer(
        cfg.get("tokenizer", {}).get("type", "byte"),
        cfg.get("tokenizer", {}).get("path"),
    )
    model = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=0.0,
        n_kv_head=cfg["model"].get("n_kv_head"),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prefix = [tok.bos_id] + tok.encode("用户:" + prompt + "\n助手:", add_special_tokens=False)
    x = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)

    captured_weights = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            h = input[0]
            B, T, C = h.shape
            q = module.wq(h).view(B, T, module.n_head, module.head_dim)
            k = module.wk(h).view(B, T, module.n_kv_head, module.head_dim)
            q, k = apply_rope(q, k, T, module.head_dim, h.device)
            k = module._repeat_kv(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * (module.head_dim ** -0.5)
            mask = torch.tril(torch.ones(T, T, device=h.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            captured_weights.append(weights[0].detach().numpy())
        return hook

    handles = []
    original_flash = []
    for i, block in enumerate(model.blocks):
        handles.append(block.attn.register_forward_hook(make_hook(i)))
        original_flash.append(block.attn.use_flash)
        block.attn.use_flash = False

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()
    for block, flash in zip(model.blocks, original_flash):
        block.attn.use_flash = flash

    tokens = []
    for tid in prefix:
        try:
            t = tok.decode([tid])
            tokens.append(t if t else f"<{tid}>")
        except Exception:
            tokens.append(f"<{tid}>")

    n_layers = len(captured_weights)

    # 每层单独图
    for layer_idx, weights in enumerate(captured_weights):
        avg = weights.mean(axis=0)

        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6), max(6, len(tokens) * 0.5)))
        im = ax.imshow(avg, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=7, rotation=90)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_xlabel("Key（被关注的词）")
        ax.set_ylabel("Query（主动关注的词）")
        ax.set_title(f"第 {layer_idx + 1} 层注意力权重（各注意力头平均）\n输入：{prompt}")
        plt.colorbar(im, ax=ax, label="注意力权重", shrink=0.8)
        plt.tight_layout()
        path = os.path.join(save_dir, f"real_attention_layer{layer_idx + 1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  第 {layer_idx + 1} 层注意力: {path}")

    # 总览图
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    for layer_idx, (ax, weights) in enumerate(zip(axes, captured_weights)):
        avg = weights.mean(axis=0)
        ax.imshow(avg, cmap="Blues", vmin=0)
        ax.set_title(f"第 {layer_idx + 1} 层")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=5, rotation=90)
        ax.set_yticklabels(tokens, fontsize=5)
    fig.suptitle(f"真实注意力权重（模型处理「{prompt}」时的关注模式）", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "real_attention_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  注意力总览: {path}")


def plot_tokenizer(text="注意力机制通过计算相关性分配权重", tok_path="tokenizer/tokenizer.json", save_dir="docs"):
    """可视化 BPE 分词：展示文本如何被切分成 token。"""
    from src.tokenizer import load_tokenizer

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(tok_path):
        print(f"  跳过分词器可视化：未找到分词器 {tok_path}")
        return

    tok = load_tokenizer("hf_tokenizers", tok_path)
    ids = tok.encode(text, add_special_tokens=False)
    token_pairs = []
    for tid in ids:
        try:
            t = tok.decode([tid])
            token_pairs.append((t if t else f"<{tid}>", tid))
        except Exception:
            token_pairs.append((f"<{tid}>", tid))

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(14, 5), gridspec_kw={"height_ratios": [1, 2, 1.2]},
    )

    # 原始文本
    ax_top.axis("off")
    ax_top.text(0.5, 0.5, f"原始文本：{text}\n共 {len(text)} 个字符", transform=ax_top.transAxes,
                fontsize=13, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))

    # Token 色块
    ax_mid.axis("off")
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(token_pairs), 1)))
    x_pos = 0
    for i, (token_text, token_id) in enumerate(token_pairs):
        width = max(len(token_text) * 0.45, 1.2)
        rect = mpatches.FancyBboxPatch(
            (x_pos, 0.3), width, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=colors[i % len(colors)], edgecolor="gray", linewidth=0.8,
        )
        ax_mid.add_patch(rect)
        ax_mid.text(x_pos + width / 2, 0.55, token_text,
                    ha="center", va="center", fontsize=11)
        ax_mid.text(x_pos + width / 2, 0.15, f"ID:{token_id}",
                    ha="center", va="center", fontsize=7, color="gray")
        x_pos += width + 0.15
    ax_mid.set_xlim(-0.3, x_pos)
    ax_mid.set_ylim(-0.05, 1.0)
    ax_mid.set_title(f"BPE 分词结果：{len(token_pairs)} 个 token", fontsize=11)

    # 统计信息
    ax_bot.axis("off")
    info = "  ".join(f"[{t}]→{tid}" for t, tid in token_pairs)
    info_text = f"Token 映射：{info}"
    if len(info_text) > 200:
        info_text = info_text[:200] + "..."
    ax_bot.text(0.5, 0.5, info_text, transform=ax_bot.transAxes,
                fontsize=8, ha="center", va="center", color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke"))

    fig.suptitle("BPE 分词器可视化：文本如何变成模型的输入", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "tokenizer_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  分词器可视化: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="核心概念可视化")
    ap.add_argument("--only", choices=["causal", "attention", "loss", "structure", "real_attention", "tokenizer"],
                    help="只生成指定的图")
    ap.add_argument("--dir", default="docs", help="输出目录")
    ap.add_argument("--ckpt", default="checkpoints/best.pt", help="模型路径（真实注意力可视化用）")
    ap.add_argument("--prompt", default="什么是注意力机制？", help="输入文本（可视化用）")
    args = ap.parse_args()

    print("生成可视化图表...")
    if not args.only or args.only == "causal":
        plot_causal_mask(save_dir=args.dir)
    if not args.only or args.only == "attention":
        plot_attention_weights(save_dir=args.dir)
    if not args.only or args.only == "loss":
        plot_loss_interpretation(save_dir=args.dir)
    if not args.only or args.only == "structure":
        plot_model_structure(save_dir=args.dir)
    if args.only == "real_attention":
        plot_real_attention(ckpt_path=args.ckpt, prompt=args.prompt, save_dir=args.dir)
    if args.only == "tokenizer":
        plot_tokenizer(text=args.prompt, save_dir=args.dir)
    print("完成！图表保存在 docs/ 目录。")
