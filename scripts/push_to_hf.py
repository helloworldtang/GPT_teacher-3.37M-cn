#!/usr/bin/env python3
"""将模型推送到 Hugging Face Hub。

用法:
    # 先登录
    huggingface-cli login

    # 推送到你的仓库
    python scripts/push_to_hf.py --repo helloworldtang/GPT-Teacher-3.37M-cn

    # 推送到组织仓库
    python scripts/push_to_hf.py --repo your-org/GPT-Teacher-3.37M-cn
"""
import argparse
import os
import json
import shutil

from huggingface_hub import HfApi, create_repo


MODEL_CARD = """\
# GPT Teacher 3.37M 中文教学模型

一个 3.16M 参数的中文 GPT 教学模型，基于 Llama 风格架构（GQA + SwiGLU + RMSNorm + RoPE）。

## 模型信息

- **参数量**: 3.16M
- **架构**: Decoder-only Transformer (Llama-style)
- **词表大小**: ~1500 (BPE)
- **序列长度**: 128
- **层数**: 4
- **注意力头**: 4 (GQA, KV头=2)
- **嵌入维度**: 256

## 使用方法

```bash
# 克隆项目
git clone https://github.com/helloworldtang/GPT_teacher-3.37M-cn.git
cd GPT_teacher-3.37M-cn

# 安装依赖
uv sync

# 从 Hugging Face 下载模型到 checkpoints/
python scripts/download_from_hf.py --repo {repo_id}

# 运行验收测试
uv run python -m src.evaluate --ckpt checkpoints/best.pt

# 启动 Web Demo
uv run python run.py --skip-train
```

## 训练细节

- **训练步数**: 5000
- **优化器**: AdamW (lr=0.001, weight_decay=0.1)
- **学习率调度**: 线性预热 + 余弦退火
- **早停**: patience=15
- **验收通过率**: 6/6 (100%)

## 示例输出

```
Q: 什么是注意力机制？
A: 注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。

Q: RoPE 是什么？
A: RoPE 是旋转位置编码，通过复数旋转注入相对位置信息。

Q: 蒸馏水和纯水有什么区别？
A: 蒸馏水通过蒸馏冷凝制得，纯水杂质极低。
```

## 限制

这是一个 3M 参数的教学模型，仅用于学习 GPT 训练流程。它：
- 只能回答训练数据中涵盖的问题
- 3M 参数能完美记住 4-7 token 的答案，18 token 以上开始力不从心
- 不具备泛化能力，不应用于生产场景

## 许可证

MIT
"""


def push_to_hub(repo_id: str, checkpoint_path: str = "checkpoints/best.pt",
                tokenizer_path: str = "tokenizer/tokenizer.json",
                config_path: str = "config/config.yml"):
    """推送模型到 Hugging Face Hub。"""
    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        print("请先运行训练: uv run python -m src.train")
        return

    print(f"推送到 Hugging Face Hub: {repo_id}")

    # 创建仓库
    create_repo(repo_id, exist_ok=True, repo_type="model")
    api = HfApi()

    # 推送模型文件
    files_to_upload = []

    # 模型权重
    files_to_upload.append(("checkpoints/best.pt", checkpoint_path))

    # 分词器
    if os.path.exists(tokenizer_path):
        files_to_upload.append(("tokenizer/tokenizer.json", tokenizer_path))

    # 配置
    if os.path.exists(config_path):
        files_to_upload.append(("config/config.yml", config_path))

    # 训练数据（用于重建分词器）
    for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join("data", name)
        if os.path.exists(path):
            files_to_upload.append((f"data/{name}", path))

    # 数据准备脚本
    prep_path = "data/prepare_data.py"
    if os.path.exists(prep_path):
        files_to_upload.append(("data/prepare_data.py", prep_path))

    # 上传文件
    for hf_path, local_path in files_to_upload:
        print(f"  上传: {local_path} → {hf_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="model",
        )

    # 上传 Model Card
    model_card = MODEL_CARD.format(repo_id=repo_id)
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\n推送完成！")
    print(f"模型页面: https://huggingface.co/{repo_id}")
    print(f"\n用户可以通过以下命令下载:")
    print(f"  python scripts/download_from_hf.py --repo {repo_id}")


def download_from_hub(repo_id: str, output_dir: str = "."):
    """从 Hugging Face Hub 下载模型。"""
    from huggingface_hub import snapshot_download

    print(f"从 Hugging Face Hub 下载: {repo_id}")
    local_dir = snapshot_download(repo_id, repo_type="model", local_dir=output_dir)
    print(f"下载完成: {local_dir}")
    print(f"\n运行验收测试: uv run python -m src.evaluate --ckpt checkpoints/best.pt")
    print(f"启动 Web Demo: uv run python run.py --skip-train")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hugging Face Hub 操作")
    sub = ap.add_subparsers(dest="action")

    # push
    push_p = sub.add_parser("push", help="推送模型到 HF Hub")
    push_p.add_argument("--repo", required=True, help="HF 仓库 ID (如 username/model-name)")
    push_p.add_argument("--ckpt", default="checkpoints/best.pt", help="模型路径")

    # download
    dl_p = sub.add_parser("download", help="从 HF Hub 下载模型")
    dl_p.add_argument("--repo", required=True, help="HF 仓库 ID")
    dl_p.add_argument("--output", default=".", help="下载目录")

    args = ap.parse_args()
    if args.action == "push":
        push_to_hub(args.repo, args.ckpt)
    elif args.action == "download":
        download_from_hub(args.repo, args.output)
    else:
        ap.print_help()
