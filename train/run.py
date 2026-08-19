#!/usr/bin/env python3
"""一键启动：构建分词器 → 训练 → 验收 → 启动 Web Demo。"""

import argparse
import os
import subprocess
import sys


def run(cmd: str, desc: str) -> None:
    """执行 shell 命令，失败时终止整个流程；Ctrl+C 干净退出。

    Args:
        cmd: 要执行的命令。
        desc: 步骤描述（用于打印）。
    """
    print(f"\n{'=' * 50}")
    print(f"步骤: {desc}")
    print(f"命令: {cmd}")
    print(f"{'=' * 50}")
    try:
        result = subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        # Ctrl+C 同时发给父进程与子进程：静默退出，不打印 traceback
        print(f"\n✓ {desc} 已停止")
        sys.exit(0)
    if result.returncode < 0:
        # 子进程被信号终止（Ctrl+C 的 SIGINT），视为正常停止
        print(f"\n✓ {desc} 已停止")
        sys.exit(0)
    if result.returncode != 0:
        print(f"错误: {desc} 失败 (退出码 {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    """命令行入口：按需构建分词器、训练并启动 Web Demo。"""
    ap = argparse.ArgumentParser(description="GPT Teacher 一键启动")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="训练设备 (默认: auto)")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练，直接启动 Web Demo")
    ap.add_argument("--skip-web", action="store_true", help="训练完成后不启动 Web Demo")
    ap.add_argument("--config", default="train/config.yml", help="配置文件路径")
    args = ap.parse_args()

    prefix = sys.executable  # 当前 Python 解释器

    if not args.skip_train:
        # 1. 构建分词器（如果不存在）
        if not os.path.exists("train/tokenizer.json"):
            run(f"{prefix} -m core.build_tokenizer", "构建中文分词器")
        else:
            print("\n分词器已存在，跳过构建")

        # 2. 训练
        run(f"{prefix} -m train.train --device {args.device} --config {args.config}", "训练模型")

    # 3. 启动 Web Demo
    if not args.skip_web:
        print(f"\n{'=' * 50}")
        print("启动 Web Demo...")
        print("打开浏览器访问: http://127.0.0.1:7860")
        print("按 Ctrl+C 停止")
        print(f"{'=' * 50}")
        run(f"{prefix} -m train.web_demo", "Web Demo")


if __name__ == "__main__":
    main()
