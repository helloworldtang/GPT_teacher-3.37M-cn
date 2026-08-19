"""通用工具：随机种子、目录、文件指纹、CPU 线程数。"""

import hashlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定全部随机源，保证训练可复现。

    Args:
        seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则递归创建。

    Args:
        path: 目录路径。
    """
    os.makedirs(path, exist_ok=True)


def num_threads() -> int:
    """返回可用 CPU 线程数，探测失败时回退为 1。"""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def file_md5(path: str | None) -> str | None:
    """计算文件 md5 指纹（前 12 位），文件不存在返回 None。

    Args:
        path: 文件路径。

    Returns:
        md5 前 12 位十六进制，或 None。
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]
