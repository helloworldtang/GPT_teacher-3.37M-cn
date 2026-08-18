"""通用工具：随机种子、目录、CPU 线程数。"""

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
