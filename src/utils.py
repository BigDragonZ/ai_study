"""通用工具函数."""

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子，保证实验可复现.

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    """获取项目根目录（src 的父目录）."""
    return Path(__file__).resolve().parent.parent
