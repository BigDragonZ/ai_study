"""utils 模块单元测试."""

from pathlib import Path

import numpy as np
import torch

from src.utils import set_seed, get_project_root


def test_set_seed() -> None:
    """测试 set_seed 保证可复现性."""
    set_seed(42)
    a = np.random.rand(3)
    b = torch.rand(3)
    set_seed(42)
    a2 = np.random.rand(3)
    b2 = torch.rand(3)
    np.testing.assert_array_almost_equal(a, a2)
    torch.testing.assert_close(b, b2)


def test_get_project_root() -> None:
    """测试项目根目录."""
    root = get_project_root()
    assert root.is_dir()
    assert (root / "src").exists()
    assert (root / "data").exists()
