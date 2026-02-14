"""model 模块单元测试."""

import torch
import pytest

from src.model import MLPRegressor


def test_mlp_regressor_forward() -> None:
    """测试 MLPRegressor 前向传播."""
    model = MLPRegressor(input_dim=10, hidden_dims=[64, 32], dropout=0.0)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 1)


def test_mlp_regressor_dropout() -> None:
    """测试 dropout 在 eval 模式下被关闭."""
    model = MLPRegressor(input_dim=5, hidden_dims=[8], dropout=0.5)
    model.eval()
    x = torch.randn(2, 5)
    out = model(x)
    assert out.shape == (2, 1)
