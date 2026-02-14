"""data_loader 模块单元测试."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data_loader import HousePriceDataset, preprocess_data, get_train_val_split


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """生成示例 DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "area": np.random.uniform(50, 200, n),
        "rooms": np.random.randint(1, 6, n),
        "age": np.random.randint(0, 50, n),
        "price": np.random.uniform(50e4, 500e4, n),
    })


def test_preprocess_data(sample_df: pd.DataFrame) -> None:
    """测试 preprocess_data 输出形状."""
    X, y = preprocess_data(sample_df)
    assert X.shape[0] == sample_df.shape[0]
    assert len(y) == sample_df.shape[0]
    assert X.shape[1] >= 1


def test_get_train_val_split(sample_df: pd.DataFrame) -> None:
    """测试 train/val 划分."""
    X, y = preprocess_data(sample_df)
    X_train, X_val, y_train, y_val = get_train_val_split(
        X, y, test_size=0.2, random_state=42
    )
    assert len(X_train) + len(X_val) == len(X)
    assert X_train.shape[1] == X_val.shape[1]


def test_house_price_dataset() -> None:
    """测试 HousePriceDataset."""
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    ds = HousePriceDataset(X, y)
    assert len(ds) == 10
    xi, yi = ds[0]
    assert xi.shape == (5,)
    assert yi.shape == (1,)
