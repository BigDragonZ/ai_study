"""数据加载与预处理模块."""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


class HousePriceDataset(Dataset):
    """房价预测数据集，兼容 PyTorch DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: 特征矩阵，shape (n_samples, n_features)
            y: 标签向量，shape (n_samples,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    从 data/raw 目录加载原始 CSV 数据.

    Args:
        data_dir: 数据目录路径

    Returns:
        加载的 DataFrame
    """
    csv_path = data_dir / "raw" / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")
    return pd.read_csv(csv_path)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    数据预处理：填充缺失值、编码分类变量等.

    Args:
        df: 原始 DataFrame

    Returns:
        (特征 DataFrame, 标签 Series)
    """
    # 示例：数值列填充中位数，分类列填充众数
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")

    # 假设最后一列为目标变量（可根据实际数据调整）
    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 数值化：仅保留数值列或进行 one-hot 编码
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def get_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和验证集，并进行标准化.

    Args:
        X: 特征
        y: 标签
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        X_train, X_val, y_train, y_val (numpy arrays)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, y_train.values, y_val.values
