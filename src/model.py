"""模型定义模块 - 所有模型定义均在此文件中."""

from typing import List

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    多层感知机回归模型，用于房价预测.

    前向传播: y = f(W_n(...f(W_2 f(W_1 x + b_1) + b_2)...) + b_n)
    其中 f 为 ReLU 激活函数，最后一层无激活（线性输出）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 各隐藏层神经元数量
            dropout: Dropout 比例
        """
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，shape (batch_size, input_dim)

        Returns:
            预测值，shape (batch_size, 1)
        """
        h = self.hidden(x)
        return self.output(h)
