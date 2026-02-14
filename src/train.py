"""训练脚本 - 支持命令行参数."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import (
    load_raw_data,
    preprocess_data,
    get_train_val_split,
    HousePriceDataset,
)
from src.model import MLPRegressor
from src.utils import set_seed, get_project_root


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="房价预测模型训练")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据目录路径，默认为项目根目录下的 data/",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32], help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="模型保存路径",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """评估模型."""
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item()
    return total_loss / len(loader)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir) if args.data_dir else get_project_root() / "data"
    if not (data_dir / "raw" / "train.csv").exists():
        print("提示: 请将 train.csv 放入 data/raw/ 目录后再运行训练。")
        print("可使用 notebooks/01_eda.ipynb 生成示例数据。")
        return

    df = load_raw_data(data_dir)
    X, y = preprocess_data(df)
    X_train, X_val, y_train, y_val = get_train_val_split(
        X, y, test_size=args.val_ratio, random_state=args.seed
    )

    train_dataset = HousePriceDataset(X_train, y_train)
    val_dataset = HousePriceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    save_path = Path(args.save_path) if args.save_path else get_project_root() / "models" / "best_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")


if __name__ == "__main__":
    main()
