# 房价预测 (House Price Prediction)

基于 PyTorch 的房价预测机器学习项目。

## 技术栈

- **Python** 3.10+
- **PyTorch** - 深度学习框架
- **Pandas** - 数据处理
- **Scikit-learn** - 数据预处理与评估
- **Seaborn** - 数据可视化

## 项目结构

```
.
├── data/
│   ├── raw/           # 原始数据 (如 train.csv, test.csv)
│   └── processed/     # 预处理后的数据
├── src/
│   ├── data_loader.py # 数据加载与预处理
│   ├── model.py       # 模型定义
│   ├── train.py       # 训练脚本
│   └── utils.py       # 工具函数
├── notebooks/
│   └── 01_eda.ipynb   # 探索性数据分析
├── tests/             # 单元测试
├── models/            # 保存的模型 (训练后生成)
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 `train.csv` 放入 `data/raw/` 目录。若无真实数据，可先运行 `notebooks/01_eda.ipynb` 生成示例数据。

### 3. 探索性分析

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. 训练模型

```bash
python -m src.train --epochs 100 --batch_size 32 --lr 1e-3
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--hidden_dims` | 64 32 | 隐藏层维度 |
| `--val_ratio` | 0.2 | 验证集比例 |

### 5. 运行测试

```bash
pytest tests/ -v
```

## 开发规范

- 优先使用 NumPy/PyTorch 向量化操作
- 所有函数需包含类型提示
- 路径处理使用 `pathlib`
- 模型定义统一在 `src/model.py`
- 训练脚本支持 argparse 命令行参数

## License

MIT
