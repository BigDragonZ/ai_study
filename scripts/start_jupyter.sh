#!/usr/bin/env bash
# 在项目内启动 Jupyter，避免写入 ~/Library/Jupyter 导致 PermissionError
# 用法: ./scripts/start_jupyter.sh  或  bash scripts/start_jupyter.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# 全部放到项目 .jupyter 下，避免写系统目录
export JUPYTER_DATA_DIR="$ROOT/.jupyter"
export JUPYTER_CONFIG_DIR="$ROOT/.jupyter"
export JUPYTER_RUNTIME_DIR="$ROOT/.jupyter/runtime"

mkdir -p "$JUPYTER_RUNTIME_DIR"

# 使用项目 .venv
if [ -d "$ROOT/.venv" ]; then
  source "$ROOT/.venv/bin/activate"
fi

echo "JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR"
echo "启动 Jupyter Notebook（不自动打开浏览器）..."
exec python -m jupyter notebook --no-browser
