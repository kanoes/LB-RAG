#!/usr/bin/env bash

# ワンキー起動スクリプト
# 1. demo/ ディレクトリでローカル仮想環境 (.venv) を作成・有効化
# 2. backend/requirements.txt の依存関係をインストール
# 3. Streamlit フロントエンドを起動

set -euo pipefail

# スクリプトのあるディレクトリ (demo/) に移動
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 仮想環境が存在しない場合は作成
if [[ ! -d ".venv" ]]; then
  echo "[run.sh] 仮想環境 (.venv) を作成中"
  python3 -m venv .venv
fi

# 仮想環境を有効化
source .venv/bin/activate

echo "[run.sh] 依存関係のインストール／アップデート中..."
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

echo "[run.sh] Streamlit アプリを起動中..."
streamlit run frontend/app.py "$@"
