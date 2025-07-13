#!/bin/bash

echo "📄 XeLaTeX による日本語PDFをコンパイル中..."

# latexmk がインストールされているか確認
if ! command -v latexmk >/dev/null 2>&1; then
  echo "❌ latexmk が見つかりません。MacTeX をインストールしてください。"
  exit 1
fi

# 引数（コンパイルしたい .tex ファイル）を取得
INPUT_TEX="$1"

# 引数が指定されていない場合はエラー
if [ -z "$INPUT_TEX" ]; then
  echo "❌ TeX ファイルを指定してください。例： ./open_tex.sh papers/x.tex"
  exit 1
fi

# ファイル名（拡張子なし）を抽出
FILENAME=$(basename "$INPUT_TEX" .tex)

# 出力ディレクトリを指定：papers/pdf/ファイル名
OUTPUT_DIR="papers/pdf/$FILENAME"
mkdir -p "$OUTPUT_DIR"

# PDF 出力先パス
PDF_PATH="$OUTPUT_DIR/$FILENAME.pdf"

# コンパイル実行（エラーが出ても続行）
latexmk -xelatex -f -interaction=nonstopmode -shell-escape -outdir="$OUTPUT_DIR" "$INPUT_TEX" || echo "⚠️ latexmk は警告を出しましたが、続行します..."

# PDF が生成されたか確認して開く
if [ -f "$PDF_PATH" ]; then
  echo "✅ PDF の生成に成功しました。開きます..."
  open "$PDF_PATH"
else
  echo "❌ PDF が生成されませんでした。ログファイルを確認してください。"
fi
