#!/bin/bash

# スクリプト自身が置かれているディレクトリ（＝app_labGuide 直下）の絶対パスを取得
PROJECT_ROOT=$(cd $(dirname $0); pwd)

echo "=== 1. ディレクトリの準備 (Root: $PROJECT_ROOT) ==="
# PROJECT_ROOT を起点に作成するため、どこから実行しても位置がズレない
mkdir -p "$PROJECT_ROOT/backend/data/raw"
mkdir -p "$PROJECT_ROOT/tools"

echo "=== 2. Cloudflaredのインストール ==="
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -P "$PROJECT_ROOT/backend"
dpkg -i "$PROJECT_ROOT/backend/cloudflared-linux-amd64.deb"

echo "=== 3. VOICEVOXエンジンの導入 ==="
# p7zip-full（7zファイルの解凍ソフト）を追加でインストール
apt-get update && apt-get install -y libsndfile1 p7zip-full

cd "$PROJECT_ROOT/tools"
# 既に解凍済みのフォルダ(linux-cpu-x64)がなければダウンロードと解凍を実行
if [ ! -d "linux-cpu-x64" ]; then
    # 公式の最新版リリース（0.25.1）を直接ダウンロード
    wget -q https://github.com/VOICEVOX/voicevox_engine/releases/download/0.25.1/voicevox_engine-linux-cpu-x64-0.25.1.7z.001

    # 解凍を実行（これで linux-cpu-x64 というフォルダが作られます）
    7z x voicevox_engine-linux-cpu-x64-0.25.1.7z.001

    # ゴミになる圧縮ファイルを削除
    rm voicevox_engine-linux-cpu-x64-0.25.1.7z.001
else
    echo "VOICEVOXエンジンは既に存在します。スキップします。"
fi

echo "=== セットアップ完了！ ==="