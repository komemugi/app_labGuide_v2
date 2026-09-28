# backend/run.py
"""
アプリを1コマンドで起動する。

    cd backend
    python run.py                 # http://localhost:8000
    python run.py --tunnel        # cloudflared で外部公開URLを発行（cloudflared が必要）
    python run.py --no-voice      # VOICEVOX を使わない
    python run.py --rag           # RAG（学内文書の検索）を使う（インデックスが必要）

ノートブック（debug_inference.ipynb）から起動する場合も、中身は同じ関数を使う。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)                      # VOICEVOXの音声保存先など、相対パスを backend/ 基準に揃える
sys.path.insert(0, BACKEND_DIR)

# ============================================================================
# 設定（研究成果物と一致させること）
# ============================================================================
MODEL_ID = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
RESEARCH_MODEL_KEY = "llama-3.1-swallow-8b-instruct-v0.5"
TARGET_LAYER = 21            # hidden_states のインデックス（研究と同じ規約）
ASSETS_ROOT = os.path.join(BACKEND_DIR, "data", "processed", "steering_assets")
ALPHA_CAP_VARIANT = "rf"     # 拒絶方向除去ありの条件で決めた上限を使う
DEFAULT_ALPHA_CAP = 0.6      # chatbot_alpha_cap.json が読めない場合のフォールバック
# 会話用の安全係数。実際の α の上限 = min(感情ごとの上限, モデル上限 0.8) × この値。
#   0.75 → 全感情 0.6 が上限。会話が崩れるなら下げる（0.5 → 0.4）、
#   感情が弱すぎると感じるなら上げる（1.0 → 0.8。研究で確かめた上限そのもの）
CHAT_SAFETY = 1.0
ABLATION_MODE = "steering_only"   # 拒絶方向の除去: steering_only / always / off

# プルチックの環順（EmotionDynamics の相関行列がこの順序を前提にしている）
EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
VOICEVOX_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "tools", "linux-cpu-x64")

# 8bit量子化で読み込むか（GPUメモリが足りずモデルが載らないときの最終手段）
# 研究の成果物（avg_norm・α上限・感情プローブ）は bf16 で作ったものなので、
#     量子化すると隠れ状態の値が少し変わり、研究で確かめた条件から外れる。
#     使う場合は、会話が崩れないか・感情推定がおかしくないかを確認すること。
QUANTIZATION_8BIT = False

# RAG（学内文書の検索）を使うか。
#   True にしても、data/processed/rag_store/ にインデックスが無ければ使われない
#   （インデックスは app/rag/build_index.py で作る）。公開版にはデータを含めていない。
USE_RAG = False


def build_manager(engine=None, use_rag: bool = USE_RAG):
    """LLM とマネージャ（感情推定・内部状態・ステアリング一式）を用意する。

    engine を渡せばそれを使い回す（ノートブックでモデルを二重にロードしないため）。
    """
    from app.core.llm_engine import LLMEngine
    from app.manager import CharacterAIManager

    if engine is None:
        # 量子化するかは QUANTIZATION_8BIT で決める（既定は False。理由は設定欄のコメントを参照）
        engine = LLMEngine(model_id=MODEL_ID, quantization_8bit=QUANTIZATION_8BIT)
    return CharacterAIManager(
        emotions=EMOTIONS, llm_engine=engine, assets_root=ASSETS_ROOT,
        research_model_key=RESEARCH_MODEL_KEY, target_layer=TARGET_LAYER,
        variant=ALPHA_CAP_VARIANT, default_alpha_cap=DEFAULT_ALPHA_CAP,
                max_strength=30.0, chat_safety=CHAT_SAFETY, ablation_mode=ABLATION_MODE,
                enable_rag=use_rag,
                )


def start_voicevox(engine_dir: str = VOICEVOX_DIR):
    """VOICEVOXエンジンを起動してクライアントを返す。エンジンが無ければ None。"""
    from app.api.voicevox_client import VoicevoxClient
    run_path = os.path.join(engine_dir, "run")
    if not os.path.exists(run_path):
        print(f"[run] VOICEVOXエンジンが見つかりません（{run_path}）。音声なしで起動します。"
              f"setup.sh でダウンロードできます。")
        return None
    subprocess.Popen(["./run"], cwd=engine_dir,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[run] VOICEVOXエンジンを起動しました（音声はVOICEVOX:春日部つむぎ）")
    return VoicevoxClient()


def start_server(manager, voicevox=None, port: int = 8000, tunnel: bool = False,
                 background: bool = False, rag: bool = USE_RAG):
    """Flask を起動する。background=True ならスレッドで起動して即座に戻る（ノートブック用）。"""
    from app.server import create_app
    app = create_app(manager, voicevox, default_rag=rag)

    def _serve():
        app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)

    if tunnel:
        threading.Thread(target=_serve, daemon=True).start()
        _open_tunnel(port)
        if not background:
            threading.Event().wait()     # Ctrl+C まで待つ
    elif background:
        threading.Thread(target=_serve, daemon=True).start()
        print(f"[run] http://localhost:{port} で起動しました")
    else:
        _serve()
    return app


def _open_tunnel(port: int):
    print("[run] cloudflared tunnel を起動中...")
    proc = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            print(f"\n[run] 公開URL: {m.group(0)}\n（発行から数分はアクセスできないことがあります）")
            break
    return proc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tunnel", action="store_true", help="cloudflared で外部公開する")
    p.add_argument("--no-voice", action="store_true", help="VOICEVOXを使わない")
    p.add_argument("--rag", action="store_true", help="RAG（学内文書の検索）を使う")
    a = p.parse_args()
    
    use_rag = USE_RAG or a.rag
    mgr = build_manager(use_rag=use_rag)
    vv = None if a.no_voice else start_voicevox()
    start_server(mgr, vv, port=a.port, tunnel=a.tunnel, rag=use_rag)
