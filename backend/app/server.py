# backend/app/server.py
"""
Flask サーバ（フロントエンドの配信と API）。

debug_inference.ipynb のセルにあったルートをそのまま移したもの。フロントエンドとの
約束事（/chat の返り値のキー、/synthesize、/logs、/reset）は変更していない。

変更点:
  - パスを backend/ 基準の絶対パスで解決（モジュール化してもフロントが見つかるように）
  - /chat と /reset をロックで直列化（複数人が同時に使っても会話履歴・感情状態・
    ステアリングのフックが混ざらないように）。★会話状態はサーバ全体で1つなので、
    同時に使うと全員が同じ会話を共有する点は変わらない（単一ユーザー想定のデモ）。
  - 音声ファイル名に連番を付けて、同じ秒に2回合成しても上書きしないようにした
"""

from __future__ import annotations

import glob
import os
import queue
import uuid

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")

# 音声の保存先（VoicevoxClient の保存先と同じ場所）
AUDIO_DIR = os.path.join(FRONTEND_DIR, "assets", "audio")
KEEP_AUDIO = 3   # 普段残しておく音声の数（再生中のファイルを消さないため）


def _clear_audio(keep: int = 0) -> None:
    """音声ファイルを古い順に削除し、新しいものを keep 個だけ残す。

    音声には会話の内容が含まれ、URL を知っていれば誰でも取得できるので、
    必要な分以上は残さない。
    """
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")), key=os.path.getmtime)
    for f in files[:max(0, len(files) - keep)]:
        try:
            os.remove(f)
        except OSError:
            pass   # 再生中などで消せなくても、次の機会に消える


def create_app(manager, voicevox=None, frontend_dir: str = FRONTEND_DIR,
               default_rag: bool = False) -> Flask:
    """ロード済みの CharacterAIManager（と VoicevoxClient）を使う Flask アプリを作る。"""
    app = Flask(__name__, static_folder=os.path.join(frontend_dir, "static"))
    CORS(app)
    log_queue: "queue.Queue[str]" = queue.Queue()
    _clear_audio(keep=0)   # 起動時に、前回の音声をすべて削除する

    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/assets/<path:filename>")
    def serve_assets(filename):
        return send_from_directory(os.path.join(frontend_dir, "assets"), filename)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.json or {}
        user_text = data.get("text", "")
        condition = data.get("condition", "steering")   # baseline / prompt / steering
        print(f"\n[Request] {user_text}  (condition={condition})")
        with manager.lock:
            res = manager.generate_reply(user_text, rag_flag=data.get("rag", default_rag),
                                         condition=condition, use_history=True)
        return jsonify({
            "reply": res["reply"],
            "emotion": res["complex_emotion"],
            "emotion_state": res.get("emotion_state", []),
            "condition": res["condition"],
            "target_emotion": res.get("target_emotion"),
            "applied_alpha": res.get("applied_alpha", 0.0),
            "emotion_alpha": res.get("emotion_alpha", []),   # α換算の8感情（チャート用）
            "alpha_max": res.get("alpha_max", 0.8),          # チャートの目盛りの上限
        })

    @app.route("/synthesize", methods=["POST"])
    def synthesize():
        data = request.json or {}
        text = data.get("text", "")
        if not text or voicevox is None:
            return jsonify({"audio_url": None})
        speaker_id = data.get("speaker_id", 8)   # 既定: 春日部つむぎ
        # 推測できないファイル名にする（他人の音声の URL を当てられないように）
        filename = f"response_{uuid.uuid4().hex}.wav"
        url = voicevox.generate_audio(text, speaker_id=speaker_id, output_filename=filename)
        _clear_audio(keep=KEEP_AUDIO)            # 古い音声を削除する
        return jsonify({"audio_url": url})

    @app.route("/logs")
    def stream_logs():
        def generate():
            while True:
                try:
                    yield f"data: {log_queue.get(timeout=1)}\n\n"
                except queue.Empty:
                    yield "data: \n\n"   # keepalive
        return Response(generate(), mimetype="text/event-stream")

    @app.route("/reset", methods=["POST"])
    def reset_emotion():
        with manager.lock:
            manager.reset_conversation()   # 感情と履歴の両方をリセット
        _clear_audio(keep=0)               # 会話の音声も削除する
        return jsonify({"status": "ok"})

    app.log_queue = log_queue   # 必要なら app.log_queue.put("...") でUIに流せる
    return app
