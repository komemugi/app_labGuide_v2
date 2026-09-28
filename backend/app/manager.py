# backend/app/manager.py
"""
CharacterAIManager: 1ターンの処理（感情推定 → 内部状態の更新 → ステアリング付き生成）をまとめる。

処理の流れ（generate_reply の中身）:

    ユーザー発話
      │
      ├─ ① EmotionProbe       : LLM の隠れ状態から、発話の感情（8感情の確率）を読む
      ├─ ② EmotionDynamics    : 前のターンの状態を減衰させ、①を足して内部状態（0〜30）を更新
      ├─ ③ EmotionSteeringController.plan : 一番強い感情の強度を α（0〜上限）に変換
      └─ ④ Steerer + LLMEngine: その感情ベクトルを α で足しながら返答を生成
                                 （同時に拒絶方向を全層で除去）

condition（比較用の条件）:
    "baseline" : 感情制御なし（拒絶方向の除去だけ）
    "prompt"   : 感情をシステムプロンプトの文章で指示する
    "steering" : 感情ベクトルを隠れ状態に足す（本アプリの既定）
    "hybrid"   : prompt と steering の両方
"""

from __future__ import annotations

import contextlib
import os
import pickle
import threading

from app.core.emotion_dynamics import EmotionDynamics
from app.core.emotion_probe import EmotionProbe
from app.core.emotion_scales import EMOTION_SCALES_JA
from app.core.emotion_steering import EmotionSteeringController
from app.core.steerer import Steerer

# backend/ ディレクトリの絶対パス（RAG データの場所を、起動場所に関係なく決めるため）
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SYSTEM_PROMPT = ("あなたは誠実で優秀な日本人のアシスタントです。名前は「紅華フミ」です。"
                         "一人称は「私」です。"
                         "日本語で自然に応答してください。返答は2〜3文の短い会話にしてください。")

class CharacterAIManager:
    def __init__(self, emotions: list, llm_engine, assets_root: str,
                 research_model_key: str, target_layer: int,
                 variant: str = "rf", default_alpha_cap: float = 0.6,
                 max_strength: float = 30.0, chat_safety: float = 1.0,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 dynamics_kwargs: dict = None,
                 rag_dir: str = os.path.join(BACKEND_DIR, "data", "processed", "rag_store"),
                 ablation_mode: str = "steering_only",
                 enable_rag: bool = False,):
        """
        Parameters
        ----------
        emotions           : 感情名のリスト。EmotionDynamics の都合でプルチックの環順にする
        llm_engine         : ロード済みの LLMEngine
        assets_root        : 研究成果物の置き場（steering_vectors / emotion_probe /
                             refusal_vectors / alpha_cap）
        research_model_key : 研究側のモデルキー（例 llama-3.1-swallow-8b-instruct-v0.5）
        target_layer       : ステアリングと感情推定に使う層（hidden_states のインデックス）
        variant            : α上限を決めた条件（"rf" = 拒絶方向を除去した条件）
        default_alpha_cap  : α上限の json が読めないときの上限
        max_strength       : 内部状態の強度の最大値。強度→α の写像にも使う
        chat_safety        : 会話用の安全係数（EmotionSteeringController.cap_for を参照）
        dynamics_kwargs    : EmotionDynamics に渡す追加パラメータ（decay_rate など）
        rag_dir            : RAG のインデックスがあるディレクトリ
        enable_rag         : RAG を使うか（run.py の USE_RAG / --rag から渡される）
        """
        self.engine = llm_engine
        self.target_layer = target_layer
        self.system_prompt = system_prompt
        self.emotions = emotions

        # Flask は複数のリクエストを並行して処理できるが、このクラスは
        # 会話履歴・内部状態・モデルに掛けるフックを1つずつしか持っていない。
        # server.py はこのロックを取ってから generate_reply / reset を呼ぶので、
        # 同時アクセスがあっても1リクエストずつ順番に処理される。
        self.lock = threading.Lock()

        # 拒絶方向の除去をいつ使うか
        #   "steering_only" : 感情ステアリングを掛けるターンだけ（既定。公開デモ向け）
        #   "always"        : 全ターン（研究の評価と同じ条件）
        #   "off"           : 使わない
        # 拒絶方向を除去すると、本来断るべき依頼も断りにくくなるため、必要なときだけ使う。
        self.ablation_mode = ablation_mode

        # ---- 研究成果物の読み込み（起動時に1回だけ）----------------------------
        # 感情ベクトル・avg_norm・拒絶方向・α上限をまとめて読む
        self.ctrl = EmotionSteeringController.from_assets(
            assets_root=assets_root, model_key=research_model_key,
            layer=target_layer, variant=variant,
            default_cap=default_alpha_cap, max_strength=max_strength,
            chat_safety=chat_safety)
        # モデルにフックを掛けてステアリングと拒絶除去を行う係
        self.steerer = Steerer(
            llm_engine.model,
            avg_norms={target_layer: self.ctrl.avg_norm},
            refusal_direction=self.ctrl.refusal_direction,
            ablation_enabled=(self.ctrl.refusal_direction is not None))

        # ---- ① 感情推定（線形プローブ）--------------------------------------
        # 出力は emotions の順（環順）に並べた8感情の確率
        self.classifier = EmotionProbe(
            self.emotions, llm_engine,
            probe_path=os.path.join(assets_root, "emotion_probe",
                                    f"probe_layer{target_layer}.npz"),
            layer=target_layer)

        # ---- ② 内部状態（感情の力学系）--------------------------------------
        # decay_rate        : 1ターンごとに前の状態に掛ける減衰率（0.5 = 半分残す）
        # sensitivity       : 入力に対する反応の強さ（tanh の中に掛かる係数）
        # neutral_threshold : 最大の感情がこの値未満なら「中立」とみなしステアリングしない
        # top_k             : ステアリングに使う感情の数（1 = 一番強い感情だけ。論文と同じ）
        dk = dict(decay_rate=0.5, sensitivity=0.5, neutral_threshold=10.0, top_k=1)
        dk.update(dynamics_kwargs or {})
        self.brain = EmotionDynamics(self.emotions, max_strength=max_strength, **dk)

        # 会話履歴（[{"role": ..., "content": ...}, ...]）と直前の条件
        self.history = []
        self.last_condition = None

        # ---- RAG（有効にしたときだけ、データがあれば埋め込みモデルを読み込む）------------------
        # 埋め込みモデルは数GBあるので、使わないときは読み込まない
        self.rag_index, self.rag_texts, self.embedder = None, None, None
        index_path = os.path.join(rag_dir, "data_index.faiss")
        texts_path = os.path.join(rag_dir, "data_texts.pkl")
        if not enable_rag:
            print("[Manager] RAG は無効です（run.py の USE_RAG または --rag で有効化）。")
        elif os.path.exists(index_path) and os.path.exists(texts_path):
            import faiss
            from sentence_transformers import SentenceTransformer
            self.rag_index = faiss.read_index(index_path)
            with open(texts_path, "rb") as f:
                self.rag_texts = pickle.load(f)
            self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
            print("RAGデータのロード完了")
        else:
            print("[Manager] RAG が有効ですが、インデックスが無いため使えません。")

    # ------------------------------------------------------------------
    def generate_reply(self, user_text, rag_flag=False, condition="steering",
                       force_emotion=None, force_alpha=None, force_level=None,
                       use_history=True, gen_kwargs=None, use_ablation=True):
        """1ターン分の返答を生成する。

        Parameters
        ----------
        user_text     : ユーザーの発話
        rag_flag      : True なら RAG で関連情報を検索して発話の前に付ける
        condition     : "baseline" / "prompt" / "steering" / "hybrid"
        force_emotion : 感情を推定せずに指定する（実験・デバッグ用）
        force_alpha   : force_emotion のときの α を直接指定する
        force_level   : force_emotion のときの強さを 1〜5 で指定する（既定 5）
        use_history   : 会話履歴を使う・更新するか
        gen_kwargs    : LLMEngine.chat に渡す生成パラメータ（temperature など）
        use_ablation  : 拒絶方向の除去を行うか
        """
        rag_info = self.get_rag_info(user_text) if rag_flag else None

        # ==== 感情と強さ（α・レベル）を決める =================================
        if force_emotion is not None:
            # --- 感情を外から指定した場合（推定も内部状態の更新もしない）---
            emo = force_emotion
            level = force_level if force_level is not None else 5
            cap = self.ctrl.cap_for(emo)          # この感情の実効α上限
            # α を直接指定されていなければ、レベル（1〜5）を上限に対する割合として使う
            #   例: レベル5 → α = 上限、レベル3 → α = 上限 × 0.6
            alpha = force_alpha if force_alpha is not None else cap * (level / 5.0)
            label = emo
        else:
            # --- 通常: ①感情推定 → ②内部状態の更新 → ③α に変換 ---
            info = self.determine_emotion(user_text)   # ①② （中で内部状態が更新される）
            plan = self.ctrl.plan(info)                 # ③ 強度 → α
            # 一番 α の大きい感情を1つ選ぶ（中立なら plan["alphas"] が空なので None）
            emo = max(plan["alphas"], key=plan["alphas"].get) if plan["alphas"] else None
            alpha = plan["alphas"].get(emo, 0.0) if emo else 0.0
            # prompt 条件で使う「レベル（1〜5）」を、α が上限の何割かから逆算する
            #   例: α=0.3、上限0.6 → 0.5 × 5 = 2.5 → レベル2（round は偶数丸め）
            cap = self.ctrl.cap_for(emo) if emo else 1.0
            level = max(1, min(5, round(alpha / cap * 5))) if emo else None
            # UI に表示する複合感情ラベル（上位2感情の組み合わせ。例: joy+trust → love）
            label = info["complex_emotion_label"]

        # ==== システムプロンプト（prompt / hybrid 条件だけ感情指示を足す）=========
        sys_p = self.system_prompt
        if condition in ("prompt", "hybrid") and emo and level:
            desc = EMOTION_SCALES_JA[emo][level]      # 例: joy のレベル3 →「はっきりとした幸福と喜び」
            sys_p += (f"\nあなたは今、次の感情状態にあります：「{desc}」"
                      f"（5段階中レベル{level}の強さ）。"
                      f"その感情が言葉選びや語気から自然に伝わるように応答してください。")

        # ==== ユーザーメッセージ（RAG の結果があれば区切って前に付ける）=============
        if rag_info and rag_info != "関連情報なし":
            content = f"[参考情報]\n{rag_info}\n[/参考情報]\n\n{user_text}"
        else:
            content = user_text

        # ==== ④ 生成 =============================================================
        hist = self.history if use_history else None
        gk = gen_kwargs or {}
        # このターンで感情ステアリングを掛けるか
        steering_now = bool(condition in ("steering", "hybrid") and emo and alpha > 0)
        # このターンで拒絶方向を除去するか（ablation_mode に従う）
        if self.ablation_mode == "always":
            ablate = use_ablation
        elif self.ablation_mode == "steering_only":
            ablate = use_ablation and steering_now
        else:
            ablate = False
        # Steerer の設定をこのターンだけ切り替え、終わったら元に戻す
        prev_abl = self.steerer.ablation_enabled
        self.steerer.ablation_enabled = ablate and self.ctrl.refusal_direction is not None
        try:
            if steering_now:
                # 感情ベクトルを α で足す（ablate が True なら拒絶除去も同時に掛かる）
                ctx = self.steerer.steering(layer=self.target_layer,
                                            vectors={emo: self.ctrl.vectors[emo]},
                                            alphas={emo: alpha})
            elif self.steerer.ablation_enabled:
                # ステアリングなしで拒絶除去だけ掛ける（ablation_mode="always" のとき）
                ctx = self.steerer.ablation_only()
            else:
                # 何も掛けない（中立のターン・baseline・prompt）
                ctx = contextlib.nullcontext()
            with ctx:
                reply, tok = self.engine.chat(content, sys_p, history=hist, **gk)
        finally:
            self.steerer.ablation_enabled = prev_abl

        # ==== 会話履歴を更新（直近5往復だけ残す）==================================
        if use_history:
            self.history += [{"role": "user", "content": content},
                             {"role": "assistant", "content": reply}]
            self.history = self.history[-10:]

        # 実際に効いた値だけを返す（steering 系でなければ α は 0、prompt 系でなければレベルは None）
        eff_alpha = alpha if condition in ("steering", "hybrid") else 0.0
        eff_level = level if condition in ("prompt", "hybrid") else None
        self.last_condition = condition

        return {"reply": reply, "condition": condition, "complex_emotion": label,
                "target_emotion": emo, "applied_alpha": eff_alpha,
                "applied_level": eff_level, "emotion_state": self.brain.state.tolist(),
                # UI 表示用: 内部状態を α に換算した値と、その最大値
                "emotion_alpha": self.ctrl.state_to_alphas(self.emotions, self.brain.state),
                "alpha_max": self.ctrl.display_max,
                "token_info": tok}

    def reset_conversation(self):
        """会話履歴と内部状態の両方をリセットする（タイトル画面に戻ったときなど）。"""
        self.history = []
        self.brain.reset()

    # ------------------------------------------------------------------
    def get_rag_info(self, user_text, top_k=4, threshold=0.8):
        """ユーザーの発話に関連する情報を RAG で検索する。

        類似度が threshold 以上の上位 top_k 件をつなげて返す。無ければ "関連情報なし"。
        """
        if self.rag_index is None:
            return "関連情報なし"
        # e5 系の埋め込みモデルは、検索クエリの先頭に "query: " を付ける決まりがある
        q = self.embedder.encode([f"query: {user_text}"], normalize_embeddings=True)
        distances, indices = self.rag_index.search(q, top_k)
        hits = [self.rag_texts[idx] for e, idx in enumerate(indices[0])
                if idx != -1 and distances[0][e] >= threshold]
        return " ".join(hits) if hits else "関連情報なし"

    def determine_emotion(self, user_text: str):
        """①発話の感情を推定し、②内部状態を更新して、ステアリングに使う感情と強度を返す。

        戻り値は EmotionDynamics.get_steering_vectors() の出力:
            {"steering": [{"name": 感情名, "strength": 強度}, ...],   # 強い順に top_k 個
             "complex_emotion_label": 複合感情ラベル}
        """
        # ① 8感情の確率（neutral の分は含まないので、中立的な文では全部小さくなる）
        emotion_vector = self.classifier.predict_emotion(user_text)
        print("Predicted emotion strength (0.0 ～ 1.0):")
        for name, val in zip(self.classifier.emotions, emotion_vector):
            print(f"  {name}: {val:.2f}")
        # ② 内部状態 = 前の状態 × 減衰 + 今回の入力（感情どうしの相関で周りにも広がる）
        self.brain.update_state(emotion_vector)
        info = self.brain.get_steering_vectors()
        print(f"適用するステアリング: {info['steering']}")
        print("-" * 30)
        return info
