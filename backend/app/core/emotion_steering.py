# backend/app/core/emotion_steering.py
"""
EmotionDynamics（内部感情の力学系）と Steerer（ステアリング適用）をつなぐ層。

このモジュールが解決している問題は2つある。

  1. 【強度のスケールが違う】
     EmotionDynamics の内部状態（state）は 0〜max_strength（既定 30.0）の値。
     一方、研究の α は「その層の平均活性化ノルムに対する割合」で、
         h' = h + α · avg_norm · v̂      （v̂ は単位ベクトル化した感情ベクトル）
     という式で使う。単位がまったく違うので、state をそのまま α にしてはいけない
     （30 を α にすると、壊れないと確認した範囲の数十倍になる）。
     → α = 上限 × (state / max_strength) の線形写像で変換する（strength_to_alpha）。

  2. 【α の上限をどう決めるか】
     上限は研究で決めた値を alpha_cap/chatbot_alpha_cap.json から読む。json には
       per_emotion : 感情ごとの上限（例: joy 1.2, anger 0.8）
       _model_cap  : 全感情の中で最も小さい上限（= per_emotion の最小値、例: 0.8）
     の2種類がある。ただし、これらは
       「システムプロンプトも会話履歴も無い状態で、1回だけ文の続きを生成し、
         20件中 usable（崩壊・拒絶・言語混入なし）が 9割以上なら合格」
     という基準で決めた値であり、10%までの失敗を許している。会話では
     内部状態が上限に張り付いて同じ α がかかり続けるので、この値のままだと壊れる
     （実際に joy=1.2 で「✨✨✨…」の崩壊が起きた）。
     → cap_for() で「per_emotion を _model_cap で頭打ち」にし、
       さらに会話用の安全係数 chat_safety を掛けたものを実際の上限にする。

感情の順序について:
  vectors.npz のキーは感情名なので、順序ではなく**名前で**ベクトルを引く。
  EmotionDynamics は環順、プローブはアルファベット順だが、どちらも名前で
  対応づけているので、このモジュールでは順序の変換は不要。
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

# 読み込む感情の一覧（研究側の8感情。ここでは「どの感情を読むか」にだけ使い、順序には意味がない）
EMOTIONS = ["joy", "sadness", "anticipation", "surprise",
            "anger", "fear", "disgust", "trust"]


class EmotionSteeringController:
    """内部感情状態 → ステアリングのパラメータ（どの感情ベクトルを、どの強さαで足すか）への変換を担う。

    使い方（実際の呼び出しは app/manager.py）:
        ctrl = EmotionSteeringController.from_assets(
            assets_root="data/processed/steering_assets",
            model_key="llama-3.1-swallow-8b-instruct-v0.5", layer=21, chat_safety=0.75)
        steerer = Steerer(model, avg_norms={21: ctrl.avg_norm},
                          refusal_direction=ctrl.refusal_direction)

        # 1ターンごと
        dynamics.update_state(感情ベクトル)                # 内部状態を更新
        plan = ctrl.plan(dynamics.get_steering_vectors())  # 強度 → α に変換
        with steerer.steering(layer=ctrl.layer, vectors=plan["vectors"], alphas=plan["alphas"]):
            text = engine.chat(...)
    """

    def __init__(self, vectors: dict, avg_norm: float, layer: int,
                 alpha_cap: float, per_emotion_cap: Optional[dict] = None,
                 max_strength: float = 30.0,
                 refusal_direction: Optional[np.ndarray] = None,
                 chat_safety: float = 1.0):
        # 感情名 → ステアリングベクトル（単位ベクトル化はしていない生の値。Steerer 側で正規化する）
        self.vectors = vectors
        # その層の平均活性化ノルム。α を「ノルムに対する割合」として使うための基準
        self.avg_norm = float(avg_norm)
        # ステアリングを掛ける層（hidden_states のインデックス）
        self.layer = int(layer)
        # モデル全体の上限（json の _model_cap）。どの感情もこれを超えない
        self.alpha_cap = float(alpha_cap)
        # 感情ごとの上限（json の per_emotion）。_model_cap より大きい値は cap_for() で頭打ちにする
        self.per_emotion_cap = dict(per_emotion_cap or {})
        # EmotionDynamics の強度の最大値。強度 → α の写像で「強度がこの値のとき α = 上限」になる
        self.max_strength = float(max_strength)
        # 拒絶方向（無ければ None）。Steerer が全層でこの方向を射影除去する
        self.refusal_direction = refusal_direction
        # 会話用の安全係数（0〜1）。上限にこれを掛けた値が、実際に会話で使う α の最大値になる
        self.chat_safety = float(chat_safety)

    # ------------------------------------------------------------------
    @classmethod
    def from_assets(cls, assets_root: str, model_key: str, layer: int,
                    variant: str = "rf", default_cap: float = 0.6,
                    max_strength: float = 30.0,
                    chat_safety: float = 1.0) -> "EmotionSteeringController":
        """研究側の成果物（steering_assets/）を読み込んでインスタンスを作る。

        読み込むファイル:
            <assets_root>/steering_vectors/<model_key>/layer_<L>/vectors.npz   感情ベクトル
            <assets_root>/steering_vectors/<model_key>/layer_<L>/meta.json     avg_norm
            <assets_root>/refusal_vectors/<model_key>/r_hat.npy                拒絶方向（任意）
            <assets_root>/alpha_cap/chatbot_alpha_cap.json                     α上限（任意）

        Parameters
        ----------
        variant     : α上限を決めたときの条件。"rf" = 拒絶方向を除去した条件（アプリと同じ）
        default_cap : α上限の json が無い・読めないときに使う上限
        chat_safety : 会話用の安全係数（cap_for() を参照）
        """
        # ---- 1. 感情ベクトル -------------------------------------------------
        vec_dir = os.path.join(assets_root, "steering_vectors", model_key, f"layer_{layer}")
        npz_path = os.path.join(vec_dir, "vectors.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"{npz_path} がありません。scripts/build_assets.py で作成するか、"
                f"研究リポジトリの成果物をコピーしてください。")
        npz = np.load(npz_path)
        # npz のキーは感情名。名前で引くので、保存時の順序には依存しない
        vectors = {e: np.asarray(npz[e], dtype=np.float32)
                   for e in EMOTIONS if e in npz.files}
        missing = [e for e in EMOTIONS if e not in vectors]
        if missing:
            print(f"[Controller][WARNING] vectors.npz に無い感情: {missing}")

        # ---- 2. avg_norm（α のスケール基準。無いと α の意味が研究とずれるので必須）------
        avg_norm = None
        meta_path = os.path.join(vec_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                avg_norm = json.load(f).get("avg_norm")
        if avg_norm is None:
            raise ValueError(
                f"{meta_path} から avg_norm を取得できません。"
                f"avg_norm が無いと α が研究で決めた上限と対応しなくなります。")

        # ---- 3. 拒絶方向（無ければ拒絶除去なしで動く）---------------------------
        r_path = os.path.join(assets_root, "refusal_vectors", model_key, "r_hat.npy")
        r_hat = np.load(r_path).astype(np.float32) if os.path.exists(r_path) else None
        if r_hat is None:
            print(f"[Controller][WARNING] {r_path} が無いため拒絶除去は無効です。")

        # ---- 4. α上限 ----------------------------------------------------------
        # json の構造:
        #   caps -> <variant> -> <model_key> -> by_layer -> "<層番号>" -> {per_emotion, _model_cap}
        # 同じモデルでも層ごとに上限が違うので、必ず「使う層」の値を引く。
        cap, per_emo = default_cap, {}
        cap_path = os.path.join(assets_root, "alpha_cap", "chatbot_alpha_cap.json")
        if os.path.exists(cap_path):
            with open(cap_path, encoding="utf-8") as f:
                data = json.load(f)
            entry = data.get("caps", {}).get(variant, {}).get(model_key)
            if entry:
                by_layer = entry.get("by_layer")
                if by_layer:
                    # 新しい形式: 層ごとの値がある → 使う層の値を取り出す
                    le = by_layer.get(str(layer))
                    if le is None:
                        print(f"[Controller][WARNING] α上限に layer={layer} の項目が無いため"
                              f"既定 {default_cap} を使います（存在する層: "
                              f"{sorted(by_layer.keys())}）")
                        le = {}
                else:
                    # 古い形式: 1つの層の値しかない → 層が一致するときだけ使う
                    le = entry
                    if int(le.get("layer", layer)) != layer:
                        print(f"[Controller][WARNING] α上限は layer={le.get('layer')} で"
                              f"決めた値なので、layer={layer} には使えません。"
                              f"既定 {default_cap} を使います。")
                        le = {}
                cap = float(le.get("_model_cap", default_cap))
                per_emo = {k: float(v) for k, v in le.get("per_emotion", {}).items()}
            else:
                print(f"[Controller][WARNING] {cap_path} に {variant}/{model_key} の"
                      f"項目が無いため既定 {default_cap} を使います。")
        else:
            print(f"[Controller] α上限ファイルが無いため既定 {default_cap} を使います。")

        ctrl = cls(vectors, avg_norm, layer, cap, per_emo, max_strength, r_hat, chat_safety)

        # 起動時に「実際に会話で使う上限」を表示して、設定が効いているか確認できるようにする
        print(f"[Controller] model={model_key} layer={layer} avg_norm={avg_norm:.3f} "
              f"_model_cap={cap} chat_safety={chat_safety} "
              f"拒絶除去={'有効' if r_hat is not None else '無効'}")
        print(f"[Controller] 実効α上限: "
              f"{ {e: round(ctrl.cap_for(e), 3) for e in vectors} }")
        return ctrl

    # ------------------------------------------------------------------
    def cap_for(self, emotion: str) -> float:
        """会話で実際に使う、その感情の α の上限を返す。

        計算は3段階:
          1. json の per_emotion からその感情の上限を取る（無ければ _model_cap）
               例: joy → 1.2、anger → 0.8
          2. _model_cap（全感情の最小値）で頭打ちにする
               例: joy → min(1.2, 0.8) = 0.8
             per_emotion は「上限を下げる方向」にだけ効く。
          3. 会話用の安全係数を掛ける
               例: 0.8 × 0.75 = 0.6
        """
        per = self.per_emotion_cap.get(emotion, self.alpha_cap)   # 1.
        capped = min(per, self.alpha_cap)                         # 2.
        return capped * self.chat_safety                          # 3.

    # ------------------------------------------------------------------
    def strength_to_alpha(self, emotion: str, strength: float) -> float:
        """内部状態の強度（0〜max_strength）→ ステアリングの α（0〜cap_for(emotion)）に変換する。

        線形写像:  α = cap_for(emotion) × (strength / max_strength)
          例: strength=15, max_strength=30, 上限0.6 → α = 0.6 × 0.5 = 0.3

        研究の α と EmotionDynamics の強度は単位が違うので、
        強度を α として使うときは必ずこの関数を通すこと。
        """
        cap = self.cap_for(emotion)
        # max_strength が 0 以下だと割り算ができない（設定ミス）ので、ステアリングしない
        if self.max_strength <= 0:
            return 0.0
        # 強度を 0〜1 の割合にする。範囲外の値（負や max_strength 超え）は端に丸める
        ratio = max(0.0, min(1.0, float(strength) / self.max_strength))
        return cap * ratio

    # ------------------------------------------------------------------
    def plan(self, steering_info: dict) -> dict:
        """EmotionDynamics.get_steering_vectors() の出力を、Steerer に渡す形に変換する。

        Parameters
        ----------
        steering_info : {"steering": [{"name": 感情名, "strength": 強度}, ...],
                         "complex_emotion_label": 複合感情ラベル}
                        steering には強度の大きい順に top_k 個の感情が入っている。

        Returns
        -------
        {"vectors": {感情名: ベクトル}, "alphas": {感情名: α},
         "scale_applied": 再スケールの倍率（1.0 なら再スケールなし）,
         "label": 複合感情ラベル（UI表示用）}
        """
        items = steering_info.get("steering", []) or []
        vectors, alphas = {}, {}

        # ---- 各感情の強度を α に変換する ---------------------------------------
        for it in items:
            emo, st = it.get("name"), float(it.get("strength", 0.0))
            if emo not in self.vectors:
                print(f"[Controller][WARNING] ベクトルが無い感情をスキップ: {emo}")
                continue
            a = self.strength_to_alpha(emo, st)
            if a == 0.0:          # 強度0の感情はステアリングしない
                continue
            vectors[emo] = self.vectors[emo]
            alphas[emo] = a

        # ---- 複数感情を同時に足す場合の安全策 -----------------------------------
        # Steerer は Σ α_e · v̂_e（感情ごとの単位ベクトルの重み付き和）を足す。
        # その合計ベクトルの長さが、単一感情でいう α に相当する。研究で上限を
        # 確かめたのは単一感情だけなので、合計の長さも上限以下に抑える。
        # ※top_k=1（既定）なら感情は1つだけなので、この処理は実行されない。
        scale = 1.0
        if len(alphas) > 1:
            limit = self.alpha_cap * self.chat_safety      # 合計に許す長さ
            combined = None
            for emo, a in alphas.items():
                u = self.vectors[emo].astype(np.float64)
                u = u / (np.linalg.norm(u) + 1e-8)          # 単位ベクトルにする
                combined = u * a if combined is None else combined + u * a
            norm = float(np.linalg.norm(combined))          # 合計ベクトルの長さ
            if norm > limit and norm > 0:
                scale = limit / norm
                alphas = {e: a * scale for e, a in alphas.items()}
                print(f"[Controller] 合成ノルム {norm:.3f} > 上限 {limit:.3f} のため "
                      f"×{scale:.3f} に再スケールしました")

        return {"vectors": vectors, "alphas": alphas, "scale_applied": scale,
                "label": steering_info.get("complex_emotion_label", "neutral")}

    # ------------------------------------------------------------------
    def state_to_alphas(self, emotions: list, state) -> list:
        """内部状態（8感情の強度 0〜max_strength）を、表示用に α 換算した値のリストにする。

        各感情について strength_to_alpha と同じ式（上限 × 強度 / max_strength）で換算する。
        実際にステアリングで掛かるのは一番強い1感情だけで、他の値は
        「その感情でステアリングしたら掛かる α」を表す（UI のレーダーチャート用）。
        """
        return [self.strength_to_alpha(e, float(s)) for e, s in zip(emotions, state)]

    @property
    def display_max(self) -> float:
        """表示用の α の最大値（= 実効上限の最大。レーダーチャートの目盛りの上限に使う）。"""
        return max((self.cap_for(e) for e in self.vectors), default=self.alpha_cap)