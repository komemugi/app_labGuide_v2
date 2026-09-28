# backend/app/core/emotion_dynamics.py
"""
内部感情の力学系（プルチックの感情環に基づく）。

★2026-07-28 ノートブックのセル12から移設。変更点は3つ:
  1. 中立閾値 neutral_threshold を独立したパラメータに分離
     旧: `if top1_strength < self.alpha` と、0〜max_strengthスケールの強度を
         感受性係数(0.5)と比較していた。単位が違うため実質ゲートが機能せず、
         中立の質問にもステアリングが掛かっていた。
  2. sensitivity の既定を 0.5 → 1.5 に較正
     旧設定では怒りを何ターン重ねても α が 0.16 で頭打ちだった（研究の上限0.6に対し1/4）。
     1.5 なら T1=0.23 / T3=0.40 / 飽和0.46 となり、上限に余裕を残しつつ体感できる。
  3. base_state（キャラクターの平常心）を任意で指定できるようにした
     指定しなければ従来通り 0 に向かって減衰する（＝案C: 直前の感情を保持しつつ減衰）。
"""

from __future__ import annotations

import numpy as np

# プルチックの環順。この順序を前提に相関行列を作るので、変更しないこと。
WHEEL_ORDER = ["joy", "trust", "fear", "surprise",
               "sadness", "disgust", "anger", "anticipation"]


class EmotionDynamics:
    def __init__(self, emotions: list, max_strength: float = 30.0,
                 decay_rate: float = 0.5, sensitivity: float = 1.5,
                 neutral_threshold: float = 8.0, base_state=None,
                 top_k: int = 1, self_boost: float = 1.0): 
        """
        Parameters
        ----------
        emotions : 感情名のリスト。★プルチックの環順であること
        max_strength : 内部状態の上限 τ_max
        decay_rate : 減衰率 λ。前ターンの感情がどれだけ残るか（慣性）
        sensitivity : 感受性 κ。入力に対する反応の強さ
        neutral_threshold : θ_neutral。最大感情がこれ未満ならステアリングしない。
                            ★max_strength と同じスケール（0〜max_strength）で指定する。
                            既定8.0は α換算で約0.16（cap=0.6のとき）。
        base_state : キャラクターの平常心（8要素）。Noneなら0に向かって減衰する。
        """
        if list(emotions) != WHEEL_ORDER:
            print(f"[EmotionDynamics][WARNING] 感情の順序が環順と異なります。\n"
                  f"  期待: {WHEEL_ORDER}\n  実際: {list(emotions)}\n"
                  f"  相関行列は環順を前提にしているため、結果が意味をなさなくなります。")
        self.emotions = list(emotions)
        self.num_emotions = len(self.emotions)
        self.max_strength = float(max_strength)
        self.decay = float(decay_rate)
        self.sensitivity = float(sensitivity)
        self.neutral_threshold = float(neutral_threshold)
        self.self_boost = float(self_boost)
        self.base_state = (np.zeros(self.num_emotions) if base_state is None
                           else np.asarray(base_state, dtype=float))
        self.state = self.base_state.copy()
        self.top_k = int(top_k)
        self.W = self._build_correlation_matrix()

        self.dyads_map = {
            frozenset(["joy", "trust"]): "love",
            frozenset(["trust", "fear"]): "submission",
            frozenset(["fear", "surprise"]): "alarm",
            frozenset(["surprise", "sadness"]): "disappointment",
            frozenset(["sadness", "disgust"]): "remorse",
            frozenset(["disgust", "anger"]): "contempt",
            frozenset(["anger", "anticipation"]): "aggressiveness",
            frozenset(["anticipation", "joy"]): "optimism",
        }

    def _build_correlation_matrix(self):
        """プルチックの角度に基づく 8x8 の相関行列。W_ij = cos(θi - θj)"""
        n = self.num_emotions
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = min((i - j) % n, (j - i) % n)
                W[i, j] = np.cos(diff * (np.pi / 4.0))
        return W

    def update_state(self, input_vector):
        """入力感情ベクトル（環順）を受け取り、内部状態を更新する。

        e_{t+1} = clip( λ(e_t - e_base) + e_base + tanh(κ W i_t) τ_max , 0, τ_max )
        ★base_state=0（既定）なら従来式 λe_t + s_t と同一。
        """
        # I = np.asarray(input_vector, dtype=float)
        # added = np.tanh(self.sensitivity * (self.W @ I)) * self.max_strength
        I = np.asarray(input_vector, dtype=float)
        # 入力された感情自身を強調する（波及だけで隣接感情が逆転するのを防ぐ）
        added = np.tanh(self.sensitivity * (self.W @ I + self.self_boost * I)) * self.max_strength
        decayed = self.decay * (self.state - self.base_state) + self.base_state
        self.state = np.clip(decayed + added, 0.0, self.max_strength)
        return self.state

    def get_steering_vectors(self):
        """Top-2の感情と複合感情ラベルを返す。

        ★最大感情が neutral_threshold 未満なら steering=[] を返す（中立扱い）。
        """
        order = np.argsort(self.state)[::-1]
        i1, i2 = int(order[0]), int(order[1])
        e1, s1 = self.emotions[i1], float(self.state[i1])
        e2, s2 = self.emotions[i2], float(self.state[i2])

        if s1 < self.neutral_threshold:
            return {"steering": [], "complex_emotion_label": "neutral",
                    "state_vector": self.state.tolist()}

        label = self.dyads_map.get(frozenset([e1, e2]), e1)
        pairs = [{"name": e1, "strength": s1}, {"name": e2, "strength": s2}]
        return {
            "steering": pairs[:self.top_k],          # ★Top-kだけ返す
            "complex_emotion_label": label,          # ラベルはTop-2から判定（UI表示用）
            "state_vector": self.state.tolist(),
        }

    def reset(self):
        """会話をリセットする（新しい被験者・新しいセッションの開始時に呼ぶ）。"""
        self.state = self.base_state.copy()