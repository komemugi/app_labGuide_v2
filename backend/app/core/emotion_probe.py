# backend/app/core/emotion_probe.py
"""
線形感情プローブ

LLM自身の隠れ状態からユーザー発話の感情を読む。scripts/train_emotion_probe.py
（研究側と同一の学習条件）が保存した重み probe_layer<L>.npz を読み込むだけなので、
別の分類モデルをロードする必要がない。

【推論時の読み方は学習時と必ず揃える】
npz に記録された dataset から抽出位置を決め、学習時と**同じ関数**
（app/core/activation.py）で隠れ状態を取る。
  dataset=synthetic_resp       → response_mean（文をassistant役に置き全トークン平均）
  dataset=synthetic_promptlast → prompt_last  （文をuser役に置き最終トークン）
同梱のプローブは synthetic_resp で学習されている。

★旧版は「生テキスト＋BOSの最終トークン」を読んでおり、学習時と役割・テンプレート・
  プーリングの3点が食い違っていた（2026-09 に修正）。

【DeBERTaを置き換えた理由】
中立の質問「大学の学食について教えてください」に fear 0.55 を出すなど、
実測で使い物にならなかったため。

ステアリングを掛けていないクリーンな順伝播で読むこと
（ステアリング後の状態を読むと「自分で書いた感情を自分で読む」循環になる）。
"""

from __future__ import annotations

import os

import numpy as np

from app.core.activation import extract, position_for_dataset


class EmotionProbe:
    """LLMの隠れ状態から感情を読む線形プローブ。

    使い方:
        probe = EmotionProbe(emotions, engine, probe_path)
        vec = probe.predict_emotion("この対応はひどい")   # emotions の順で8要素
    """

    def __init__(self, emotions: list, engine, probe_path: str,
                 layer: int = None, temperature: float = 1.0, position: str = None):
        """
        Parameters
        ----------
        emotions : 出力の並び順（EmotionDynamics と同じ環順を渡す）
        engine   : LLMEngine（model と tokenizer を使う）
        probe_path : probe_layer<L>.npz へのパス
        layer    : hidden_states のインデックス。Noneならnpzに記録された層
        temperature : softmaxの温度。小さいほど1つの感情に集中する
        position : 抽出位置を明示したい場合のみ指定。Noneならnpzの dataset から決める
        """
        if not os.path.exists(probe_path):
            raise FileNotFoundError(
                f"'{probe_path}' がありません。scripts/build_assets.py で作成するか、"
                f"同梱の steering_assets を配置してください。")
        d = np.load(probe_path, allow_pickle=True)
        self.W = d["W"].astype(np.float32)          # (n_classes, hidden_dim)
        self.b = d["b"].astype(np.float32)          # (n_classes,)
        self.classes = [str(c) for c in d["classes"]]
        self.layer = int(layer if layer is not None else d["layer"])
        if layer is not None and "layer" in d.files and int(d["layer"]) != int(layer):
            print(f"[EmotionProbe][WARNING] プローブは layer={int(d['layer'])} で学習されていますが、"
                  f"layer={layer} で読もうとしています。")
        self.dataset = str(d["dataset"]) if "dataset" in d.files else "unknown"
        self.position = position or position_for_dataset(self.dataset)
        self.engine = engine
        self.emotions = list(emotions)
        self.temperature = float(temperature)

        missing = [e for e in self.emotions if e not in self.classes]
        if missing:
            raise ValueError(f"プローブに存在しない感情があります: {missing}\n"
                             f"プローブのクラス: {self.classes}")
        self._order = [self.classes.index(e) for e in self.emotions]
        self._neutral_idx = self.classes.index("neutral") if "neutral" in self.classes else None
        if self._neutral_idx is None:
            print("[EmotionProbe][WARNING] プローブに 'neutral' クラスがありません。")

        self._last_text, self._last_probs = None, None   # 同じ文の二重計算を避ける

        acc = float(d["test_accuracy"]) if "test_accuracy" in d.files else float("nan")
        print(f"[EmotionProbe] {probe_path} を読み込みました (layer={self.layer}, "
              f"dataset={self.dataset}, position={self.position}, test_acc={acc:.3f}, "
              f"classes={len(self.classes)})")

    # ------------------------------------------------------------------
    def _hidden_state(self, text: str) -> np.ndarray:
        """学習時と同じ位置・同じ関数で、指定層の隠れ状態を1本取る。"""
        acts = extract(self.engine.model, self.engine.tokenizer, [text], [self.layer],
                       self.position, batch_size=1, verbose=False)
        return acts[self.layer][0]

    def _probs(self, text: str) -> np.ndarray:
        """全クラス（neutral含む）の確率。直前と同じ文ならキャッシュを返す。"""
        if text == self._last_text and self._last_probs is not None:
            return self._last_probs
        h = self._hidden_state(text)
        z = (self.W @ h + self.b) / max(self.temperature, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        self._last_text, self._last_probs = text, p
        return p

    def predict_emotion(self, text: str) -> list:
        """テキスト → 8感情の確率（emotions の順）。neutral の分は含めない。

        neutral に確率が集まった文では8感情がすべて小さくなり、内部状態がほぼ動かない。
        """
        p = self._probs(text)
        return [float(p[i]) for i in self._order]

    def predict_top(self, text: str, k: int = 3) -> list:
        """デバッグ用: 上位k感情を (名前, 確率) で返す。"""
        v = self.predict_emotion(text)
        idx = np.argsort(v)[::-1][:k]
        return [(self.emotions[i], float(v[i])) for i in idx]

    def neutral_score(self, text: str) -> float:
        """中立らしさ（0〜1）。neutralクラスが無いプローブでは 0.0。"""
        if self._neutral_idx is None:
            return 0.0
        return float(self._probs(text)[self._neutral_idx])
