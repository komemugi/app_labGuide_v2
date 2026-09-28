# backend/app/core/steerer.py
"""
感情ステアリング＋拒絶方向除去のフック管理。

★2026-07-28 全面改訂（研究側 src/steering/hooks.py と数式を一致させた）。
  旧実装からの変更点:

  1. 【バグ修正】ブロック出力がタプルのとき `hidden_states = outputs` としており、
     タプルに .clone() を呼んで AttributeError で落ちる状態だった
     （正しくは outputs[0]）。現在動いているのは transformers が新しく
     DecoderLayer がテンソルを直接返すため else 分岐に入っているからで、
     ライブラリ更新で突然壊れる。研究側でも同じバグを修正済み。

  2. 【αのノルムスケール化】旧: h' = h + strength * v （vは未正規化）
                            新: h' = h + α * avg_norm * (v/||v||)
     αが「その層の典型的な活性化ノルムに対する割合」になり、
     **モデル・層・感情をまたいで意味が同じ**になる。研究の α と直接比較できる。

  3. 【拒絶方向除去の追加】旧実装には無い。Arditi et al. 方式の射影除去
         h' = h - r_hat (r_hat · h)
     を**全ブロック・全トークン位置**に適用する。固定量の減算ではない点に注意
     （射影なので、拒絶方向の成分を持たない入力にはほぼ何もしない＝無害）。

  4. 【層インデックスの規約統一】研究側は hidden_states のインデックスで層を指定する。
     hidden_states[0] は埋め込み層なので、model.model.layers[i] に対応するのは
     hidden_states[i+1]。convert_hidden_index_to_block_index() で変換する。
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Iterable, Optional, Sequence

import numpy as np
import torch


# =============================================================================
# 層インデックスの変換とブロック取得
# =============================================================================
def convert_hidden_index_to_block_index(hidden_index: int) -> int:
    """hidden_states のインデックス → model.model.layers のインデックス。

    hidden_states[0]   = 埋め込み層の出力（対応するTransformerブロックは無い）
    hidden_states[i+1] = model.model.layers[i] の出力
    """
    if hidden_index <= 0:
        raise ValueError(
            "hidden_index=0 は埋め込み層であり、対応するTransformerブロックがありません。"
            "1以上を指定してください。"
        )
    return hidden_index - 1


def get_blocks(model) -> Sequence:
    """モデルのTransformerブロック列を取得する（アーキテクチャ差を吸収）。

    LFM2 / Llama / Mistral / Qwen / Gemma など、多くは model.model.layers。
    見つからない場合は候補を順に探す。
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.decoder.layers,
        lambda m: m.transformer.h,
        lambda m: m.layers,
    ]
    for get in candidates:
        try:
            blocks = get(model)
            if blocks is not None and len(blocks) > 0:
                return blocks
        except AttributeError:
            continue
    raise AttributeError(
        "Transformerブロック列が見つかりません。model の構造を確認し、"
        "get_blocks() に候補を追加してください。"
    )


def _unpack(output):
    """ブロック出力から hidden_states と残りを取り出す（タプル対応）。"""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _repack(modified, rest):
    """元の構造に戻す。"""
    if rest is None:
        return modified
    return (modified,) + rest


# =============================================================================
# フック本体
# =============================================================================
def build_steering_hook(vector, alpha: float, avg_norm: float,
                        normalize: bool = True, apply_to: str = "all"):
    """感情ステアリングの forward hook を作る。

    normalize=True のとき h' = h + alpha * avg_norm * (v/||v||)  ← 研究側と同一
    apply_to="all" は全トークン位置に適用（生成タスクの通常運用）。
    """
    v = torch.as_tensor(np.asarray(vector, dtype=np.float32))
    if normalize:
        v = v / (v.norm() + 1e-8)
        scale = float(alpha) * float(avg_norm)
    else:
        scale = float(alpha)

    def hook_fn(module, inputs, output):
        hidden_states, rest = _unpack(output)
        steer = v.to(device=hidden_states.device, dtype=hidden_states.dtype) * scale
        if apply_to == "last":
            modified = hidden_states.clone()
            modified[:, -1, :] = modified[:, -1, :] + steer
        else:
            modified = hidden_states + steer
        return _repack(modified, rest)

    return hook_fn


def build_ablation_hook(direction, eps: float = 1e-8):
    """拒絶方向の除去 h' = h - r_hat (r_hat · h) を行う forward hook。

    ★固定量の減算ではなく**射影除去**である点が重要。
      拒絶方向の成分を持たない入力にはほぼ恒等写像として働くため、
      強度の較正が不要で、通常の会話を壊さない。
    """
    r = torch.as_tensor(np.asarray(direction, dtype=np.float32))
    r = r / (r.norm() + eps)

    def hook_fn(module, inputs, output):
        hidden_states, rest = _unpack(output)
        r_dev = r.to(device=hidden_states.device, dtype=hidden_states.dtype)
        coeff = torch.matmul(hidden_states, r_dev).unsqueeze(-1)  # (batch, seq, 1)
        modified = hidden_states - coeff * r_dev
        return _repack(modified, rest)

    return hook_fn


# =============================================================================
# Steerer
# =============================================================================
class Steerer:
    """感情ステアリングと拒絶方向除去のフックを管理する。

    使い方（推奨: with構文。例外が出てもフックが必ず外れる）:
        steerer = Steerer(model, avg_norms={20: 12.3})
        steerer.load_refusal_direction("path/to/r_hat.npy")
        with steerer.steering(layer=20, vectors={"anger": v_anger}, alphas={"anger": 0.4}):
            out = model.generate(...)

    手動で使う場合は apply_hook() / remove_hook() のペアを必ず try/finally で囲むこと。
    """

    def __init__(self, model, avg_norms: Optional[dict] = None,
                 refusal_direction=None, ablation_enabled: bool = True):
        """
        Parameters
        ----------
        model : PreTrainedModel
        avg_norms : dict[int, float], optional
            {hidden_index: その層の平均活性化ノルム}。
            α のノルムスケーリングに使う（研究側 compute_avg_norm の出力と同じもの）。
            未指定の層では normalize=False にフォールバックし、警告する。
        refusal_direction : np.ndarray, optional
            拒絶方向 r_hat（単位ベクトルでなくてよい）。
        ablation_enabled : bool
            Trueかつrefusal_directionがあるとき、ステアリング時に方向除去も同時に適用する。
        """
        self.model = model
        self.avg_norms = dict(avg_norms or {})
        self.refusal_direction = (
            np.asarray(refusal_direction, dtype=np.float32)
            if refusal_direction is not None else None
        )
        self.ablation_enabled = ablation_enabled
        self._hook_handles = []

    # ------------------------------------------------------------------
    # 読み込み系
    # ------------------------------------------------------------------
    def load_refusal_direction(self, path: str) -> bool:
        """refusal_vectors/<model_key>/r_hat.npy を読み込む。無ければFalseを返す。"""
        if not os.path.exists(path):
            print(f"[Steerer] 拒絶ベクトルが見つかりません（除去は無効）: {path}")
            self.refusal_direction = None
            return False
        self.refusal_direction = np.load(path).astype(np.float32)
        print(f"[Steerer] 拒絶ベクトルを読み込みました: {path} "
              f"(dim={self.refusal_direction.shape[-1]})")
        return True

    @staticmethod
    def load_alpha_caps(path: str, variant: str = "rf",
                        default_cap: float = 0.6) -> dict:
        """chatbot_alpha_cap.json を読み込む。

        研究側 scripts/select_alpha_cap_judge.py の出力形式:
          {"criteria": {...},
           "caps": {"rf": {"<model_key>": {"layer": 20,
                                           "per_emotion": {"anger": 0.6, ...},
                                           "_model_cap": 0.6}}}}

        ★ファイルが無い場合は空dictを返す。呼び出し側は default_cap にフォールバックすること
          （judgeの結果が出る前でもアプリを動かせるようにするため）。
        """
        if not os.path.exists(path):
            print(f"[Steerer] α上限ファイルが未作成のため既定値 {default_cap} を使います: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        caps = data.get("caps", {}).get(variant, {})
        if not caps:
            print(f"[Steerer][WARNING] {path} に variant='{variant}' の項目がありません。")
        return caps

    # ------------------------------------------------------------------
    # フックの登録・解除
    # ------------------------------------------------------------------
    def apply_hook(self, layer, vectors: dict, alphas: dict,
                   normalize: bool = True, apply_to: str = "all",
                   with_ablation: Optional[bool] = None) -> None:
        """指定層に感情ステアリングを、（有効なら）全層に拒絶方向除去を掛ける。

        Parameters
        ----------
        layer : int | Sequence[int]
            **hidden_states のインデックス**で指定する（研究側と同じ規約）。
            ★2026-07-28（CM節）: リストを渡すと**複数層に同時適用**する。
              ステアリング研究では層の範囲に適用するのが一般的で、
              単層だけでは長い応答生成に対して効果が弱いことがあるため。
              複数層に同じαを掛けると効果が累積するので、
              1層あたりのαは単層時より小さくすること（目安: 単層α / 層数の平方根）。
        vectors : dict[str, np.ndarray]   {感情名: ベクトル}
        alphas : dict[str, float]         {感情名: α}。0のものは無視。負値も可。
        """
        blocks = get_blocks(self.model)
        layers = [layer] if isinstance(layer, (int, np.integer)) else list(layer)
        for L in layers:
            bi = convert_hidden_index_to_block_index(int(L))
            if not (0 <= bi < len(blocks)):
                raise IndexError(
                    f"layer={L}（block={bi}）は範囲外です。"
                    f"このモデルのブロック数は {len(blocks)} です。")

        # --- 1. 感情ステアリング（合成ベクトルを1本にまとめてから適用）---
        combined = None
        for emo, vec in vectors.items():
            a = float(alphas.get(emo, 0.0))
            if vec is None or a == 0.0:
                continue
            v = np.asarray(vec, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            combined = (v * a) if combined is None else (combined + v * a)

        if combined is not None:
            for L in layers:
                L = int(L)
                bi = convert_hidden_index_to_block_index(L)
                avg_norm = self.avg_norms.get(L)
                if avg_norm is None and self.avg_norms:
                    # 指定層のavg_normが無ければ既知の層の値で代用する
                    avg_norm = next(iter(self.avg_norms.values()))
                use_norm = normalize and (avg_norm is not None)
                if normalize and avg_norm is None:
                    print(f"[Steerer][WARNING] layer={L} の avg_norm が未設定のため、"
                          f"ノルムスケーリングを無効化します。")
                payload = combined * float(avg_norm) if use_norm else combined
                hook = build_steering_hook(
                    payload, alpha=1.0, avg_norm=1.0, normalize=False, apply_to=apply_to,
                )
                self._hook_handles.append(blocks[bi].register_forward_hook(hook))
            print(f"[Steerer] 感情ステアリングを layer={layers} "
                  f"(block={[convert_hidden_index_to_block_index(int(L)) for L in layers]}) に適用 "
                  f"alphas={ {k: v for k, v in alphas.items() if v} } normalize={normalize}")

        # --- 2. 拒絶方向の除去（全ブロック）---
        do_abl = self.ablation_enabled if with_ablation is None else with_ablation
        if do_abl and self.refusal_direction is not None:
            abl_hook = build_ablation_hook(self.refusal_direction)
            for blk in blocks:
                self._hook_handles.append(blk.register_forward_hook(abl_hook))
            print(f"[Steerer] 拒絶方向除去を全 {len(blocks)} ブロックに適用")

    def remove_hook(self) -> None:
        """登録済みのフックをすべて解除する。"""
        if self._hook_handles:
            for handle in self._hook_handles:
                handle.remove()
            n = len(self._hook_handles)
            self._hook_handles.clear()
            print(f"[Steerer] {n}個のフックを解除しました。")

    @contextlib.contextmanager
    def steering(self, layer: int, vectors: dict, alphas: dict, **kwargs):
        """with構文用。例外が出てもフックを必ず解除する。"""
        try:
            self.apply_hook(layer, vectors, alphas, **kwargs)
            yield self
        finally:
            self.remove_hook()

    @contextlib.contextmanager
    def ablation_only(self):
        """感情ステアリング無しで拒絶方向除去だけを掛ける（比較実験・素の応答用）。"""
        blocks = get_blocks(self.model)
        handles = []
        try:
            if self.refusal_direction is not None:
                hook = build_ablation_hook(self.refusal_direction)
                for blk in blocks:
                    handles.append(blk.register_forward_hook(hook))
            yield self
        finally:
            for h in handles:
                h.remove()
