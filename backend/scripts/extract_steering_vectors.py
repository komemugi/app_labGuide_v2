"""
backend/scripts/extract_steering_vectors.py  （手順2: CPUのみ）

手順1のキャッシュから、CAA＋中立PCAで感情ステアリングベクトルを計算し、
アプリが読む場所（steering_assets/steering_vectors/<model_key>/layer_<L>/）に保存する。
研究側 scripts/extract_steering_vectors.py の移植（計算は app/research/caa.py で研究側と同一）。

  v_e = (I - U U^T)(mean(H_e) - mean(H_all))     U: 中立文PCAの上位成分（累積寄与率50%）
  avg_norm = 中立文の活性化ノルムの平均           （α のスケール基準）

使用例:
  python scripts/extract_steering_vectors.py --layers 21
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import ASSETS_ROOT, CACHE_ROOT, DEFAULT_LAYER, DEFAULT_MODEL_KEY  # noqa: E402
from app.research.caa import (SteeringVectorSet, compute_avg_norm,  # noqa: E402
                              compute_caa_vectors, compute_neutral_pca)


def _read_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_cached(cache_root: str, model_key: str, dataset: str, neutral_dir_name: str,
                layer: int):
    """キャッシュを読み、({感情: (n, dim)}, 中立 (n, dim)) を返す。"""
    emo_dir = os.path.join(cache_root, model_key, dataset)
    x_path = os.path.join(emo_dir, f"layer_{layer}.npy")
    y_path = os.path.join(emo_dir, "labels.npy")
    n_path = os.path.join(cache_root, model_key, neutral_dir_name, f"layer_{layer}.npy")
    for p in (x_path, y_path, n_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' がありません。先に手順1（scripts/extract_activations.py --layers {layer}）"
                f"を実行してください。")
    X = np.load(x_path)
    labels = np.load(y_path, allow_pickle=True)
    if len(X) != len(labels):
        raise ValueError(f"layer_{layer}.npy({len(X)}行) と labels.npy({len(labels)}行) が不一致です。")
    H_by_emotion = {e: X[labels == e] for e in sorted(set(labels.tolist()))}
    return H_by_emotion, np.load(n_path)


def run(model_key: str = DEFAULT_MODEL_KEY, layers=(DEFAULT_LAYER,),
        dataset: str = "synthetic_resp", neutral_dir_name: str = None,
        cache_root: str = CACHE_ROOT, assets_root: str = ASSETS_ROOT,
        pca_threshold: float = 0.5) -> list:
    neutral_dir_name = neutral_dir_name or (
        "neutral_resp" if dataset.endswith("_resp") else "neutral")
    out_root = os.path.join(assets_root, "steering_vectors")
    saved = []
    for l in layers:
        print(f"\n{'='*60}\n[steering_vectors] layer={l}\n{'='*60}")
        H_by_emotion, H_neutral = load_cached(cache_root, model_key, dataset,
                                              neutral_dir_name, l)
        print(f"[steering_vectors] 感情ごとの件数: "
              f"{ {e: len(h) for e, h in H_by_emotion.items()} } / 中立: {len(H_neutral)}")

        U_neu, pca_info = compute_neutral_pca(H_neutral, pca_threshold)
        vectors = compute_caa_vectors(H_by_emotion, U_neu)
        avg_norm = compute_avg_norm(H_neutral)
        print(f"[steering_vectors] avg_norm(layer={l}) = {avg_norm:.4f}  ※αのスケール基準")

        # 研究側で meta に抜けていた抽出条件（データのパス・位置）も記録する
        emo_meta = _read_json(os.path.join(cache_root, model_key, dataset, "meta.json"))
        neu_meta = _read_json(os.path.join(cache_root, model_key, neutral_dir_name, "meta.json"))
        out_dir = SteeringVectorSet(
            model_key=model_key, layer=l, vectors=vectors, avg_norm=avg_norm,
            meta={
                "dataset": dataset,
                "position": emo_meta.get("position"),
                "neutral_dir_name": neutral_dir_name,
                "pca": pca_info,
                "n_samples_per_emotion": {e: int(len(v)) for e, v in H_by_emotion.items()},
                "n_neutral_samples": int(len(H_neutral)),
                "source_activations_meta": emo_meta,
                "source_neutral_meta": neu_meta,
            },
        ).save(out_root)
        saved.append(out_dir)
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", default=DEFAULT_MODEL_KEY)
    p.add_argument("--layers", type=int, nargs="+", default=[DEFAULT_LAYER])
    p.add_argument("--dataset", default="synthetic_resp",
                   choices=["synthetic_resp", "synthetic_promptlast"])
    p.add_argument("--neutral_dir_name", default=None)
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--assets_root", default=ASSETS_ROOT)
    p.add_argument("--pca_threshold", type=float, default=0.5)
    a = p.parse_args()
    run(a.model_key, a.layers, a.dataset, a.neutral_dir_name, a.cache_root,
        a.assets_root, a.pca_threshold)
