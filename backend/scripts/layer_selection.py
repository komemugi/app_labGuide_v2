"""
backend/scripts/layer_selection.py  （任意: CPUのみ）

各層の隠れ状態で線形感情分類器を学習し、test accuracy を比較する（研究側
scripts/layer_selection.py の移植）。別のモデルや層を試すときだけ使う。
事前に手順1を --layers all で実行しておくこと。

  python scripts/extract_activations.py --layers all
  python scripts/layer_selection.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
from common import CACHE_ROOT, DEFAULT_MODEL_KEY  # noqa: E402


def probe_single_layer(layer, X, labels, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return {"layer": layer,
            "train_accuracy": accuracy_score(y_tr, clf.predict(X_tr)),
            "test_accuracy": accuracy_score(y_te, clf.predict(X_te))}


def run(model_key: str = DEFAULT_MODEL_KEY, dataset: str = "synthetic_resp",
        cache_root: str = CACHE_ROOT, n_jobs: int = -1) -> list:
    from joblib import Parallel, delayed

    act_dir = os.path.join(cache_root, model_key, dataset)
    labels = np.load(os.path.join(act_dir, "labels.npy"), allow_pickle=True)
    layers = sorted(int(m.group(1)) for f in glob.glob(os.path.join(act_dir, "layer_*.npy"))
                    if (m := re.search(r"layer_(\d+)\.npy$", f)))
    if not layers:
        raise FileNotFoundError(f"'{act_dir}' に layer_*.npy がありません。")
    print(f"[layer_selection] 対象層: {layers}")

    tasks = [delayed(probe_single_layer)(l, np.load(os.path.join(act_dir, f"layer_{l}.npy")),
                                         labels) for l in layers]
    results = sorted(Parallel(n_jobs=n_jobs)(tasks), key=lambda r: r["layer"])
    for r in results:
        gap = r["train_accuracy"] - r["test_accuracy"]
        print(f"  layer {r['layer']:3d}: test={r['test_accuracy']:.4f} "
              f"train={r['train_accuracy']:.4f}{'  ← 過学習気味' if gap > 0.3 else ''}")
    best = max(results, key=lambda r: r["test_accuracy"])
    print(f"[layer_selection] 最良層: layer={best['layer']} (test={best['test_accuracy']:.4f})")

    out_path = os.path.join(act_dir, "layer_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model_key": model_key, "dataset": dataset, "best_layer": best["layer"],
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[layer_selection] 保存 -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", default=DEFAULT_MODEL_KEY)
    p.add_argument("--dataset", default="synthetic_resp")
    p.add_argument("--cache_root", default=CACHE_ROOT)
    a = p.parse_args()
    run(a.model_key, a.dataset, a.cache_root)
