"""
backend/scripts/train_emotion_probe.py  （手順3: CPUのみ）

手順1のキャッシュから線形感情プローブ（多クラスロジスティック回帰、8感情＋neutral）を学習し、
steering_assets/emotion_probe/probe_layer<L>.npz に保存する。
研究側 scripts/train_emotion_probe.py の移植（学習条件は同一）:
  標準化なし / LogisticRegression(max_iter=1000, class_weight="balanced")
  / train_test_split(test_size=0.2, random_state=42, stratify=y)

★プローブは dataset（= 抽出位置）を npz に記録する。アプリの EmotionProbe はこれを読んで、
  推論時も**同じ位置**でユーザー発話の隠れ状態を取る（学習と推論の条件ずれを防ぐ）。

使用例:
  python scripts/train_emotion_probe.py --layer 21
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
from common import ASSETS_ROOT, CACHE_ROOT, DEFAULT_LAYER, DEFAULT_MODEL_KEY  # noqa: E402


def run(model_key: str = DEFAULT_MODEL_KEY, layer: int = DEFAULT_LAYER,
        dataset: str = "synthetic_resp", neutral_dir_name: str = None,
        with_neutral: bool = True, max_neutral: int = None,
        cache_root: str = CACHE_ROOT, assets_root: str = ASSETS_ROOT,
        test_size: float = 0.2, random_state: int = 42,
        use_cv: bool = False, cs: int = 10, cv_folds: int = 3) -> dict:
    neutral_dir_name = neutral_dir_name or (
        "neutral_resp" if dataset.endswith("_resp") else "neutral")
    act_dir = os.path.join(cache_root, model_key, dataset)
    x_path = os.path.join(act_dir, f"layer_{layer}.npy")
    y_path = os.path.join(act_dir, "labels.npy")
    for p in (x_path, y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"'{p}' がありません。先に手順1を実行してください。")

    X = np.load(x_path)
    y = np.load(y_path, allow_pickle=True)
    if len(X) != len(y):
        raise ValueError(f"サンプル数が不一致: X={len(X)} vs labels={len(y)}")

    if with_neutral:
        neu_path = os.path.join(cache_root, model_key, neutral_dir_name, f"layer_{layer}.npy")
        if not os.path.exists(neu_path):
            raise FileNotFoundError(f"'{neu_path}' がありません。")
        Xn = np.load(neu_path)
        if max_neutral is None:
            max_neutral = int(np.median(np.unique(y, return_counts=True)[1]))
        if len(Xn) > max_neutral:
            rs = np.random.RandomState(random_state)
            Xn = Xn[rs.choice(len(Xn), max_neutral, replace=False)]
        print(f"[probe] 中立クラスを追加: {len(Xn)}件 ({neu_path})")
        X = np.concatenate([X, Xn], axis=0)
        y = np.concatenate([y, np.array(["neutral"] * len(Xn), dtype=object)])

    classes, counts = np.unique(y, return_counts=True)
    print(f"[probe] {model_key} layer={layer} dataset={dataset}")
    print(f"[probe] X={X.shape}  クラス分布={dict(zip(classes.tolist(), counts.tolist()))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    if use_cv:
        clf = LogisticRegressionCV(Cs=cs, cv=cv_folds, max_iter=1000,
                                   class_weight="balanced", random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"[probe] train_accuracy={tr_acc:.4f}  test_accuracy={te_acc:.4f}")
    if tr_acc - te_acc > 0.25:
        print(f"[probe][WARNING] train と test の差が {tr_acc-te_acc:.2f} と大きく、過学習の疑いがあります。"
              f"use_cv=True を試してください。")

    cm = confusion_matrix(y_te, clf.predict(X_te), labels=clf.classes_)
    print("\n[probe] 混同行列（行=正解, 列=予測）:")
    print("            " + " ".join(f"{c[:6]:>7s}" for c in clf.classes_))
    for i, c in enumerate(clf.classes_):
        acc_i = cm[i, i] / max(cm[i].sum(), 1)
        print(f"  {c:10s}" + " ".join(f"{v:7d}" for v in cm[i]) + f"   ({acc_i:.2f})")

    out_dir = os.path.join(assets_root, "emotion_probe")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"probe_layer{layer}.npz")
    np.savez(
        out_path,
        W=clf.coef_.astype(np.float32), b=clf.intercept_.astype(np.float32),
        classes=np.array(clf.classes_, dtype=object), layer=layer,
        train_accuracy=tr_acc, test_accuracy=te_acc, n_samples=len(X),
        dataset=dataset, model_key=model_key, with_neutral=with_neutral,
        neutral_dir_name=neutral_dir_name,
        trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    print(f"\n[probe] 保存 -> {out_path}  (W={clf.coef_.shape}, classes={list(clf.classes_)})")
    return {"path": out_path, "train_accuracy": tr_acc, "test_accuracy": te_acc,
            "classes": list(clf.classes_)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", default=DEFAULT_MODEL_KEY)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--dataset", default="synthetic_resp",
                   choices=["synthetic_resp", "synthetic_promptlast"])
    p.add_argument("--neutral_dir_name", default=None)
    p.add_argument("--no_neutral", action="store_true", help="neutralクラスを加えない")
    p.add_argument("--max_neutral", type=int, default=None)
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--assets_root", default=ASSETS_ROOT)
    p.add_argument("--use_cv", action="store_true")
    a = p.parse_args()
    run(a.model_key, a.layer, a.dataset, a.neutral_dir_name, not a.no_neutral,
        a.max_neutral, a.cache_root, a.assets_root, use_cv=a.use_cv)
