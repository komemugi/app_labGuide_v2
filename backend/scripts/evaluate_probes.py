"""
backend/scripts/evaluate_probes.py

2つの確認をまとめたスクリプト（ノートブックから呼ぶ想定）。

  1. evaluate_probes : 複数の感情プローブを、自分で書いた評価セットで比べる
       - 評価セットは「実際にチャットで話しかけそうな発話」と正解ラベルの jsonl
         （data/eval/probe_eval_set.jsonl。1行 {"text": ..., "label": ...}）
       - 合成データ上の test acc（0.883）は「合成データに似た文」での成績なので、
         実際の発話でどれくらい当たるかを別に測る必要がある
       - 各プローブは npz に記録された dataset から読み方（生成条件 / 理解条件）を
         自動で決めるので、学習時と同じ読み方で評価される

  2. compare_assets  : 作り直した成果物が同梱品と一致するかを確かめる（再現確認）
       - ステアリングベクトルのコサイン類似度（全感情で ≈1.0 なら移植は正しい）
       - avg_norm、PCA の成分数、プローブの test acc の比較

使い方（ノートブック。engine はロード済みの LLMEngine）:
    from scripts.evaluate_probes import evaluate_probes, compare_assets
    evaluate_probes(engine, {
        "同梱（生成条件）":  "data/processed/steering_assets/emotion_probe/probe_layer21.npz",
        "理解条件":          "/tmp/rebuilt_pl/emotion_probe/probe_layer21.npz",
    })
    compare_assets("data/processed/steering_assets", "/tmp/rebuilt")
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import ASSETS_ROOT, DEFAULT_LAYER, DEFAULT_MODEL_KEY, bpath  # noqa: E402

EVAL_SET_PATH = bpath("data", "eval", "probe_eval_set.jsonl")
EVAL_OUT_DIR = bpath("data", "eval", "results")

# EmotionProbe に渡す8感情（評価では neutral を含む全クラスで判定するので、順序はどれでもよい）
EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


# ============================================================================
# 1. プローブの比較
# ============================================================================
def load_eval_set(path: str = EVAL_SET_PATH) -> list:
    """評価セット（jsonl）を読む。各行 {"text": 発話, "label": 正解, "note": 任意のメモ}。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    labels = sorted({it["label"] for it in items})
    print(f"[eval] 評価セット {len(items)}件 読み込み: {path}")
    print(f"[eval] ラベルごとの件数: { {l: sum(it['label'] == l for it in items) for l in labels} }")
    return items


def _predict_all(probe, text: str) -> dict:
    """neutral を含む全クラスの確率を {クラス名: 確率} で返す。

    EmotionProbe.predict_emotion は neutral を除いた8感情しか返さないので、
    評価では内部の _probs（全クラスの確率）を直接使う。
    """
    p = probe._probs(text)
    return {c: float(p[i]) for i, c in enumerate(probe.classes)}


def evaluate_probes(engine, probes: dict, eval_path: str = EVAL_SET_PATH,
                    save: bool = True) -> dict:
    """複数のプローブを同じ評価セットで比べる。

    Parameters
    ----------
    engine    : ロード済みの LLMEngine（プローブが隠れ状態を読むのに使う）
    probes    : {表示名: probe_layer<L>.npz のパス}
    eval_path : 評価セットの jsonl
    save      : True なら data/eval/results/ に1文ごとの結果を CSV で保存する

    判定方法:
      予測 = neutral を含む全クラスの中で確率が最大のクラス。
      正解ラベルと一致すれば正解。
    """
    from app.core.emotion_probe import EmotionProbe

    items = load_eval_set(eval_path)
    results = {}       # {表示名: [{"pred": ..., "prob": ..., "true_prob": ...}, ...]}
    summary = {}

    for name, path in probes.items():
        print(f"\n{'='*70}\n[eval] {name}: {path}\n{'='*70}")
        probe = EmotionProbe(EMOTIONS, engine, probe_path=path)
        rows = []
        for it in items:
            probs = _predict_all(probe, it["text"])
            pred = max(probs, key=probs.get)
            rows.append({"pred": pred, "prob": probs[pred],
                         "true_prob": probs.get(it["label"], 0.0),
                         "correct": pred == it["label"]})
        results[name] = rows

        # ---- 集計 ----------------------------------------------------------
        n = len(items)
        emo_idx = [i for i, it in enumerate(items) if it["label"] != "neutral"]
        neu_idx = [i for i, it in enumerate(items) if it["label"] == "neutral"]
        acc_all = sum(r["correct"] for r in rows) / n
        acc_emo = (sum(rows[i]["correct"] for i in emo_idx) / len(emo_idx)) if emo_idx else float("nan")
        acc_neu = (sum(rows[i]["correct"] for i in neu_idx) / len(neu_idx)) if neu_idx else float("nan")
        # 中立の文を「何かの感情」と誤判定した割合（ステアリングが不要にかかる原因になる）
        false_emo = (sum(rows[i]["pred"] != "neutral" for i in neu_idx) / len(neu_idx)) if neu_idx else float("nan")
        mean_true = float(np.mean([r["true_prob"] for r in rows]))
        summary[name] = {"acc_all": acc_all, "acc_emotion": acc_emo, "acc_neutral": acc_neu,
                         "neutral_misfire": false_emo, "mean_true_prob": mean_true,
                         "probe_test_acc": probe_test_acc(path)}

        # ラベルごとの正解数
        per_label = {}
        for it, r in zip(items, rows):
            c, t = per_label.get(it["label"], (0, 0))
            per_label[it["label"]] = (c + r["correct"], t + 1)
        print("[eval] ラベルごとの正解数: " +
              ", ".join(f"{l} {c}/{t}" for l, (c, t) in sorted(per_label.items())))

    # ---- 比較表（全体の成績）------------------------------------------------
    print(f"\n{'='*70}\n[eval] 全体の比較（評価セット {len(items)}件）\n{'='*70}")
    head = f"{'':24s}{'全体':>8s}{'感情文':>8s}{'中立文':>8s}{'中立誤爆':>9s}{'正解確率':>9s}{'合成test':>9s}"
    print(head)
    for name, s in summary.items():
        print(f"{name:24s}{s['acc_all']:8.3f}{s['acc_emotion']:8.3f}{s['acc_neutral']:8.3f}"
              f"{s['neutral_misfire']:9.3f}{s['mean_true_prob']:9.3f}{s['probe_test_acc']:9.3f}")
    print("  全体/感情文/中立文 : 正解率（高いほど良い）")
    print("  中立誤爆           : 中立の文を感情ありと判定した割合（低いほど良い）")
    print("  正解確率           : 正解ラベルに付いた確率の平均（高いほど自信を持って当たっている）")
    print("  合成test           : 学習時の合成データでの test acc（参考）")

    # ---- 1文ごとの比較（外れた文を見るため）----------------------------------
    names = list(probes.keys())
    print(f"\n[eval] 1文ごとの予測（✓=正解）")
    for i, it in enumerate(items):
        cells = []
        for nm in names:
            r = results[nm][i]
            cells.append(f"{'✓' if r['correct'] else '✗'}{r['pred'][:7]:7s}{r['prob']:.2f}")
        print(f"  {it['label'][:7]:7s} | " + " | ".join(cells) + f" | {it['text'][:28]}")
    print(f"  （列の順: {' | '.join(names)}）")

    # ---- CSV に保存 ----------------------------------------------------------
    if save:
        os.makedirs(EVAL_OUT_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(EVAL_OUT_DIR, f"probe_eval_{stamp}.csv")
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"] + [f"{nm}:{k}" for nm in names for k in ("pred", "prob", "correct")])
            for i, it in enumerate(items):
                row = [it["text"], it["label"]]
                for nm in names:
                    r = results[nm][i]
                    row += [r["pred"], f"{r['prob']:.3f}", int(r["correct"])]
                w.writerow(row)
        print(f"\n[eval] 1文ごとの結果を保存 -> {out}")

    return {"summary": summary, "results": results, "items": items}


def probe_test_acc(path: str) -> float:
    d = np.load(path, allow_pickle=True)
    return float(d["test_accuracy"]) if "test_accuracy" in d.files else float("nan")


# ============================================================================
# 2. 再現確認
# ============================================================================
def compare_assets(original_root: str = ASSETS_ROOT, rebuilt_root: str = "/tmp/rebuilt",
                   model_key: str = DEFAULT_MODEL_KEY, layer: int = DEFAULT_LAYER) -> dict:
    """作り直した成果物（rebuilt_root）が同梱品（original_root）と一致するかを確かめる。

    判定の目安:
      コサイン類似度が全感情で 0.99 以上 → 移植は正しい（GPU計算の揺れで 1.0 ぴったりにはならない）
      0.9〜0.99 → 小さな条件の違いがある（チャットテンプレートの日付など）
      0.9 未満  → 抽出条件が違う。設定を見直す必要がある
    """
    rel = os.path.join("steering_vectors", model_key, f"layer_{layer}")
    a = np.load(os.path.join(original_root, rel, "vectors.npz"))
    b = np.load(os.path.join(rebuilt_root, rel, "vectors.npz"))
    with open(os.path.join(original_root, rel, "meta.json"), encoding="utf-8") as f:
        ma = json.load(f)
    with open(os.path.join(rebuilt_root, rel, "meta.json"), encoding="utf-8") as f:
        mb = json.load(f)

    print(f"[compare] 同梱: {original_root}\n[compare] 再生成: {rebuilt_root}\n")
    print(f"{'感情':14s}{'コサイン':>10s}{'ノルム(同梱)':>14s}{'ノルム(再生成)':>16s}")
    cos = {}
    for e in sorted(a.files):
        if e not in b.files:
            print(f"{e:14s}  再生成側に無し")
            continue
        va, vb = a[e].astype(np.float64), b[e].astype(np.float64)
        cos[e] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        print(f"{e:14s}{cos[e]:10.4f}{np.linalg.norm(va):14.3f}{np.linalg.norm(vb):16.3f}")

    print(f"\n{'項目':20s}{'同梱':>12s}{'再生成':>12s}")
    print(f"{'avg_norm':20s}{ma.get('avg_norm', float('nan')):12.4f}{mb.get('avg_norm', float('nan')):12.4f}")
    print(f"{'PCA 成分数':20s}{ma.get('pca', {}).get('n_components', '-'):>12}"
          f"{mb.get('pca', {}).get('n_components', '-'):>12}")

    pa = os.path.join(original_root, "emotion_probe", f"probe_layer{layer}.npz")
    pb = os.path.join(rebuilt_root, "emotion_probe", f"probe_layer{layer}.npz")
    if os.path.exists(pa) and os.path.exists(pb):
        print(f"{'プローブ test acc':20s}{probe_test_acc(pa):12.4f}{probe_test_acc(pb):12.4f}")

    worst = min(cos.values()) if cos else float("nan")
    verdict = ("移植は正しい" if worst >= 0.99 else
               "小さな条件差あり（許容範囲か確認）" if worst >= 0.9 else "抽出条件が違う（要確認）")
    print(f"\n[compare] 最小コサイン = {worst:.4f} → {verdict}")
    return {"cosine": cos, "min_cosine": worst, "verdict": verdict}
