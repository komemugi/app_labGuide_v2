"""
backend/app/research/datasets.py

成果物生成に使うデータの読み込み。研究リポジトリの
src/data/synthetic.py・src/data/neutral_corpus.py・extract_refusal_vectors.load_refusal_pairs
を1ファイルにまとめたもの（WRIMEへの依存は削除）。

データ形式（jsonl、1行1文）:
  {"emotion": "anger", "level": 3, "topic": "...", "style": "spoken",
   "text": "...", "generator": "gemma-4-12b-it"}
中立文は emotion="neutral", level=0。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from typing import Optional

# 研究側 WRIME_8 と同じタクソノミー（名前で引くので順序には依存しない）
EMOTIONS = ["joy", "sadness", "anticipation", "surprise",
            "anger", "fear", "disgust", "trust"]


def load_synthetic(path: str, min_level: int = 1,
                   max_per_emotion: Optional[int] = None) -> dict:
    """感情文の jsonl を読み、{感情: [文, ...]} を返す（研究側 load_synthetic と同じ選別）。"""
    by_emo: dict = defaultdict(list)
    n_skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            emo = r.get("emotion")
            if emo not in EMOTIONS:
                continue
            if r.get("level", 0) < min_level:
                n_skipped += 1
                continue
            by_emo[emo].append(r["text"])

    result = {}
    for emo in EMOTIONS:
        texts = by_emo.get(emo, [])
        if max_per_emotion is not None:
            texts = texts[:max_per_emotion]
        result[emo] = texts
        print(f"[load_synthetic] {emo:13s}: {len(texts)}件 (level>={min_level})")
    if n_skipped:
        print(f"[load_synthetic] level<{min_level}のため{n_skipped}件を除外")
    missing = [e for e in EMOTIONS if not result[e]]
    if missing:
        raise ValueError(f"次の感情のサンプルが0件です: {missing}（{path} を確認してください）")
    return result


def load_neutral(path: str, max_samples: Optional[int] = None) -> list:
    """中立文の jsonl を読む（各行の "text" を使う）。"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
            if max_samples is not None and len(texts) >= max_samples:
                break
    print(f"[load_neutral] {len(texts)}件読み込み: {path}")
    return texts


def load_refusal_pairs(path: str) -> list:
    """target/baseline のペアCSVを読む（研究側 load_refusal_pairs と同一）。"""
    pairs = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "target" not in cols or "baseline" not in cols:
            raise ValueError(f"CSVに target/baseline 列が必要です。実際の列: {cols}")
        for i, row in enumerate(reader):
            t = (row.get("target") or "").strip()
            b = (row.get("baseline") or "").strip()
            if not t or not b:
                continue
            pairs.append({
                "id": row.get("id") or f"rf{i:04d}",
                "target": t, "baseline": b,
                "pair_type": row.get("pair_type") or "unknown",
                "source": row.get("source") or "unknown",
            })
    print(f"[load_refusal_pairs] {len(pairs)}ペア読み込み: {path}")
    return pairs
