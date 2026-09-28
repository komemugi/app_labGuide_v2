"""
backend/scripts/extract_activations.py  （手順1: GPU必須）

合成データ（感情文・中立文）をモデルに通し、指定層の隠れ状態をキャッシュする。
研究側 scripts/extract_activations.py の移植。WRIME・Wikipediaの選択肢は削除し、
gemma-4-12b-it で生成した中立文を既定にした。

キャッシュの配置（手順2・3がここを読む）:
  <cache_root>/<model_key>/<dataset>/layer_<L>.npy   (n_total, dim) 感情名のソート順に連結
  <cache_root>/<model_key>/<dataset>/labels.npy
  <cache_root>/<model_key>/<dataset>/meta.json
  <cache_root>/<model_key>/<neutral_dir>/layer_<L>.npy
  <cache_root>/<model_key>/<neutral_dir>/meta.json

  dataset / neutral_dir は抽出位置で決まる（研究側と同じ命名）:
    response_mean → synthetic_resp / neutral_resp   ← 同梱の成果物はこちら
    prompt_last   → synthetic_promptlast / neutral

使用例:
  python scripts/extract_activations.py --layers 21
  python scripts/extract_activations.py --layers all        # 層選択（layer_selection.py）用
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (sys.path に backend/ も追加される)
from common import (CACHE_ROOT, DEFAULT_LAYER, DEFAULT_MODEL_KEY, EMOTIONS_PATH,
                    NEUTRAL_PATH, load_engine, resolve_layers, unpack_model)
from app.core.activation import extract
from app.research.datasets import load_neutral, load_synthetic


def dir_names(position: str) -> tuple:
    """抽出位置 → (感情のサブディレクトリ名, 中立のサブディレクトリ名)。"""
    if position == "response_mean":
        return "synthetic_resp", "neutral_resp"
    if position == "prompt_last":
        return "synthetic_promptlast", "neutral"
    raise ValueError(f"unknown position: {position}")


def run(engine=None, model=None, tokenizer=None,
        model_key: str = DEFAULT_MODEL_KEY, layers=(DEFAULT_LAYER,),
        position: str = "response_mean",
        emotions_path: str = EMOTIONS_PATH, neutral_path: str = NEUTRAL_PATH,
        cache_root: str = CACHE_ROOT, min_level: int = 1,
        max_per_emotion=None, max_neutral=None, batch_size: int = 16) -> dict:
    """感情文と中立文の活性化を抽出・保存する。ロード済みの engine を渡せる（ノートブック用）。"""
    model, tokenizer = unpack_model(engine, model, tokenizer)
    layers = resolve_layers(list(layers) if not isinstance(layers, str) else layers, model)
    emo_dir_name, neu_dir_name = dir_names(position)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[extract_activations] model={model_key} position={position} layers={layers}")

    # ---- 1. 感情文 ----------------------------------------------------------
    texts_by_emotion = load_synthetic(emotions_path, min_level=min_level,
                                      max_per_emotion=max_per_emotion)
    emo_dir = os.path.join(cache_root, model_key, emo_dir_name)
    os.makedirs(emo_dir, exist_ok=True)

    # ★行順と labels の対応を固定するため、感情名のソート順で連結する（研究側と同じ規則）
    emotion_order = sorted(texts_by_emotion.keys())
    labels, chunks = [], {l: [] for l in layers}
    for emo in emotion_order:
        texts = texts_by_emotion[emo]
        print(f"\n[extract_activations] '{emo}' を抽出中... ({len(texts)}件)")
        acts = extract(model, tokenizer, texts, layers, position, batch_size=batch_size)
        for l in layers:
            chunks[l].append(acts[l])
        labels.extend([emo] * len(texts))

    labels = np.array(labels)
    for l in layers:
        X = np.concatenate(chunks[l], axis=0)
        assert X.shape[0] == len(labels), f"layer {l}: 行数 {X.shape[0]} != ラベル数 {len(labels)}"
        np.save(os.path.join(emo_dir, f"layer_{l}.npy"), X)
    np.save(os.path.join(emo_dir, "labels.npy"), labels)
    with open(os.path.join(emo_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_key": model_key, "dataset": emo_dir_name, "position": position,
            "synthetic_path": os.path.relpath(emotions_path, common.BACKEND_DIR),
            "min_level": min_level, "max_per_emotion": max_per_emotion,
            "layers": layers, "emotion_order": emotion_order,
            "n_samples_per_emotion": {e: len(texts_by_emotion[e]) for e in emotion_order},
            "extracted_at": now,
        }, f, ensure_ascii=False, indent=2)
    print(f"[extract_activations] 感情文を保存 -> {emo_dir} (計{len(labels)}件)")

    # ---- 2. 中立文 ----------------------------------------------------------
    neutral_texts = load_neutral(neutral_path, max_samples=max_neutral)
    print(f"\n[extract_activations] 中立文を抽出中... ({len(neutral_texts)}件)")
    neu = extract(model, tokenizer, neutral_texts, layers, position, batch_size=batch_size)
    neu_dir = os.path.join(cache_root, model_key, neu_dir_name)
    os.makedirs(neu_dir, exist_ok=True)
    for l in layers:
        np.save(os.path.join(neu_dir, f"layer_{l}.npy"), neu[l])
    with open(os.path.join(neu_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_key": model_key, "position": position,
            "neutral_corpus_path": os.path.relpath(neutral_path, common.BACKEND_DIR),
            "layers": layers, "n_samples": len(neutral_texts), "extracted_at": now,
        }, f, ensure_ascii=False, indent=2)
    print(f"[extract_activations] 中立文を保存 -> {neu_dir} ({len(neutral_texts)}件)")

    return {"dataset": emo_dir_name, "neutral_dir_name": neu_dir_name, "layers": layers}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", default=DEFAULT_MODEL_KEY)
    p.add_argument("--layers", nargs="+", default=[str(DEFAULT_LAYER)],
                   help="hidden_states のインデックス。'all' で全層")
    p.add_argument("--position", default="response_mean", choices=["response_mean", "prompt_last"])
    p.add_argument("--emotions_path", default=EMOTIONS_PATH)
    p.add_argument("--neutral_path", default=NEUTRAL_PATH)
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--min_level", type=int, default=1)
    p.add_argument("--max_per_emotion", type=int, default=None)
    p.add_argument("--max_neutral", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    layers = "all" if a.layers == ["all"] else [int(x) for x in a.layers]
    run(engine=load_engine(a.model_key), model_key=a.model_key, layers=layers,
        position=a.position, emotions_path=a.emotions_path, neutral_path=a.neutral_path,
        cache_root=a.cache_root, min_level=a.min_level, max_per_emotion=a.max_per_emotion,
        max_neutral=a.max_neutral, batch_size=a.batch_size)
