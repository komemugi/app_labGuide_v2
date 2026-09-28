"""
backend/scripts/build_assets.py

アプリが使う研究成果物（ステアリングベクトル・感情プローブ・拒絶方向）を
まとめて作り直す入口。**同梱の成果物をそのまま使うなら実行不要**。

ステップ（steps で選べる）:
  activations : 合成データの隠れ状態を抽出してキャッシュ          [GPU]
  vectors     : CAA＋中立PCAで感情ステアリングベクトルを計算       [CPU]
  probe       : 線形感情プローブ（8感情＋neutral）を学習           [CPU]
  refusal     : 拒絶方向 r_hat を抽出（拒絶ペアのCSVが必要）       [GPU]

※ α上限（alpha_cap/chatbot_alpha_cap.json）はここでは作らない。
  生成品質の評価（WRIME中立文の続き生成＋LLM評価）が必要なため、研究リポジトリで決めた値を同梱している。
  ベクトルを作り直した場合、上限も決め直さないと意味が保証されない点に注意。

ノートブックから（ロード済みモデルを使い回す。8Bモデルを二重にロードしない）:
    from scripts.build_assets import build_all
    build_all(engine, model_key=RESEARCH_MODEL_KEY, layer=TARGET_LAYER,
              steps=["activations", "vectors", "probe"])

コマンドラインから（backend/ で実行）:
    python scripts/build_assets.py --steps activations vectors probe
    python scripts/build_assets.py --steps vectors probe        # キャッシュがあればGPU不要
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import extract_activations  # noqa: E402
import extract_refusal_vectors  # noqa: E402
import extract_steering_vectors  # noqa: E402
import train_emotion_probe  # noqa: E402

ALL_STEPS = ["activations", "vectors", "probe", "refusal"]
GPU_STEPS = {"activations", "refusal"}


def build_all(engine=None, model_key: str = common.DEFAULT_MODEL_KEY,
              layer: int = common.DEFAULT_LAYER, steps=("activations", "vectors", "probe"),
              position: str = "response_mean",
              assets_root: str = common.ASSETS_ROOT, cache_root: str = common.CACHE_ROOT,
              emotions_path: str = common.EMOTIONS_PATH,
              neutral_path: str = common.NEUTRAL_PATH,
              refusal_pairs_path: str = common.REFUSAL_PAIRS_PATH,
              batch_size: int = 16) -> dict:
    """成果物を作り直す。engine はアプリの LLMEngine（None ならGPUステップの前に読み込む）。"""
    steps = list(steps)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise ValueError(f"不明なステップ: {unknown}（{ALL_STEPS} から選択）")
    if GPU_STEPS & set(steps) and engine is None:
        engine = common.load_engine(model_key)

    dataset, neutral_dir = extract_activations.dir_names(position)
    results, t0 = {}, time.time()

    if "activations" in steps:
        results["activations"] = extract_activations.run(
            engine=engine, model_key=model_key, layers=[layer], position=position,
            emotions_path=emotions_path, neutral_path=neutral_path,
            cache_root=cache_root, batch_size=batch_size)
    if "vectors" in steps:
        results["vectors"] = extract_steering_vectors.run(
            model_key=model_key, layers=[layer], dataset=dataset,
            neutral_dir_name=neutral_dir, cache_root=cache_root, assets_root=assets_root)
    if "probe" in steps:
        results["probe"] = train_emotion_probe.run(
            model_key=model_key, layer=layer, dataset=dataset,
            neutral_dir_name=neutral_dir, with_neutral=True,
            cache_root=cache_root, assets_root=assets_root)
    if "refusal" in steps:
        if not os.path.exists(refusal_pairs_path):
            raise FileNotFoundError(
                f"拒絶ペアのCSVがありません: {refusal_pairs_path}\n"
                f"（公開リポジトリには含めていません。研究リポジトリからコピーしてください）")
        results["refusal"] = extract_refusal_vectors.run(
            engine=engine, model_key=model_key, dataset_path=refusal_pairs_path,
            assets_root=assets_root, cache_root=cache_root, batch_size=batch_size)

    print(f"\n[build_assets] 完了: {steps}（{time.time() - t0:.0f}秒）")
    if {"vectors", "refusal"} & set(steps):
        print("[build_assets] ★ベクトルを作り直したので、α上限（chatbot_alpha_cap.json）が"
              "今のベクトルに対して妥当かは保証されません。会話で壊れないか必ず確認してください。")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", default=common.DEFAULT_MODEL_KEY)
    p.add_argument("--layer", type=int, default=common.DEFAULT_LAYER)
    p.add_argument("--steps", nargs="+", default=["activations", "vectors", "probe"],
                   choices=ALL_STEPS)
    p.add_argument("--position", default="response_mean", choices=["response_mean", "prompt_last"])
    p.add_argument("--assets_root", default=common.ASSETS_ROOT)
    p.add_argument("--cache_root", default=common.CACHE_ROOT)
    p.add_argument("--refusal_pairs_path", default=common.REFUSAL_PAIRS_PATH)
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    build_all(model_key=a.model_key, layer=a.layer, steps=a.steps, position=a.position,
              assets_root=a.assets_root, cache_root=a.cache_root,
              refusal_pairs_path=a.refusal_pairs_path, batch_size=a.batch_size)
