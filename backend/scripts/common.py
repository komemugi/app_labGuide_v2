"""
backend/scripts/common.py

成果物生成スクリプト共通の設定（パス・モデル一覧・モデル読み込み）。

パスはすべて backend/ を基準に解決するので、どのディレクトリから実行しても同じ場所を指す。
"""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def bpath(*parts) -> str:
    """backend/ からの相対パスを絶対パスにする。"""
    return os.path.join(BACKEND_DIR, *parts)


# --- 既定パス -----------------------------------------------------------------
# 成果物（アプリが読むもの。git管理する）
ASSETS_ROOT = bpath("data", "processed", "steering_assets")
# 活性化キャッシュ（数百MB〜数GBになる。git管理しない）
CACHE_ROOT = bpath("data", "processed", "activations")
# 合成データ（gemma-4-12b-it で生成、話し言葉のみ、各739件）
EMOTIONS_PATH = bpath("data", "raw", "steering", "synthetic", "emotions_spoken_739.jsonl")
NEUTRAL_PATH = bpath("data", "raw", "steering", "synthetic", "neutral_spoken_739.jsonl")
# 拒絶ペア（target=拒絶されやすい依頼 / baseline=応じられる依頼）。公開しないデータ
REFUSAL_PAIRS_PATH = bpath("data", "raw", "steering", "refusal_vector",
                           "safe_refusal_vector_dataset_v2.csv")

# --- モデル ---------------------------------------------------------------------
# 研究側 configs/models.yaml のキー → Hugging Face のパス
MODELS = {
    "llama-3.1-swallow-8b-instruct-v0.5": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}
DEFAULT_MODEL_KEY = "llama-3.1-swallow-8b-instruct-v0.5"
DEFAULT_LAYER = 21


def load_engine(model_key: str):
    """アプリと同じ LLMEngine（bf16・量子化なし）でモデルを読み込む。

    ★量子化すると活性化の値が変わり、成果物の条件とずれるので使わない。
    """
    from app.core.llm_engine import LLMEngine
    if model_key not in MODELS:
        raise KeyError(f"未登録のモデルキー: {model_key}（scripts/common.py の MODELS に追加してください）")
    return LLMEngine(model_id=MODELS[model_key], quantization_8bit=False)


def unpack_model(engine=None, model=None, tokenizer=None):
    """engine（LLMEngine）か model/tokenizer の組を受け取り、(model, tokenizer) を返す。"""
    if engine is not None:
        return engine.model, engine.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("engine か (model, tokenizer) のどちらかを渡してください。")
    return model, tokenizer


def num_hidden_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def resolve_layers(layers, model) -> list:
    """"all" または [-1] を全層（0=埋め込み層 〜 num_hidden_layers）に展開する。"""
    if layers in ("all", [-1], -1):
        return list(range(num_hidden_layers(model) + 1))
    if isinstance(layers, int):
        return [layers]
    return [int(l) for l in layers]
