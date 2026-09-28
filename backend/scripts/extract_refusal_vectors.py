"""
backend/scripts/extract_refusal_vectors.py  （手順4: GPU必須）

拒絶方向 r_hat を Arditi et al. (2024) の方式で抽出する。研究側
scripts/extract_refusal_vectors.py の移植（手順・既定値は同一）。方向除去のフックは
アプリの app/core/steerer.py のものを使う（研究側 hooks.ablation_context と同じ式
h' = h - r̂(r̂·h) を全ブロックに掛ける）。

  1. 拒絶ペア（target=拒絶されやすい依頼 / baseline=応じられる依頼）を読む
  2. 行動フィルタ: 実際に「target を拒絶し baseline を拒絶しない」ペアだけ残す
     （通過が min_kept_pairs 未満なら fallback: 方向は全ペア、検証は実拒絶targetのみ）
  3. 全層で prompt_last の隠れ状態を取り、層ごとに r_l = mean(H_target) - mean(H_baseline)
  4. 深さ25〜75%の層を分離度で上位K本に絞り、各方向で除去して生成 → 拒絶率が最小の層を採用

出力:
  steering_assets/refusal_vectors/<model_key>/r_hat.npy, selected.json, meta.json, directions.npz
  <cache_root>/<model_key>/refusal/behavioral_filter_log.jsonl
    ★このログには拒絶されやすい依頼文とモデルの応答が入るので、公開しないこと
      （cache_root は .gitignore 対象）。

使用例:
  python scripts/extract_refusal_vectors.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (ASSETS_ROOT, CACHE_ROOT, DEFAULT_MODEL_KEY,  # noqa: E402
                    REFUSAL_PAIRS_PATH, load_engine, num_hidden_layers, unpack_model)
from app.core.activation import build_prompt, extract_activations  # noqa: E402
from app.core.steerer import build_ablation_hook, get_blocks  # noqa: E402
from app.research.datasets import load_refusal_pairs  # noqa: E402
from app.research.refusal_detect import detect_refusal  # noqa: E402


# ---------------------------------------------------------------------------
# 生成と方向除去
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def ablation_context(model, direction):
    """全ブロックに方向除去フックを掛け、抜けたら必ず外す。"""
    hook = build_ablation_hook(direction)
    handles = [blk.register_forward_hook(hook) for blk in get_blocks(model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate_short(model, tokenizer, prompt_text: str, max_new_tokens: int = 48,
                   repetition_penalty: float = 1.0) -> str:
    """チャットテンプレート込みで greedy 生成する（研究側と同一の条件）。"""
    full = build_prompt(tokenizer, prompt_text, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if repetition_penalty and repetition_penalty != 1.0:
        kw["repetition_penalty"] = repetition_penalty
    out = model.generate(**inputs, **kw)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def behavioral_filter(model, tokenizer, pairs, max_new_tokens=48, repetition_penalty=1.0):
    kept, dropped = [], []
    for j, p in enumerate(pairs):
        t_text = generate_short(model, tokenizer, p["target"], max_new_tokens, repetition_penalty)
        b_text = generate_short(model, tokenizer, p["baseline"], max_new_tokens, repetition_penalty)
        t_det, b_det = detect_refusal(t_text), detect_refusal(b_text)
        rec = {**p,
               "target_refused": int(t_det["is_refusal"]),
               "baseline_refused": int(b_det["is_refusal"]),
               "target_generation": t_text, "baseline_generation": b_text,
               "target_refusal_reason": t_det["refusal_reason"],
               "baseline_refusal_reason": b_det["refusal_reason"]}
        (kept if (t_det["is_refusal"] and not b_det["is_refusal"]) else dropped).append(rec)
        if (j + 1) % 20 == 0:
            print(f"[behavioral_filter]   {j+1}/{len(pairs)} 処理 (kept={len(kept)})")
    print(f"[behavioral_filter] 通過 {len(kept)} / 除外 {len(dropped)}")
    return kept, dropped


# ---------------------------------------------------------------------------
# 方向の計算と選定
# ---------------------------------------------------------------------------
def separation_score(H_t, H_b, r) -> float:
    rn = r / (np.linalg.norm(r) + 1e-8)
    pt, pb = H_t @ rn, H_b @ rn
    return float((pt.mean() - pb.mean()) / (np.sqrt((pt.var() + pb.var()) / 2.0) + 1e-8))


def candidate_layers_in_band(all_layers, n_layers, lo=0.25, hi=0.75):
    out = [L for L in all_layers if L > 0 and lo <= L / max(n_layers, 1) <= hi]
    return out or [L for L in all_layers if L > 0]


def validate_by_generation(model, tokenizer, r_l, hidden_layer, val_pairs, n_val=20,
                           max_new_tokens=48, repetition_penalty=1.0):
    val = val_pairs[:n_val]
    n_ref = n_empty = 0
    for p in val:
        with ablation_context(model, r_l):
            txt = generate_short(model, tokenizer, p["target"], max_new_tokens, repetition_penalty)
        d = detect_refusal(txt)
        n_ref += d["is_refusal"]
        n_empty += d["is_empty"]
    n = max(len(val), 1)
    return {"hidden_layer": int(hidden_layer), "refusal_rate_after": n_ref / n,
            "empty_rate_after": n_empty / n, "n_val": n}


def select_direction(directions, H_t, H_b, n_layers, model, tokenizer, val_pairs,
                     top_k=5, n_val=20, do_validation=True, repetition_penalty=1.0,
                     force_layer=None):
    cand = candidate_layers_in_band(sorted(directions.keys()), n_layers)
    sep = {l: separation_score(H_t[l], H_b[l], directions[l]) for l in cand}
    log = {"separation": {int(l): sep[l] for l in cand}}

    if force_layer is not None:
        log.update(selected_by="force_layer", forced=int(force_layer))
        return force_layer, directions[force_layer], log

    screened = sorted(cand, key=lambda l: sep[l], reverse=True)[:top_k]
    log["screened"] = screened
    print("[select_direction] 分離度上位: " + ", ".join(f"L{l}(d={sep[l]:.2f})" for l in screened))

    if not do_validation or not val_pairs:
        log["selected_by"] = "separation_only_no_refusals" if not val_pairs else "separation_only"
        return screened[0], directions[screened[0]], log

    results = []
    for l in screened:
        vr = validate_by_generation(model, tokenizer, directions[l], l, val_pairs, n_val,
                                    repetition_penalty=repetition_penalty)
        vr["separation"] = sep[l]
        results.append(vr)
        print(f"[select_direction]   L{l}: 拒絶率 {vr['refusal_rate_after']:.2f}, "
              f"空率 {vr['empty_rate_after']:.2f}")
    results.sort(key=lambda v: (v["refusal_rate_after"], v["empty_rate_after"], -v["separation"]))
    best = results[0]["hidden_layer"]
    log.update(validation=results, selected_by="generation_validation")
    print(f"[select_direction] 採用層(hidden index) = {best}")
    return best, directions[best], log


# ---------------------------------------------------------------------------
def run(engine=None, model=None, tokenizer=None, model_key: str = DEFAULT_MODEL_KEY,
        dataset_path: str = REFUSAL_PAIRS_PATH, assets_root: str = ASSETS_ROOT,
        cache_root: str = CACHE_ROOT, max_new_tokens=48, top_k=5, n_val=20,
        batch_size=16, skip_behavioral_filter=False, no_generation_validation=False,
        repetition_penalty=1.0, min_kept_pairs=10, on_few_pairs="fallback",
        force_layer=None):
    model, tokenizer = unpack_model(engine, model, tokenizer)
    n_layers = num_hidden_layers(model)
    all_layers = list(range(n_layers + 1))
    pairs = load_refusal_pairs(dataset_path)

    log_dir = os.path.join(cache_root, model_key, "refusal")
    os.makedirs(log_dir, exist_ok=True)

    if skip_behavioral_filter:
        kept = [{**p, "target_refused": -1, "baseline_refused": -1} for p in pairs]
        dropped, records, filtered, val_pairs = [], None, False, kept
    else:
        kept, dropped = behavioral_filter(model, tokenizer, pairs, max_new_tokens,
                                          repetition_penalty)
        records, filtered, val_pairs = kept + dropped, True, kept
        with open(os.path.join(log_dir, "behavioral_filter_log.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if filtered and len(kept) < min_kept_pairs:
        msg = (f"行動フィルタの通過ペアが {len(kept)} 件（最小 {min_kept_pairs} 件）でした。"
               f"{log_dir}/behavioral_filter_log.jsonl の target_generation を確認してください。")
        if on_few_pairs == "fallback":
            print(f"[run][WARNING] {msg}\n[run] fallback: 方向は全{len(pairs)}ペア、"
                  f"検証は実拒絶targetのみで続行します。")
            kept = [{**p, "target_refused": -1, "baseline_refused": -1} for p in pairs]
            val_pairs = [r for r in records if r["target_refused"] == 1]
            filtered = False
        elif on_few_pairs == "skip":
            print(f"[run][SKIP] {msg}\n拒絶ベクトルを作らずに終了します。")
            return None
        else:
            raise RuntimeError(msg)
    if not kept:
        raise RuntimeError("使用できるペアが0件です。")

    print(f"[run] 全層の prompt_last 隠れ状態を抽出中 (各{len(kept)}件)...")
    H_t = extract_activations(model, tokenizer, [p["target"] for p in kept], all_layers,
                              pooling="last_token", batch_size=batch_size)
    H_b = extract_activations(model, tokenizer, [p["baseline"] for p in kept], all_layers,
                              pooling="last_token", batch_size=batch_size)
    directions = {l: H_t[l].mean(0) - H_b[l].mean(0) for l in H_t}

    sel_layer, sel_vec, sel_log = select_direction(
        directions, H_t, H_b, n_layers, model, tokenizer, val_pairs, top_k=top_k,
        n_val=n_val, do_validation=not no_generation_validation,
        repetition_penalty=repetition_penalty, force_layer=force_layer)

    out_dir = os.path.join(assets_root, "refusal_vectors", model_key)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "directions.npz"),
             **{f"layer_{l}": v.astype(np.float32) for l, v in directions.items()})
    r_hat = sel_vec / (np.linalg.norm(sel_vec) + 1e-8)
    np.save(os.path.join(out_dir, "r_hat.npy"), r_hat.astype(np.float32))
    with open(os.path.join(out_dir, "selected.json"), "w", encoding="utf-8") as f:
        json.dump({"model_key": model_key, "selected_hidden_layer": int(sel_layer),
                   "selection_log": sel_log, "dim": int(sel_vec.shape[0])},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_key": model_key,
            "dataset_path": os.path.relpath(dataset_path, common.BACKEND_DIR),
            "n_pairs_total": len(pairs), "n_kept": len(kept), "n_dropped": len(dropped),
            "behavioral_filter_applied": filtered, "n_validation_pairs": len(val_pairs),
            "kept_by_pair_type": dict(Counter(r.get("pair_type", "unknown") for r in kept)),
            "max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty,
            "num_hidden_layers": n_layers, "method": "arditi_directional_ablation",
            "pooling": "last_token",
        }, f, ensure_ascii=False, indent=2)
    print(f"[run] 保存 -> {out_dir} (採用層={sel_layer})")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", default=DEFAULT_MODEL_KEY)
    p.add_argument("--dataset_path", default=REFUSAL_PAIRS_PATH)
    p.add_argument("--assets_root", default=ASSETS_ROOT)
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--skip_behavioral_filter", action="store_true")
    p.add_argument("--no_generation_validation", action="store_true")
    p.add_argument("--force_layer", type=int, default=None)
    p.add_argument("--min_kept_pairs", type=int, default=10)
    p.add_argument("--on_few_pairs", default="fallback", choices=["error", "fallback", "skip"])
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    a = p.parse_args()
    run(engine=load_engine(a.model_key), model_key=a.model_key, dataset_path=a.dataset_path,
        assets_root=a.assets_root, cache_root=a.cache_root, max_new_tokens=a.max_new_tokens,
        top_k=a.top_k, n_val=a.n_val, batch_size=a.batch_size,
        skip_behavioral_filter=a.skip_behavioral_filter,
        no_generation_validation=a.no_generation_validation,
        repetition_penalty=a.repetition_penalty, min_kept_pairs=a.min_kept_pairs,
        on_few_pairs=a.on_few_pairs, force_layer=a.force_layer)
