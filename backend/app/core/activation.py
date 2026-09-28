# backend/app/core/activation.py
"""
隠れ状態（活性化）の抽出を1か所にまとめたモジュール。

研究リポジトリ linear_rep_geometry_iiwas の src/activation/extract.py と
model_loader.build_prompt を移植したもの。**抽出ロジックは研究側と一字一句同じ**に
保っている（同梱の成果物 vectors.npz / probe_layer21.npz を再現できるようにするため）。

このモジュールを使う場所:
  - scripts/*（成果物の生成時）
  - app/core/emotion_probe.py（チャット中にユーザー発話の感情を読むとき）
学習時と推論時で**同じ関数**を通すことで、入力形式のずれが構造的に起きないようにする。

2つの抽出位置:
  prompt_last   : ユーザー発話をuser役に置き、generation prompt直後の最終トークン
                  （理解条件。拒絶ベクトルに使う）
  response_mean : 文をassistant役に置き、応答トークン全体を平均
                  （生成条件。感情ベクトルと感情プローブに使う）

層番号は hidden_states のインデックス（0=埋め込み層、i+1 = model.model.layers[i] の出力）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# 研究側 extract_activations_at_response の既定値。変えると成果物と条件がずれる。
RESPONSE_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
RESPONSE_USER_PROMPTS = ["どう思う？", "何かあった？", "聞かせて。", "話して。"]

POSITIONS = ("prompt_last", "response_mean")


def build_prompt(tokenizer, user_content: str, system_content: Optional[str] = None,
                 add_generation_prompt: bool = True, use_chat_template: bool = True) -> str:
    """チャットテンプレートでプロンプト文字列を作る（研究側 model_loader.build_prompt と同一）。

    チャットテンプレートを持たないトークナイザでは素のテキストを返す。
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if not use_chat_template or not has_template:
        return user_content
    messages = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _prepare_tokenizer(tokenizer) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def apply_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                  padding_side: str, method: str = "last_token") -> torch.Tensor:
    """(batch, seq, dim) → (batch, dim)。左右どちらのパディングでも最後の実トークンを取る。"""
    if method == "last_token":
        if padding_side == "right":
            last_idx = attention_mask.sum(dim=1) - 1
        else:
            last_idx = torch.full((hidden_states.size(0),), hidden_states.size(1) - 1,
                                  device=hidden_states.device, dtype=torch.long)
        return hidden_states[torch.arange(hidden_states.size(0),
                                          device=hidden_states.device), last_idx]
    if method == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"Unknown pooling method: {method}")


@torch.no_grad()
def extract_activations(model, tokenizer, texts: list, layers: list,
                        pooling: str = "last_token", batch_size: int = 16,
                        max_length: int = 256, use_chat_template: bool = True,
                        system_prompt: Optional[str] = None,
                        verbose: bool = True) -> dict:
    """prompt_last 位置の抽出。戻り値は {層: (n_texts, dim) float32}。"""
    _prepare_tokenizer(tokenizer)
    out = {l: [] for l in layers}
    n_batches = -(-len(texts) // batch_size)

    for batch_idx, start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[start:start + batch_size]
        if use_chat_template:
            batch = [build_prompt(tokenizer, t, system_content=system_prompt,
                                  add_generation_prompt=True) for t in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        for l in layers:
            pooled = apply_pooling(outputs.hidden_states[l], inputs["attention_mask"],
                                   tokenizer.padding_side, pooling)
            out[l].append(pooled.to(torch.float32).cpu().numpy())

        if verbose and start == 0:
            print(f"[extract_activations] 入力例:\n{batch[0][:200]}\n")
        if verbose and ((batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches):
            print(f"[extract_activations]   batch {batch_idx + 1}/{n_batches} 処理済み")
        del outputs
        if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return {l: np.concatenate(v, axis=0) for l, v in out.items()}


@torch.no_grad()
def extract_activations_at_response(model, tokenizer, texts: list, layers: list,
                                    batch_size: int = 16, max_length: int = 512,
                                    system_prompt: Optional[str] = None,
                                    user_prompts: Optional[list] = None,
                                    verbose: bool = True) -> dict:
    """response_mean 位置の抽出。戻り値は {層: (n_texts, dim) float32}。

    文をassistant役に置き、応答トークン全体を平均する。ユーザー発話は
    user_prompts からインデックスの剰余で決定的に割り当てる（研究側と同一）。

    ★研究側との互換のため、以下の挙動も**そのまま**残している（変更しないこと）:
      - プレフィックス長は add_special_tokens=False で数え、本体は既定（True）で
        トークナイズする。テンプレートがBOSを含むモデルでは平均区間が1トークン
        前にずれるが、同梱の成果物はこの条件で作られている。
    """
    _prepare_tokenizer(tokenizer)
    sys_p = system_prompt or RESPONSE_SYSTEM_PROMPT
    ups = user_prompts or RESPONSE_USER_PROMPTS

    out = {L: [] for L in layers}
    n_batches = -(-len(texts) // batch_size)
    for bi, start in enumerate(range(0, len(texts), batch_size)):
        chunk = texts[start:start + batch_size]
        prefixes, fulls = [], []
        for j, t in enumerate(chunk):
            u = ups[(start + j) % len(ups)]
            try:
                base = tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_p},
                     {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                base = f"System: {sys_p}\nUser: {u}\nAssistant: "
            prefixes.append(base)
            fulls.append(base + t)

        enc = tokenizer(fulls, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(model.device)
        plens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefixes]
        tlens = enc["attention_mask"].sum(1).tolist()

        res = model(**enc, output_hidden_states=True)
        for L in layers:
            h = res.hidden_states[L]
            for k in range(len(chunk)):
                s, e = plens[k], tlens[k]
                if e <= s:
                    e = min(s + 1, h.shape[1])
                out[L].append(h[k, s:e, :].mean(0).float().cpu().numpy())

        del res
        if torch.cuda.is_available() and (bi + 1) % 20 == 0:
            torch.cuda.empty_cache()
        if verbose and ((bi + 1) % 10 == 0 or bi + 1 == n_batches):
            print(f"[extract@response]   batch {bi+1}/{n_batches}", flush=True)

    return {L: np.stack(v).astype(np.float32) for L, v in out.items()}


def extract(model, tokenizer, texts: list, layers: list, position: str,
            batch_size: int = 16, verbose: bool = True) -> dict:
    """position 名で2つの抽出関数を切り替える窓口。"""
    if position == "response_mean":
        return extract_activations_at_response(model, tokenizer, texts, layers,
                                               batch_size=batch_size, verbose=verbose)
    if position == "prompt_last":
        return extract_activations(model, tokenizer, texts, layers,
                                   pooling="last_token", batch_size=batch_size,
                                   verbose=verbose)
    raise ValueError(f"unknown position: {position}（{POSITIONS} のいずれか）")


def position_for_dataset(dataset: str) -> str:
    """成果物に記録された dataset 名から抽出位置を決める。

    研究側の命名: synthetic_resp = response_mean、それ以外（synthetic / synthetic_promptlast）
    = prompt_last。
    """
    return "response_mean" if str(dataset).endswith("_resp") else "prompt_last"
