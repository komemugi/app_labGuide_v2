"""
backend/app/research/caa.py（研究リポジトリ src/steering/caa.py をそのまま移植）

CAA（Contrastive Activation Addition）+ 中立PCA によるステアリングベクトルの計算。
（旧 steering_vector.py から分離。2026-07-24 のディレクトリ再編で移動）

【手法】Sofroniew et al. (2026) の式(20):
    v_e = (I - U_neu @ U_neu^T) @ ( mean(H_e) - mean(H_all) )

★2026-07-24 訂正: 以前このファイル（旧steering_vector.py）には
「ユニーク効果ベクトル」（Tak et al. 2025、他感情ベクトルへの射影除去）を
CAAベクトルに追加適用する機能を入れていたが、ユーザーから
「ユニーク効果ベクトルはCAAとは別枠のステアリング手法であり、CAAに
交わるものではない」という指摘を受け、削除した。
（実際、CAAベクトルには Σ_e n_e*v_e = 0 という恒等的な制約があり、
 他感情の張る部分空間をまるごと除去する版は数学的に必ずゼロになる、
 という問題も発覚していた。history.md V節に経緯を記録。）
「ユニーク効果ベクトル」を独立した手法として実装する場合は、
将来 src/steering/unique_effect.py のような別ファイルとして追加すること
（CAAのモジュールに混ぜない）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os

import numpy as np


def compute_neutral_pca(H_neutral: np.ndarray, explained_variance_threshold: float = 0.5):
    """中立コーパスの hidden state に対してPCAを行い、累積寄与率が閾値に達するまでの
    主成分（U_neu）を返す。

    Parameters
    ----------
    H_neutral : (n_samples, dim)
    explained_variance_threshold : float
        累積寄与率の閾値（既定0.5＝50%。Sofroniew et al.の設定）

    Returns
    -------
    U_neu : (dim, r)  上位r主成分を列に持つ行列
    info : dict       r や実際の累積寄与率などの情報

    Notes
    -----
    - PCAは標準的に「平均を引いてから」行う（sklearnのPCAも同様）。
    - U_neu は**全感情で共通のものを1回だけ計算して使い回す**
      （感情ごとにPCAし直すのではない点に注意。論文の設計通り）。
    """
    from sklearn.decomposition import PCA

    n_samples, dim = H_neutral.shape
    max_components = min(n_samples, dim)
    pca = PCA(n_components=max_components, svd_solver="auto")
    pca.fit(H_neutral)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    r = int(np.searchsorted(cumsum, explained_variance_threshold) + 1)
    r = max(1, min(r, max_components))

    U_neu = pca.components_[:r].T  # (dim, r)
    info = {
        "n_components": r,
        "explained_variance_ratio_cumulative": float(cumsum[r - 1]),
        "threshold": explained_variance_threshold,
        "n_neutral_samples": int(n_samples),
    }
    print(f"[compute_neutral_pca] r={r} 主成分で累積寄与率 {cumsum[r-1]:.1%} "
          f"(閾値 {explained_variance_threshold:.0%})")
    return U_neu, info


def remove_neutral_subspace(v: np.ndarray, U_neu: np.ndarray) -> np.ndarray:
    """(I - U_neu U_neu^T) v を計算する（中立部分空間への射影を除去）。

    dim×dim の巨大な行列 I - U U^T を作らずに、
        v - U @ (U^T @ v)
    と計算する（数学的に同値で、メモリ・計算量ともに大幅に節約できる）。
    """
    return v - U_neu @ (U_neu.T @ v)


def compute_caa_vectors(
    H_by_emotion: dict[str, np.ndarray],
    U_neu: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sofroniew et al. 式(20) のステアリングベクトルを感情ごとに計算する。

        v_e = (I - U_neu U_neu^T) ( mean(H_e) - mean(H_all) )

    H_all は「全感情のサンプルをまとめたもの」。
    ★注意：mean(H_all) は「各感情の平均をさらに平均したもの」ではなく
    「全サンプルをまとめた平均」。感情ごとのサンプル数が違うと両者は一致しないため、
    ここでは論文の記述通り「全サンプルの平均」を使う
    （サンプル数の偏りが気になる場合は、load_wrimeの max_per_emotion で
      感情ごとの件数を揃えてから渡すこと）。
    """
    H_all = np.concatenate([H_by_emotion[e] for e in sorted(H_by_emotion.keys())], axis=0)
    mean_all = H_all.mean(axis=0)

    vectors = {}
    for emo, H_e in H_by_emotion.items():
        diff = H_e.mean(axis=0) - mean_all
        v = remove_neutral_subspace(diff, U_neu)
        vectors[emo] = v
        print(f"[compute_caa_vectors] {emo:13s}: ||diff||={np.linalg.norm(diff):.4f} "
              f"-> ||v||={np.linalg.norm(v):.4f} "
              f"(中立成分除去で {1 - np.linalg.norm(v)/max(np.linalg.norm(diff),1e-12):.1%} 減衰)")
    return vectors


def compute_avg_norm(H: np.ndarray) -> float:
    """その層の「典型的な活性化ノルム」= サンプルごとのL2ノルムの平均。
    αのスケール正規化に使う（h' = h + alpha * avg_norm * v_hat）。
    ★感情データではなく中立文（gemma生成の中立発話）で計算する
    （感情特有のノルム変動の影響を受けないようにするため）。
    """
    return float(np.linalg.norm(H, axis=1).mean())


@dataclass
class SteeringVectorSet:
    """1モデル・1層分のステアリングベクトル一式。"""
    model_key: str
    layer: int
    vectors: dict            # emotion -> np.ndarray (dim,)
    avg_norm: float          # その層の典型的な活性化ノルム
    meta: dict = field(default_factory=dict)

    def save(self, root: str = "steering_vectors") -> str:
        out_dir = os.path.join(root, self.model_key, f"layer_{self.layer}")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "vectors.npz"),
            **{k: v for k, v in self.vectors.items()},
        )
        meta = dict(self.meta)
        meta.update({"model_key": self.model_key, "layer": self.layer, "avg_norm": self.avg_norm})
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[SteeringVectorSet.save] -> {out_dir}")
        return out_dir

    @classmethod
    def load(cls, model_key: str, layer: int, root: str = "steering_vectors") -> "SteeringVectorSet":
        out_dir = os.path.join(root, model_key, f"layer_{layer}")
        npz = np.load(os.path.join(out_dir, "vectors.npz"))
        vectors = {k: npz[k] for k in npz.files}
        with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(model_key=model_key, layer=layer, vectors=vectors,
                   avg_norm=meta["avg_norm"], meta=meta)
