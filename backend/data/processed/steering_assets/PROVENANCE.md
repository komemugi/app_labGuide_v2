# 研究成果物の出どころ

このディレクトリの成果物は、研究リポジトリ（linear_rep_geometry_iiwas）で作成したものを
そのままコピーしている。`backend/scripts/` は同じ手順を移植したもので、再生成できる。

| ファイル | 内容 | 作成手順 |
|---|---|---|
| `steering_vectors/<model>/layer_21/vectors.npz` | 8感情のステアリングベクトル | 生成条件（assistant役・応答トークン平均）の活性化から CAA＋中立PCA（累積寄与率50%、14成分） |
| `steering_vectors/<model>/layer_21/meta.json` | 抽出条件、avg_norm | 同上（avg_norm は中立文739件の活性化ノルム平均） |
| `emotion_probe/probe_layer21.npz` | 線形感情プローブ（8感情＋neutral） | 同じ活性化でロジスティック回帰（test acc 0.883） |
| `refusal_vectors/<model>/r_hat.npy` | 拒絶方向 | Arditi et al. の方向除去（拒絶ペアの差分、生成検証で層を選定） |
| `alpha_cap/chatbot_alpha_cap.json` | α上限 | WRIME中立文の続き生成を LLM（gemma-4-12b-it）で評価し、usable率などの基準で決定 |

- モデル: tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5（bf16）
- 層番号は hidden_states のインデックス（21 = model.model.layers[20] の出力）
- α上限の基準（`criteria`）: usable率 ≥ 0.9、崩壊率 ≤ 0.1、拒絶率 ≤ 0.1、言語混入率 ≤ 0.2。
  `per_emotion` は感情ごとの上限、`_model_cap` は全感情の最小値。
  評価は「システムプロンプト・会話履歴なしの1回きりの続き生成」で行った値である点に注意。
