import os
import glob
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# 1. 検索させたいテキスト情報（一旦は直接リストに書くか、テキストファイルから読み込む）
raw_data_dir = "data/raw/rag_docs"
texts = []
for filepath in glob.glob(f"{raw_data_dir}/*.txt"): # rag_docs内のtxtファイル全てを読み込む処理
    with open(filepath, "r", encoding="utf-8") as f:
        # ファイル内のテキストを読み込み、必要に応じて改行で分割してリスト化
        texts.extend([line.strip() for line in f if line.strip()])

# 2. 埋め込みモデルのロード（日本語に強い定番モデル）
print("モデルをロード中...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# 3. テキストをベクトル化（このモデルは保存する文章の先頭に 'passage: ' をつけるルールがある）
passages = [f"passage: {t}" for t in texts]
embeddings = embedder.encode(passages, normalize_embeddings=True)

# 4. FAISSインデックス（ベクトルDB）の作成と保存
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) # コサイン類似度（内積）用の設定
index.add(embeddings)

# データを backend/data/rag_store/ に保存
output_dir = "data/processed/rag_store"
os.makedirs(output_dir, exist_ok=True)

faiss.write_index(index, f"{output_dir}/data_index.faiss")
with open(f"{output_dir}/data_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("ベクトルDBの作成が完了しました！")