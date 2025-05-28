# In Untitled0.ipynb

!pip install sentence_transformers
!pip install umap
!pip install sklearn
!pip install hdbscan

import os
import re
import json
import random
import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt

# --- Set Random Seed for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# For UMAP, we'll set random_state=SEED in the reducer
# For HDBSCAN, the clustering is deterministic unless soft clustering/prediction is used

# --- Config ---
INPUT_CSV = ""
OUTPUT_DIR = "semantic_similarity_outputs_hierarchical"
ID_COLUMN = "model_id"
CARD_COLUMN = "card"
META_COLUMN = "metadata"
DEPTH_COLUMN = "depth"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helpers ---
def safe_parse_metadata(md_str):
    if isinstance(md_str, dict):
        return md_str
    if pd.isna(md_str) or not isinstance(md_str, str) or not md_str.strip():
        return {}
    try:
        return json.loads(md_str)
    except json.JSONDecodeError:
        fixed = md_str.replace("'", '"')
        fixed = re.sub(r",\s*}", "}", fixed)
        try:
            return json.loads(fixed)
        except Exception:
            return {}

def extract_text_for_model(row):
    card = row.get(CARD_COLUMN)
    if pd.notna(card) and str(card).strip() and str(card).lower() != "n/a":
        return str(card)
    md = safe_parse_metadata(row.get(META_COLUMN, {}))
    if isinstance(md, dict):
        cd = md.get("card_data")
        if isinstance(cd, str) and cd.strip():
            return cd
        pt = md.get("pipeline_tag")
        if isinstance(pt, str) and pt.strip():
            return pt
    return str(row.get(ID_COLUMN, ""))

# --- Load Data ---
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False, quotechar='"', escapechar='\\', on_bad_lines='skip')
except Exception:
    df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines='skip')

df["text_for_embedding"] = df.apply(extract_text_for_model, axis=1)
df[DEPTH_COLUMN] = pd.to_numeric(df[DEPTH_COLUMN], errors='coerce').fillna(0).astype(int)
texts = df["text_for_embedding"].astype(str).tolist()
model_ids = df[ID_COLUMN].astype(str).tolist()
depths = df[DEPTH_COLUMN].tolist()

# --- Embedding & Similarity ---
class ModelEmbeddings:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=2000)
        self.bert = SentenceTransformer('all-MiniLM-L6-v2')
        self.mpnet = SentenceTransformer('all-mpnet-base-v2')
    def tfidf_emb(self, texts: List[str]) -> np.ndarray:
        return self.tfidf.fit_transform(texts).toarray()
    def bert_emb(self, texts: List[str]) -> np.ndarray:
        return self.bert.encode(texts, show_progress_bar=True)
    def mpnet_emb(self, texts: List[str]) -> np.ndarray:
        return self.mpnet.encode(texts, show_progress_bar=True)

def hybrid_similarity(emb: np.ndarray, depths: List[int], alpha=0.8):
    sem_sim = cosine_similarity(emb)
    depth_diff = np.abs(np.subtract.outer(depths, depths))
    depth_sim = 1 / (1 + depth_diff)
    return alpha * sem_sim + (1 - alpha) * depth_sim

def reduce2d(emb: np.ndarray, method="umap") -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=SEED)
    else:
        reducer = umap.UMAP(n_components=2, random_state=SEED)
    return reducer.fit_transform(emb)

def cluster(emb: np.ndarray) -> np.ndarray:
    # HDBSCAN is deterministic unless soft clustering/prediction is used
    return hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(emb)

def plot2d(pts, labels, title, path):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(sc)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# --- Pipeline ---
emb = ModelEmbeddings()
bert_embs = emb.bert_emb(texts)
mpnet_embs = emb.mpnet_emb(texts)

bert_sim_hybrid = hybrid_similarity(bert_embs, depths)
mpnet_sim_hybrid = hybrid_similarity(mpnet_embs, depths)

bert_2d = reduce2d(bert_embs)
mpnet_2d = reduce2d(mpnet_embs)

bert_cl = cluster(bert_embs)
mpnet_cl = cluster(mpnet_embs)

plot2d(bert_2d, bert_cl, "BERT + Depth Embedding Clusters", os.path.join(OUTPUT_DIR, "bert_clusters.png"))
plot2d(mpnet_2d, mpnet_cl, "MPNet + Depth Embedding Clusters", os.path.join(OUTPUT_DIR, "mpnet_clusters.png"))

pd.DataFrame(bert_sim_hybrid, index=model_ids, columns=model_ids).to_csv(os.path.join(OUTPUT_DIR, "bert_similarity.csv"))
pd.DataFrame(mpnet_sim_hybrid, index=model_ids, columns=model_ids).to_csv(os.path.join(OUTPUT_DIR, "mpnet_similarity.csv"))

cluster_df = pd.DataFrame({
    ID_COLUMN: model_ids,
    "depth": depths,
    "bert_cluster": bert_cl,
    "mpnet_cluster": mpnet_cl
})
cluster_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_assignments.csv"), index=False)

cluster_df.head()
