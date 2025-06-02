'''Outputted data: {'Silhouette Score': np.float64(0.3168252293823175), 'Number of Clusters': 21, 'Top Library (by count)': 'transformers', 'Mean Model Count': np.float64(1740.87), 'Median Model Count': 2.0, 'Exported Tables': {'Top-5 Nearest Neighbors': '/content/tables/top_5_nearest_neighbors.png', 'Cluster Sizes': '/content/tables/cluster_sizes.png', 'Cluster Similarity Heatmap': '/content/tables/centroid_similarity_heatmap.png'}}
Cluster labels: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
Valid clusters: 21'''

# Install required libraries if needed
!pip install pandas scikit-learn matplotlib seaborn hdbscan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import hdbscan

# --- Load embeddings CSV ---
file_path = "/content/model_library_with_embeddings.csv"
df = pd.read_csv(file_path)

# --- Extract embedding vectors ---
embedding_cols = [col for col in df.columns if col.startswith("dim_")]
X = df[embedding_cols].values

# --- Apply HDBSCAN clustering FIRST ---
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, prediction_data=True)
df['Cluster'] = clusterer.fit_predict(X)

# --- 1. Cosine Similarity Matrix ---
cos_sim_matrix = cosine_similarity(X)
nearest_neighbors = {
    lib: df['Library'].iloc[np.argsort(-cos_sim_matrix[i])[1:6]].tolist()
    for i, lib in enumerate(df['Library'])
}
nn_df = pd.DataFrame([
    {"Library": lib, "Top 5 Nearest": ", ".join(neighs)}
    for lib, neighs in nearest_neighbors.items()
])

# --- 2. Cluster Size Distribution ---
cluster_sizes = df['Cluster'].value_counts().reset_index()
cluster_sizes.columns = ['Cluster', 'Size']

# --- 3. Inter-Cluster Distance Matrix ---
cluster_centroids = df.groupby('Cluster')[embedding_cols].mean()
centroid_sim_matrix = cosine_similarity(cluster_centroids.values)
centroid_sim_df = pd.DataFrame(centroid_sim_matrix, index=cluster_centroids.index, columns=cluster_centroids.index)

# --- 4. Silhouette Score ---
if df['Cluster'].nunique() > 1 and -1 not in df['Cluster'].unique():
    sil_score = silhouette_score(X, df['Cluster'])
else:
    sil_score = np.nan

# --- Export PNGs ---
os.makedirs("/content/tables", exist_ok=True)

# Nearest Neighbors Table
fig, ax = plt.subplots(figsize=(12, len(nn_df.head(20)) * 0.4))
ax.axis('off')
table_nn = ax.table(cellText=nn_df.head(20).values, colLabels=nn_df.columns, loc='center', cellLoc='left')
table_nn.auto_set_font_size(False)
table_nn.set_fontsize(10)
plt.title("Top-5 Nearest Libraries (Cosine Similarity)", fontsize=14, weight='bold')
nn_path = "/content/tables/top_5_nearest_neighbors.png"
plt.savefig(nn_path, dpi=300, bbox_inches='tight')

# Cluster Sizes Table
fig, ax = plt.subplots(figsize=(6, len(cluster_sizes) * 0.4))
ax.axis('off')
table_cs = ax.table(cellText=cluster_sizes.values, colLabels=cluster_sizes.columns, loc='center')
table_cs.auto_set_font_size(False)
table_cs.set_fontsize(10)
plt.title("Cluster Size Distribution", fontsize=14, weight='bold')
cs_path = "/content/tables/cluster_sizes.png"
plt.savefig(cs_path, dpi=300, bbox_inches='tight')

# Cluster Centroid Similarity Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    centroid_sim_df,
    cmap="viridis",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},  # smaller font size
    ax=ax
)
plt.title("Inter-Cluster Cosine Similarity", fontsize=14, weight='bold')
plt.tight_layout()
centroid_path = "/content/tables/centroid_similarity_heatmap.png"
plt.savefig(centroid_path, dpi=300)

non_noise_mask = df['Cluster'] != -1
if df.loc[non_noise_mask, 'Cluster'].nunique() > 1:
    sil_score = silhouette_score(X[non_noise_mask], df.loc[non_noise_mask, 'Cluster'])
else:
    sil_score = np.nan



# --- Summary Output ---
summary = {

    "Silhouette Score": sil_score,
    "Number of Clusters": df['Cluster'].nunique(),
    "Top Library (by count)": df.loc[df['Model Count'].idxmax(), 'Library'],
    "Mean Model Count": round(df['Model Count'].mean(), 2),
    "Median Model Count": df['Model Count'].median(),
    "Exported Tables": {
        "Top-5 Nearest Neighbors": nn_path,
        "Cluster Sizes": cs_path,
        "Cluster Similarity Heatmap": centroid_path
    }
}

print(summary)

print("Cluster labels:", np.unique(df['Cluster']))
print("Valid clusters:", df['Cluster'].nunique())
