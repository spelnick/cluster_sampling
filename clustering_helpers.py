import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_best_k(df: pd.DataFrame, col: str, max_k: int):
    # Check that max_k doesn't exceed number of observations minus 1
    if len(df) < max_k:
        max_k = len(df) - 1

    silhouette_scores = []
    valid_embeddings = []

    # Extract embedding column
    embeddings = df[col]

    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            valid_embeddings.append(emb)
        else:
            valid_embeddings.append(np.array(emb))

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(np.array(valid_embeddings))
        silhouette_avg = silhouette_score(valid_embeddings, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Find index of highest score (+2 on the end since we started with k=2 in for loop above)
    best_k = np.argmax(silhouette_scores) + 2

    return best_k


def kmeans_clustering(df: pd.DataFrame, col: str, best_k: int):
    valid_embeddings = []

    # Extract embedding column
    embeddings = df[col]

    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            valid_embeddings.append(emb)
        else:
            valid_embeddings.append(np.array(emb))

    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(valid_embeddings)

    # Add cluster column
    df['kmeans_cluster'] = kmeans.labels_

    return df, kmeans
