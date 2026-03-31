import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_centroid(vectors: np.ndarray) -> np.ndarray:
    """Calculate the mean vector (centroid) of a set of vectors."""
    return np.mean(vectors, axis=0)

def l2_distance(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Calculate the L2 distance between sets of vectors and a single vector."""
    if len(v1.shape) == 1:
        v1 = v1.reshape(1, -1)
    if len(v2.shape) == 1:
        v2 = v2.reshape(1, -1)
    return np.linalg.norm(v1 - v2, axis=1)

def compute_metrics(d1_vectors, d2_vectors, d3_vectors, d4_vectors):
    d1_centroid = calculate_centroid(d1_vectors)
    d2_centroid = calculate_centroid(d2_vectors)

    metrics = {
        "D3_to_D1_L2": float(np.mean(l2_distance(d3_vectors, d1_centroid))),
        "D3_to_D2_L2": float(np.mean(l2_distance(d3_vectors, d2_centroid))),
        "D4_to_D1_L2": float(np.mean(l2_distance(d4_vectors, d1_centroid))),
        "D4_to_D2_L2": float(np.mean(l2_distance(d4_vectors, d2_centroid))),
        
        "D3_to_D1_cos": float(np.mean(cosine_similarity(d3_vectors, d1_centroid.reshape(1, -1)))),
        "D3_to_D2_cos": float(np.mean(cosine_similarity(d3_vectors, d2_centroid.reshape(1, -1)))),
        "D4_to_D1_cos": float(np.mean(cosine_similarity(d4_vectors, d1_centroid.reshape(1, -1)))),
        "D4_to_D2_cos": float(np.mean(cosine_similarity(d4_vectors, d2_centroid.reshape(1, -1)))),
    }
    return metrics
