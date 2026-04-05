import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

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

def compute_metrics(d1_vectors, d2_vectors, d3_vectors, d4_vectors, d5_vectors, d6_vectors):
    d1_centroid = calculate_centroid(d1_vectors)
    d2_centroid = calculate_centroid(d2_vectors)

    metrics = {
        "D3_to_D1_L2": float(np.mean(l2_distance(d3_vectors, d1_centroid))),
        "D3_to_D2_L2": float(np.mean(l2_distance(d3_vectors, d2_centroid))),
        "D4_to_D1_L2": float(np.mean(l2_distance(d4_vectors, d1_centroid))),
        "D4_to_D2_L2": float(np.mean(l2_distance(d4_vectors, d2_centroid))),
        
        "D5_to_D1_L2": float(np.mean(l2_distance(d5_vectors, d1_centroid))),
        "D5_to_D2_L2": float(np.mean(l2_distance(d5_vectors, d2_centroid))),
        "D6_to_D1_L2": float(np.mean(l2_distance(d6_vectors, d1_centroid))),
        "D6_to_D2_L2": float(np.mean(l2_distance(d6_vectors, d2_centroid))),
        
        "D3_to_D1_cos": float(np.mean(cosine_similarity(d3_vectors, d1_centroid.reshape(1, -1)))),
        "D3_to_D2_cos": float(np.mean(cosine_similarity(d3_vectors, d2_centroid.reshape(1, -1)))),
        "D4_to_D1_cos": float(np.mean(cosine_similarity(d4_vectors, d1_centroid.reshape(1, -1)))),
        "D4_to_D2_cos": float(np.mean(cosine_similarity(d4_vectors, d2_centroid.reshape(1, -1)))),
        
        "D5_to_D1_cos": float(np.mean(cosine_similarity(d5_vectors, d1_centroid.reshape(1, -1)))),
        "D5_to_D2_cos": float(np.mean(cosine_similarity(d5_vectors, d2_centroid.reshape(1, -1)))),
        "D6_to_D1_cos": float(np.mean(cosine_similarity(d6_vectors, d1_centroid.reshape(1, -1)))),
        "D6_to_D2_cos": float(np.mean(cosine_similarity(d6_vectors, d2_centroid.reshape(1, -1)))),
    }
    return metrics

def calculate_direction_projections(d1_inst, d2_inst, d3_inst, d4_inst, d5_inst, d6_inst, 
                                    d1_post, d2_post, d3_post, d4_post, d5_post, d6_post):
    """
    Project vectors onto Harmfulness Direction (D2_inst - D1_inst) 
    and Refusal Direction (D2_post - D1_post).
    Returns dictionary with x and y coordinates for each dataset.
    """
    # Centroids
    c1_inst = calculate_centroid(d1_inst)
    c2_inst = calculate_centroid(d2_inst)
    c1_post = calculate_centroid(d1_post)
    c2_post = calculate_centroid(d2_post)
    
    # Directions
    dir_harm = c2_inst - c1_inst
    dir_ref = c2_post - c1_post
    
    # Normalize directions for scalar projection
    dir_harm_norm = dir_harm / np.linalg.norm(dir_harm)
    dir_ref_norm = dir_ref / np.linalg.norm(dir_ref)
    
    def project(data_inst, data_post):
        # We project centered vectors (relative to D1) for better interpretability
        # data_inst centered around c1_inst
        proj_harm = np.dot(data_inst - c1_inst, dir_harm_norm)
        # data_post centered around c1_post
        proj_ref = np.dot(data_post - c1_post, dir_ref_norm)
        return proj_harm, proj_ref

    datasets = {
        "D1": project(d1_inst, d1_post),
        "D2": project(d2_inst, d2_post),
        "D3": project(d3_inst, d3_post),
        "D4": project(d4_inst, d4_post),
        "D5": project(d5_inst, d5_post),
        "D6": project(d6_inst, d6_post),
    }
    
    return datasets

def calculate_orthogonality_test(d1_inst, d2_inst, d4_inst, d6_inst) -> float:
    """
    Calculate the cosine similarity between the Text Safety Direction 
    (D2 - D1) and the Audio Safety Direction (D4 - D6).
    A value close to 0 proves the safety manifolds are orthogonal and disjoint.
    """
    v_text = calculate_centroid(d2_inst) - calculate_centroid(d1_inst)
    v_audio = calculate_centroid(d4_inst) - calculate_centroid(d6_inst)
    
    v_text_norm = v_text / np.linalg.norm(v_text)
    v_audio_norm = v_audio / np.linalg.norm(v_audio)
    
    cos_sim = float(np.dot(v_text_norm, v_audio_norm))
    return cos_sim

def calculate_paired_svd_projections(d1_inst, d2_inst, d3_inst, d4_inst, d5_inst, d6_inst):
    """
    Calculate Residuals (Delta = D2 - D1).
    Perform PCA (or SVD) on these residuals.
    PC1 -> Pure Text Safety Direction (removing input-specific noise).
    PC2 -> Orthogonal noise factor.
    Project all data onto these two pure axes.
    """
    # 1. Pairing and Residual calculation
    residuals = d2_inst - d1_inst # shape (N, D)
    
    # 2. Extract principal components
    pca = PCA(n_components=2)
    pca.fit(residuals)
    
    # pc1 -> Pure Safety direction (the major axis of change when generating harm vs safe)
    # pc2 -> Orthogonal noise variation
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    
    # Base for projection
    c1_inst = calculate_centroid(d1_inst)

    def svd_project(data):
        centered_data = data - c1_inst
        proj_x = np.dot(centered_data, pc1)
        proj_y = np.dot(centered_data, pc2)
        return proj_x, proj_y

    svd_datasets = {
        "D1": svd_project(d1_inst),
        "D2": svd_project(d2_inst),
        "D3": svd_project(d3_inst),
        "D4": svd_project(d4_inst),
        "D5": svd_project(d5_inst),
        "D6": svd_project(d6_inst),
    }

    return svd_datasets, pca.explained_variance_ratio_
