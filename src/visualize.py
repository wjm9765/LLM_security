import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_pca_safety_collapse(d1_vectors, d2_vectors, d3_vectors, d4_vectors, d5_vectors, d6_vectors, output_path: str):
    # Fit PCA only on D1 (Safe) and D2 (Harmful)
    pca = PCA(n_components=2)
    fit_data = np.vstack([d1_vectors, d2_vectors])
    pca.fit(fit_data)

    # Project all datasets
    d1_pca = pca.transform(d1_vectors)
    d2_pca = pca.transform(d2_vectors)
    d3_pca = pca.transform(d3_vectors)
    d4_pca = pca.transform(d4_vectors)
    d5_pca = pca.transform(d5_vectors)
    d6_pca = pca.transform(d6_vectors)

    plt.figure(figsize=(12, 10))
    
    plt.scatter(d1_pca[:, 0], d1_pca[:, 1], c='blue', label='D1 (Safe Text)', alpha=0.5, marker='o')
    plt.scatter(d2_pca[:, 0], d2_pca[:, 1], c='red', label='D2 (Harmful Text)', alpha=0.5, marker='o')
    
    plt.scatter(d3_pca[:, 0], d3_pca[:, 1], c='orange', label='D3 (Noise + Harmful)', alpha=0.6, marker='x')
    plt.scatter(d4_pca[:, 0], d4_pca[:, 1], c='purple', label='D4 (Audio + Harmful)', alpha=0.6, marker='^')
    
    plt.scatter(d5_pca[:, 0], d5_pca[:, 1], c='cyan', label='D5 (Noise + Safe)', alpha=0.6, marker='s')
    plt.scatter(d6_pca[:, 0], d6_pca[:, 1], c='green', label='D6 (Audio + Safe)', alpha=0.6, marker='D')

    plt.title("PCA Projection of Activation Vectors (Safety Collapse Analysis)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
