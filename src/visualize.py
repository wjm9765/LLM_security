import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_pca_safety_collapse(d1_vectors, d2_vectors, d3_vectors, d4_vectors, output_path: str):
    # Fit PCA only on D1 (Safe) and D2 (Harmful)
    pca = PCA(n_components=2)
    fit_data = np.vstack([d1_vectors, d2_vectors])
    pca.fit(fit_data)

    # Project all datasets
    d1_pca = pca.transform(d1_vectors)
    d2_pca = pca.transform(d2_vectors)
    d3_pca = pca.transform(d3_vectors)
    d4_pca = pca.transform(d4_vectors)

    plt.figure(figsize=(10, 8))
    
    plt.scatter(d1_pca[:, 0], d1_pca[:, 1], c='blue', label='D1 (Safe Text)', alpha=0.6)
    plt.scatter(d2_pca[:, 0], d2_pca[:, 1], c='red', label='D2 (Harmful Text)', alpha=0.6)
    plt.scatter(d3_pca[:, 0], d3_pca[:, 1], c='orange', label='D3 (Harmful Text + Noise)', alpha=0.6, marker='x')
    plt.scatter(d4_pca[:, 0], d4_pca[:, 1], c='purple', label='D4 (Harmful Audio + Text)', alpha=0.6, marker='^')

    plt.title("PCA Projection of Activation Vectors (Safety Collapse)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
