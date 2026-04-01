import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_pca_safety_collapse(d1_vectors, d2_vectors, d3_vectors, d4_vectors, d5_vectors, d6_vectors, tts_engines, output_path: str):
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

    plt.figure(figsize=(14, 10))
    
    plt.scatter(d1_pca[:, 0], d1_pca[:, 1], c='blue', label='D1 (Safe Text)', alpha=0.5, marker='o')
    plt.scatter(d2_pca[:, 0], d2_pca[:, 1], c='red', label='D2 (Harmful Text)', alpha=0.5, marker='o')
    plt.scatter(d3_pca[:, 0], d3_pca[:, 1], c='orange', label='D3 (Noise + Harmful)', alpha=0.6, marker='x')
    plt.scatter(d5_pca[:, 0], d5_pca[:, 1], c='cyan', label='D5 (Noise + Safe)', alpha=0.6, marker='s')
    
    # Plot D4 and D6 separated by TTS engine
    unique_engines = list(set(tts_engines))
    colors_d4 = {'gtts': 'purple', 'edge-tts': 'darkviolet'}
    colors_d6 = {'gtts': 'green', 'edge-tts': 'limegreen'}
    markers = {'gtts': '^', 'edge-tts': 'v'}
    
    for engine in unique_engines:
        idx = [i for i, e in enumerate(tts_engines) if e == engine]
        
        plt.scatter(d4_pca[idx, 0], d4_pca[idx, 1], 
                    c=colors_d4.get(engine, 'purple'), 
                    label=f'D4 (Audio + Harmful) - {engine}', 
                    alpha=0.7, 
                    marker=markers.get(engine, '^'))
                    
        plt.scatter(d6_pca[idx, 0], d6_pca[idx, 1], 
                    c=colors_d6.get(engine, 'green'), 
                    label=f'D6 (Audio + Safe) - {engine}', 
                    alpha=0.7, 
                    marker=markers.get(engine, 'D'))

    plt.title("PCA Projection of Activation Vectors by TTS Engine")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_mechanism_directions(projections, tts_engines, output_path: str):
    plt.figure(figsize=(14, 10))
    
    plt.scatter(projections["D1"][0], projections["D1"][1], c='blue', label='D1 (Safe Text)', alpha=0.5, marker='o')
    plt.scatter(projections["D2"][0], projections["D2"][1], c='red', label='D2 (Harmful Text)', alpha=0.5, marker='o')
    plt.scatter(projections["D3"][0], projections["D3"][1], c='orange', label='D3 (Noise + Harmful)', alpha=0.6, marker='x')
    plt.scatter(projections["D5"][0], projections["D5"][1], c='cyan', label='D5 (Noise + Safe)', alpha=0.6, marker='s')
    
    unique_engines = list(set(tts_engines))
    colors_d4 = {'gtts': 'purple', 'edge-tts': 'darkviolet'}
    colors_d6 = {'gtts': 'green', 'edge-tts': 'limegreen'}
    markers = {'gtts': '^', 'edge-tts': 'v'}
    
    for engine in unique_engines:
        idx = [i for i, e in enumerate(tts_engines) if e == engine]
        
        plt.scatter(projections["D4"][0][idx], projections["D4"][1][idx], 
                    c=colors_d4.get(engine, 'purple'), 
                    label=f'D4 (Audio + Harmful) - {engine}', 
                    alpha=0.7, 
                    marker=markers.get(engine, '^'))
                    
        plt.scatter(projections["D6"][0][idx], projections["D6"][1][idx], 
                    c=colors_d6.get(engine, 'green'), 
                    label=f'D6 (Audio + Safe) - {engine}', 
                    alpha=0.7, 
                    marker=markers.get(engine, 'D'))

    plt.title("Mechanism Directions: Harmfulness vs Refusal Projection")
    plt.xlabel("Projection on Harmfulness Direction (D2_inst - D1_inst)")
    plt.ylabel("Projection on Refusal Direction (D2_post - D1_post)")
    
    # Draw reference lines at 0 (D1 centroid) and 1 (if normalized)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
