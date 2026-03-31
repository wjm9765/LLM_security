import sys
import numpy as np
from pathlib import Path
import librosa

# Add project root to sys.path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_utils import load_config, load_json, save_json
from src.model_utils import TargetModelWrapper
from src.metrics import compute_metrics
from src.visualize import plot_pca_safety_collapse

def load_audio(path, target_sr=16000):
    audio, sr = librosa.load(path, sr=target_sr)
    return audio

def main():
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config(base_dir / "config.yaml")
    
    inputs_dir = base_dir / "inputs"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    datasets = load_json(inputs_dir / "processed_datasets.json")
    
    print(f"Loading target model: {config['target_model']['model_id']}")
    model_wrapper = TargetModelWrapper(
        model_id=config["target_model"]["model_id"],
        device=config["target_model"]["device"]
    )
    
    layer_idx = config["target_model"]["target_layer"]
    
    d1_vectors, d2_vectors, d3_vectors, d4_vectors = [], [], [], []
    
    print("Running forward passes to extract hidden states...")
    for item in datasets:
        # D1: Safe
        messages_d1 = [
            {"role": "system", "content": item["d1"]["system"]},
            {"role": "user", "content": item["d1"]["user"]}
        ]
        h1 = model_wrapper.get_hidden_states(messages_d1, target_layer=layer_idx)
        d1_vectors.append(h1[0]) # batch_size=1, taking the sequence element
        
        # D2: Harmful Text
        messages_d2 = [
            {"role": "system", "content": item["d2"]["system"]},
            {"role": "user", "content": item["d2"]["user"]}
        ]
        h2 = model_wrapper.get_hidden_states(messages_d2, target_layer=layer_idx)
        d2_vectors.append(h2[0])
        
        # D3: Harmful Text + Noise Audio
        audio_d3 = load_audio(item["d3"]["audio"], target_sr=config["data"]["sample_rate"])
        messages_d3 = [
            {"role": "system", "content": item["d3"]["system"]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_d3}, {"type": "text", "text": item["d3"]["user"]}]}
        ]
        # Qwen2-Audio typical prompt formulation for multimodal msg:
        h3 = model_wrapper.get_hidden_states(
            messages=[
                {"role": "system", "content": item["d3"]["system"]},
                {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" + item["d3"]["user"]}
            ], 
            audio=[audio_d3], 
            target_layer=layer_idx
        )
        d3_vectors.append(h3[0])
        
        # D4: Harmful Audio + Text
        audio_d4 = load_audio(item["d4"]["audio"], target_sr=config["data"]["sample_rate"])
        h4 = model_wrapper.get_hidden_states(
            messages=[
                {"role": "system", "content": item["d4"]["system"]},
                {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" + item["d4"]["user"]}
            ],
            audio=[audio_d4],
            target_layer=layer_idx
        )
        d4_vectors.append(h4[0])
        
    d1_vectors = np.array(d1_vectors)
    d2_vectors = np.array(d2_vectors)
    d3_vectors = np.array(d3_vectors)
    d4_vectors = np.array(d4_vectors)
    
    print("Computing metrics...")
    metrics = compute_metrics(d1_vectors, d2_vectors, d3_vectors, d4_vectors)
    save_json(metrics, outputs_dir / "metrics_summary.json")
    print(f"Metrics saved to {outputs_dir / 'metrics_summary.json'}")
    
    print("Generating PCA visualization...")
    plot_pca_safety_collapse(d1_vectors, d2_vectors, d3_vectors, d4_vectors, outputs_dir / "pca_safety_collapse.png")
    print(f"Plot saved to {outputs_dir / 'pca_safety_collapse.png'}")

if __name__ == "__main__":
    main()
