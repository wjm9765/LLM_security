import sys
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm

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
    
    d1_vectors, d2_vectors, d3_vectors, d4_vectors, d5_vectors, d6_vectors = [], [], [], [], [], []
    model_responses = []
    
    print("Running forward passes & generating responses for all Datasets...")
    for item in tqdm(datasets, desc="Evaluating Data (Activation & Response)"):
        # Helper to map inputs
        def proc(messages, audio_path=None):
            a_data = [load_audio(audio_path, target_sr=16000)] if audio_path else None
            h, r = model_wrapper.get_hidden_states_and_response(
                messages=messages, 
                audio=a_data, 
                target_layer=layer_idx,
                max_new_tokens=30 # Response 텍스트 생성을 OOM 방지를 위해 짧게 설정
            )
            return h, r

        # D1: Safe
        h1, r1 = proc([{"role": "system", "content": item["d1"]["system"]}, {"role": "user", "content": item["d1"]["user"]}])
        d1_vectors.append(h1)
        
        # D2: Harmful Text
        h2, r2 = proc([{"role": "system", "content": item["d2"]["system"]}, {"role": "user", "content": item["d2"]["user"]}])
        d2_vectors.append(h2)
        
        # D3: Noise + Harmful
        h3, r3 = proc([{"role": "system", "content": item["d3"]["system"]}, {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" + item["d3"]["user"]}], item["d3"]["audio"])
        d3_vectors.append(h3)
        
        # D4: TTS Harmful Audio + Text
        h4, r4 = proc([{"role": "system", "content": item["d4"]["system"]}, {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" + item["d4"]["user"]}], item["d4"]["audio"])
        d4_vectors.append(h4)
        
        # D5: Noise + Safe Text
        h5, r5 = proc([{"role": "system", "content": item["d5"]["system"]}, {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" + item["d5"]["user"]}], item["d5"]["audio"])
        d5_vectors.append(h5)
        
        # D6: TTS Safe Audio + Text
        h6, r6 = proc([{"role": "system", "content": item["d6"]["system"]}, {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" + item["d6"]["user"]}], item["d6"]["audio"])
        d6_vectors.append(h6)
        
        model_responses.append({
            "id": item["id"],
            "d1_safe_text": {"input": item["d1"]["user"], "output": r1},
            "d2_harmful_text": {"input": item["d2"]["user"], "output": r2},
            "d3_noise_harmful": {"input": item["d3"]["user"], "output": r3},
            "d4_audio_harmful": {"input": item["d4"]["user"], "output": r4},
            "d5_noise_safe": {"input": item["d5"]["user"], "output": r5},
            "d6_audio_safe": {"input": item["d6"]["user"], "output": r6},
        })
        
    d1_vectors = np.array(d1_vectors)
    d2_vectors = np.array(d2_vectors)
    d3_vectors = np.array(d3_vectors)
    d4_vectors = np.array(d4_vectors)
    d5_vectors = np.array(d5_vectors)
    d6_vectors = np.array(d6_vectors)
    
    print("Saving text generation responses...")
    save_json(model_responses, outputs_dir / "model_responses.json")
    
    print("Computing metrics...")
    metrics = compute_metrics(d1_vectors, d2_vectors, d3_vectors, d4_vectors, d5_vectors, d6_vectors)
    save_json(metrics, outputs_dir / "metrics_summary.json")
    print(f"Metrics saved to {outputs_dir / 'metrics_summary.json'}")
    
    print("Generating PCA visualization...")
    plot_pca_safety_collapse(d1_vectors, d2_vectors, d3_vectors, d4_vectors, d5_vectors, d6_vectors, outputs_dir / "pca_safety_collapse.png")
    print(f"Plot saved to {outputs_dir / 'pca_safety_collapse.png'}")

if __name__ == "__main__":
    main()
