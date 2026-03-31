# Data utilities
import yaml
import urllib.request
import pandas as pd
import json
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_advbench(url: str, num_samples: int, save_path: str) -> pd.DataFrame:
    urllib.request.urlretrieve(url, save_path)
    df = pd.read_csv(save_path)
    return df.head(num_samples)

def save_json(data, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)
