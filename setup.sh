#!/bin/bash
set -e

echo "🚀 Starting Runpod Environment Setup for Safety Collapse Analysis..."

echo "1️⃣ Installing System Dependencies (ffmpeg, mecab)..."
# sudo is used just in case, but Runpod is usually root already
sudo apt-get update
sudo apt-get install -y ffmpeg mecab libmecab-dev mecab-ipadic-utf8 build-essential

echo "2️⃣ Syncing Python Dependencies via uv..."
# Install all python packages from pyproject.toml
uv sync

echo "3️⃣ Downloading Unidic dictionary for MeloTTS..."
uv run python -m unidic download

echo "✅ Setup Complete! You can now run the pipeline:"
echo "👉 ./scripts/run.py"
