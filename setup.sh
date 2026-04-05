#!/bin/bash
set -e

echo "🚀 Starting Runpod Environment Setup for Safety Collapse Analysis..."

echo "1️⃣ Installing System Dependencies (ffmpeg)..."
# sudo is used just in case, but Runpod is usually root already
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential

echo "2️⃣ Syncing Python Dependencies via uv..."
# Install all python packages from pyproject.toml
uv sync

echo "✅ Setup Complete! You can now run the pipeline:"
echo "👉 ./scripts/run.py"
