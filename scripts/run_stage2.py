#!/usr/bin/env -S uv run
import subprocess
import sys
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    scripts_dir = base_dir / "scripts"
    
    print("🚀 Starting Safety Collapse Analysis Pipeline (Stage 2 Only)...")
    
    # 2단계: 실험 및 시각화 실행
    print("\n🧪 Running Step 2: Experiment & Visualization...")
    subprocess.run([sys.executable, str(scripts_dir / "02_run_experiment.py")], check=True)
    
    print("\n✅ Pipeline finished successfully!")
    print("📂 Check the 'outputs/' directory for results (metrics_summary.json, pca_safety_collapse.png).")

if __name__ == "__main__":
    main()