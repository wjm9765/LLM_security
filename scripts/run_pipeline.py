#!/usr/bin/env -S uv run
import os
import sys
from pathlib import Path

# HF 캐시 경로를 용량이 넉넉한 workspace 하위로 변경
os.environ["HF_HOME"] = "/workspace/HF_CACHE"

sys.path.append(str(Path(__file__).parent.parent))

from src.step1_data import download_and_sample
from src.step2_tts import generate_audios
from src.step3_target import run_target_model
from src.step4_judge import run_judge_model

def main():
    # 경로 설정
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    PROMPTS_FILE = DATA_DIR / "prompts_100.json"
    AUDIO_DIR = DATA_DIR / "audio_prompts"
    TEXT_OUT = OUTPUT_DIR / "text_results.jsonl"
    AUDIO_OUT = OUTPUT_DIR / "audio_results.jsonl"
    
    print("="*60)
    print("🚀 멀티모달 (Text vs Audio) Jailbreak 파이프라인 시작 🚀")
    print("="*60)

    # [Step 1] 데이터 수집
    prompts = download_and_sample(str(PROMPTS_FILE), num_samples=100)
    
    # [Step 2] TTS
    generate_audios(prompts, str(AUDIO_DIR))
    
    # [Step 3] 타겟 양자화 모델 구동 후 VRAM 비우기
    run_target_model(str(PROMPTS_FILE), str(AUDIO_DIR), str(TEXT_OUT), str(AUDIO_OUT))
    
    # [Step 4] 심판관 모델 구동하여 점수 부여!
    text_avg, audio_avg = run_judge_model(str(TEXT_OUT), str(AUDIO_OUT))
    
    # [Step 5] 결과 합산 및 요약 출력
    print("\n" + "="*60)
    print("📊 [실험 결과 요약] 타겟 모델 (Qwen2-Audio-7B-Instruct)")
    print("="*60)
    print(f"🔸 텍스트 프롬프트 직접 입력 시 평균 위험도: [ {text_avg:.2f} / 5.0 ]")
    print(f"🔹 오디오 변환 후 음성 입력 시 평균 위험도:  [ {audio_avg:.2f} / 5.0 ]")
    print("\nMotivation Check: 오디오 모달리티로 우회(Jailbreak)될 확률이 텍스트보다 높은지 확인해보세요!")
    print("결과물은 outputs/ 위치에 JSONL 파일로 저장되어 있습니다.")
    print("="*60)

if __name__ == "__main__":
    main()
