#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "sounddevice",
# ]
# ///

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os

def record_and_augment(duration=5, fs=44100, num_samples=5):
    print(f"🎤 {duration}초 동안 마이크 녹음을 시작합니다...")
    # 마이크 녹음 (1채널 모노, float32 형식)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    # 3, 2, 1 카운트다운 표시 (선택사항)
    for i in range(duration, 0, -1):
        print(f"녹음 중... {i}초 남음")
        sd.sleep(1000)
        
    sd.wait() # 녹음이 완전히 끝날 때까지 대기
    print("✅ 녹음이 완료되었습니다!")

    # 결과물을 저장할 폴더 생성
    output_dir = "outputs/audio_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # 원본 오디오 저장
    original_path = os.path.join(output_dir, "original.wav")
    write(original_path, fs, recording)
    print(f"📁 원본 오디오 저장 완료: {original_path}")

    # 백색 소음(Gaussian Noise) 크기 설정 
    # 값이 너무 크면 목소리가 안 들리므로 0.005 ~ 0.02 정도로 작게 설정
    noise_factor = 0.015

    print(f"\n🎧 내용 전달이 가능하도록 미세한 노이즈가 추가된 {num_samples}개의 변형 샘플을 생성합니다...")
    for i in range(1, num_samples + 1):
        # 원본과 동일한 배열 크기의 랜덤 가우시안 노이즈 생성
        noise = np.random.normal(0, 1, recording.shape)
        
        # 원본에 노이즈 섞기
        noisy_audio = recording + (noise_factor * noise)
        
        # 오디오 신호 값이 [-1.0, 1.0] 범위를 넘어가면 소리가 심각하게 깨지므로(Clipping) 범위 제한
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        # 16-bit PCM WAV 형식으로 저장 가능하도록 변환해서 저장해도 되지만 float32도 지원됨
        output_path = os.path.join(output_dir, f"sample_{i}_noisy.wav")
        write(output_path, fs, noisy_audio)
        print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    # 5초 동안 녹음하고 5개의 샘플 생성
    record_and_augment(duration=5, num_samples=5)
