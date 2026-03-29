# Multimodal Jailbreak Pipeline 🛡️

본 프로젝트는 VRAM 제약(16GB~24GB)을 고려하여 구축된 **텍스트 vs 오디오 멀티모달 탈옥(Jailbreak) 비교 평가 벤치마크 파이프라인**입니다.

## 🛠️ 실행 환경 및 설치 가이드 (Git Clone 직후)

**1. 패키지 동기화 (`uv` 기반)**
터미널에서 레포지토리에 들어간 뒤, 아래 명령어로 환경과 패키지를 한 번에 동기화합니다.
```bash
uv sync
```

*※ 주의사항: 4-bit 양자화를 위한 `bitsandbytes`는 Linux + NVIDIA GPU 환경에 가장 적합합니다. macOS에서도 코드가 실행되도록 예외 처리는 해두었으나, 빠른 인퍼런스와 양자화 효과를 위해 GPU 서버 실행을 권장합니다.*

**2. 원클릭 파이프라인 실행**
스크립트에 실행 권한을 주고 바로 실행하면 1단계부터 5단계까지 VRAM을 자동으로 비우면서 연속으로 진행됩니다.
```bash
chmod +x scripts/run_pipeline.py
./scripts/run_pipeline.py
```

## 🔄 파이프라인 순서 로직
1. **다운로드**: HuggingFace의 `advbench` (또는 대안 안전 데이터셋)에서 100개의 악성 프롬프트를 무작위 다운.
2. **음성 합성**: `edge-tts` (무료 라이브러리)를 거쳐 100개의 오디오 `.wav` 파일 생성.
3. **타겟 공격 (Target Test)**: `Qwen2-Audio-7B-Instruct`을 4-bit로 로드하여 ①텍스트 입력과 ②오디오 입력을 순차적으로 공격한 뒤 각각의 답변 수집. **가장 무거운 과정이며, 즉각 VRAM을 해제(Empty Cache)합니다.**
4. **LLM 심판 (Score Eval)**: 빠르고 똑똑한 `Qwen2.5-1.5B-Instruct`를 로드시켜 각 응답의 심각성(Jailbreak 정도)을 1~5점 사이의 점수로 스코어링.
5. **결과 시각화결과**: 모달리티(Text vs Audio)에 따른 Jailbreak 취약점 차이 요약 리포트 표출.
