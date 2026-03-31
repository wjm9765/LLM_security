# AI_security: Multimodal LLM Safety Collapse Analysis

이 프로젝트는 Qwen2-Audio와 같은 멀티모달 LLM(Multimodal LLM)에서 발생하는 "Safety Collapse" 현상을 활성화 벡터(Activation Vector) 분석을 통해 입증하기 위한 파이프라인입니다.

## 📂 프로젝트 구조
- `config.yaml`: 대상 모델 경로, 레이어 인덱스, 데이터 생성 등의 메인 환경 설정
- `pyproject.toml`: 프로젝트 의존성 설정 (`uv`, `pip` 등으로 설치되는 패키지 명세)
- `src/`: 핵심 로직 모음
  - `data_utils.py`: 데이터 다운로드 및 설정 파싱 유틸리티
  - `model_utils.py`: Target 모델의 Forward pass 및 Hooking 파이프라인
  - `metrics.py`: L2 및 코사인 유사도 연산 메트릭 구현 
  - `visualize.py`: PCA 계산 및 플롯 시각화 담당
- `scripts/`: 실행 가능한 파이프라인 스크립트 모음
  - `01_prepare_data.py`: AdvBench 데이터셋 로드, Helper 모델을 이용한 안전한 문장 생성, 노이즈 오디오 합성 등 전체 데이터 구축 진행
  - `02_run_experiment.py`: 추출된 데이터를 Target 모델에 인퍼런스(Forward)하여 은닉 상태(hidden states) 추출 후, 지표 계산 및 시각화 수행
  - `run.py`: 전체 파이프라인 통합 자동 실행 파일
- `inputs/`: 다운로드 된 데이터셋 및 생성 완료된 오디오/텍스트 파일이 저장되는 공간
- `outputs/`: PCA 프로젝션 플롯, 도출된 메트릭스가 담긴 JSON이 추출되는 공간

## 🚀 빠른 시작

[uv](https://github.com/astral-sh/uv) 패키지 관리자를 사용하여 매우 빠른 속도로 가상환경 구성 및 패키지 라이브러리를 동기화(`uv sync`)한 뒤, 단일 스크립트(`/scripts/run.py`)로 전체 파이프라인을 실행할 수 있습니다. 이미 환경이 설정되어 있다면 추가 설치 과정 없이 바로 실험이 수행됩니다.

```bash
# 1. 의존성 라이브러리 및 가상환경 동기화 (최초 1회 혹은 패키지 변경 시)
uv sync

# 2. 통합 파이프라인 스크립트 실행
./scripts/run.py
```

**`run.py` 작동 순서**:
1. 최상단 shebang(`#!/usr/bin/env -S uv run`)을 통해 `uv` 환경 위에서 즉시 실행
2. `scripts/01_prepare_data.py`를 호출하여 4가지 데이터셋 포맷 생성
3. `scripts/02_run_experiment.py`를 호출하여 모델 내 Hidden States 추출, PCA 시각화 플롯 ও 계산 결과물을 `outputs/`에 저장

## 📊 Outputs 설명
실험이 종료되면 `outputs/` 폴더에 다음 파일이 자동 생성됩니다.

1. **`pca_safety_collapse.png`**: D1~D4 4개의 데이터 분포를 2D 공간에 매핑하여 Multimodal Modality일 때의 Safety 현황을 가시화한 산점도
2. **`metrics_summary.json`**: 모델 활성화 층(예: layer 15)에서 각 데이터 군집 간 거리를 L2 Distance 및 Cosine Similarity로 산출한 정량적 결괏값

## ⚙️ 요구 사항
- `uv` (빠른 패키지 설치용 툴)
  `curl -LsSf https://astral.sh/uv/install.sh | sh` 혹은 `pip install uv`로 설치 가능
- CUDA 지원 GPU (설정의 \`device: "cuda"\` 세팅을 위해 권장)