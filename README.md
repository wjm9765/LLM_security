# AI_security: Multimodal LLM Safety Collapse Analysis

이 프로젝트는 Qwen2-Audio와 같은 멀티모달 LLM에서 오디오 모달리티가 텍스트 안전망을 어떻게 우회(Bypass)하는지, 즉 **"Safety Collapse" 현상**을 활성화 벡터(Activation Vector) 분석을 통해 입증하기 위한 연구용 파이프라인입니다.

특히 최신 논문 레벨(ICLR 사전 연구)의 분석 체계를 도입하여, "모달리티 갭"과 "유해성 갭"을 엄밀하게 분리해냅니다.

## 🌟 핵심 분석 방법론

1. **직교성 검증 (Orthogonality Test)**
   - 텍스트 유해성 판단 축($V_{text}$)과 오디오 유해성 판단 축($V_{audio}$) 간의 코사인 유사도를 계산합니다.
   - 두 축이 직교(Orthogonal)함을 증명하여, 오디오가 기존 텍스트 안전망을 거치지 않는 독립적인 차원으로 처리됨을 밝힙니다.
   
2. **쌍방향 잔차 SVD (Paired Residual SVD)**
   - 데이터 쌍의 차이 벡터($\Delta = D2 - D1$)에 대해 SVD를 수행합니다.
   - 문장 길이나 포맷 등 노이즈를 제거한 **"순수 텍스트 유해성 축(Pure Text Safety Direction)"**을 도출하고, 오디오 데이터가 이 축을 완벽히 비껴감을 시각적으로 증명합니다.

3. **이중 토큰 추적 (Double Token Tracking)**
   - $t_{inst}$: 지시문의 마지막 토큰 (의도 파악 벡터)
   - $t_{post-inst}$: 생성이 시작되기 직전의 토큰 (거절/수락 발화 벡터)
   - 두 토큰에서의 은닉 상태(Hidden states)를 추출하여 거절 매커니즘의 발동 양상을 추적합니다.

## 📂 프로젝트 구조

- `config.yaml`: 대상 모델, 데이터 수, 레이어 인덱스 설정
- `pyproject.toml`: `uv` 패키지 매니저용 Python 의존성 명세 (gTTS, edge-tts 통합, 불필요 패키지 정리 완료)
- `setup.sh`: Runpod(Ubuntu) 환경용 사전 패키지(ffmpeg 등) 및 의존성 원클릭 설치 스크립트
- `src/`
  - `data_utils.py`: 데이터 다운로드 (AdvBench) 및 Config 파싱
  - `model_utils.py`: Target 모델 (Qwen)의 Forward Pass 및 은닉 상태 추출 (이중 토큰)
  - `metrics.py`: 직교성 테스트, 쌍방향 잔차 SVD, 거리 및 코사인 유사도 연산
  - `visualize.py`: PCA, 다방향 투영 및 SVD 결과 시각화 모듈
- `scripts/`
  - `01_prepare_data.py`: D1~D6 데이터셋 자동 포맷팅 및 생성 (gTTS 50%, edge-tts 50% 분할 배분)
  - `02_run_experiment.py`: 타겟 모델 구동, 지표 계산(Metrics) 및 시각화(Plot) 호출
  - `run.py`: 1, 2단계를 연속으로 실행하는 통합 실행 스크립트

## 🚀 설치 및 실행 (Runpod / Linux 기준)

GPU가 탑재된 인스턴스에 레포지토리를 Clone한 뒤, 단 2줄의 명령어로 설치 및 실행이 가능합니다.

```bash
# 1. 시스템 패키지 설치 및 Python 의존성(uv) 동기화
chmod +x setup.sh
./setup.sh

# 2. 통합 파이프라인 실행
./scripts/run.py
```

## 📊 생성되는 데이터셋 (D1 ~ D6)

- **D1**: 순수 안전 텍스트 (Helper LLM 자동 생성)
- **D2**: 유해 텍스트 (AdvBench)
- **D3**: 가우시안 노이즈(오디오) + 유해 텍스트
- **D4**: 유해 내용을 읽어주는 오디오 (gTTS & edge-tts) + 텍스트
- **D5**: 가우시안 노이즈(오디오) + 안전 텍스트
- **D6**: 안전 내용을 읽어주는 오디오 (gTTS & edge-tts) + 텍스트

## 📈 Outputs (결과물)

파이프라인 실행이 완료되면 `outputs/` 폴더에 다음 3종의 플롯과 2개의 데이터 파일이 생성됩니다:

1. **`model_responses.json`**: 모델이 실제로 생성한 텍스트 답변(LLM-as-a-Judge나 휴먼 정성 평가용)
2. **`metrics_summary.json`**: 거리 연산 및 **직교성 점수(Orthogonality_CosSim_Vtext_Vaudio)** 기록 전체 요약
3. **`pca_safety_collapse.png`**: D1~D6의 단순 은닉 공간 2D PCA 투영도 (전체 분산 기반)
4. **`harmful_vs_refusal_directions.png`**: Harmfulness 축 (X) vs Refusal 발동 축 (Y) 모델 판별 기전 2D 산점도
5. **`paired_residual_svd.png`**: 쌍방향 잔차 SVD를 통한 **순수 텍스트 유해 축(PC1)** 공간에서의 모달리티 우회(Safety Bypass) 완벽 증명 시각화
