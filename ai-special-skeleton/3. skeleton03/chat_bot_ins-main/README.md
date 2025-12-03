# 보험 상담 챗봇 교육용 프로젝트

이 프로젝트는 보험 도메인 질의응답 데이터셋를 기반으로 SmolLM2-360M-Instruct 모델을 파인튜닝하는 교육용 코드입니다. LoRA(Low-Rank Adaptation)와 4-bit 양자화 기법을 통해, 경량 GPU 환경에서도 실시간 상담 챗봇을 구현하는 전 과정을 경험할 수 있도록 구성되어 있습니다.

## 프로젝트 목표

-   **LoRA를 활용한 효율적인 모델 파인튜닝 방법 학습**
-   **지시문(Instruction) 기반의 프롬프트 엔지니어링 이해**
-   **4-bit 양자화를 통한 모델 경량화 및 최적화 경험**
-   **`trl` 라이브러리의 `SFTTrainer`를 활용한 간편한 모델 미세조정**

## 사전 요구사항

-   **Python**: 3.11.8 고정
-   **GPU**: CUDA 지원 GPU (최소 4GB VRAM 권장)
-   **운영체제**: Windows 11

## 프로젝트 구조

```text
chat_bot_ins/
├── train.py              # 모델 학습 스크립트
├── quantization.py       # 모델 양자화 스크립트
├── inference.py          # 학습된 모델로 추론을 실행하는 스크립트
├── requirements.txt      # 프로젝트에 필요한 패키지 및 버전 명시
├── setup.py              # 환경 설정 및 패키지 설치 스크립트
├── utils/                  # 학습 및 추론에 사용되는 유틸리티 모듈
│   ├── data.py           # 데이터셋 로딩 및 전처리
│   ├── metric.py         # 모델 성능 평가 지표
│   └── prompts.py        # 챗봇 프롬프트 형식 관리
└── data/                 # 학습용 데이터
    └── train.jsonl       # 실제 학습에 사용될 데이터 파일 (train)
    └── test.jsonl        # 실제 학습에 사용될 데이터 파일 (test)
    └── valid.jsonl       # 실제 학습에 사용될 데이터 파일 (valid)
```

## 시작하기

### 1. 환경 설정

1. `cd 커맨드 사용 시, 본인이 압축을 푼 디렉토리로 이동해야 합니다.`

2. conda 가상환경을 만듭니다.

3. `pip install -r requirements.txt`와 `python setup.py` 스크립트로 기본 환경을 설정합니다.
    - `pip install -r requirements.txt`: 라이브러리 설치
    - `python setup.py`: 폴더 생성 및 환경 셋팅

```bash
cd chat_bot_ins-main
conda create -n chatbot_ins python=3.11.8 -y
conda activate chatbot_ins
python setup.py
pip install -r requirements.txt
```

### 2. 모델 훈련

`data/train.csv` 파일이 준비된 상태에서 아래 명령어로 모델 학습을 시작합니다. GPU 환경에서 약 30분에서 2시간이 소요될 수 있습니다.

```bash
python train.py
```

학습이 완료되면, `checkpoints/SmolLM2-360M-Instruct-lora` 경로에 LoRA 어댑터가 저장됩니다.

### 3. 모델 최적화 (선택사항)

훈련된 LoRA 어댑터를 원본 모델과 병합한 뒤, 4-bit 양자화를 진행하여 모델을 경량화합니다.

```bash
python quantization.py
```

### 4. 모델 추론

최적화된 모델을 사용하여 샘플 질문에 대한 답변을 생성합니다.

```bash
python inference.py
```

## 주요 기능 및 코드 설명

### 1. 데이터 처리 (`utils/`)

| 파일         | 역할                                                                        |
| :----------- | :-------------------------------------------------------------------------- |
| **`data.py`** | 	로컬 JSONL(train/valid/test) 파일을 로드하여 SmolLM2 채팅 템플릿 형식으로 변환하고, 토크나이즈·라벨 마스킹까지 완료된 datasets.Dataset 객체를 생성합니다. Train/Validation/Eval 데이터셋을 파일 단위로 불러옵니다(랜덤 분할 없음). |
| **`metric.py`** | HuggingFace `evaluate` 라이브러리를 사용하여 BLEU 점수와 Perplexity를 계산하는 함수를 포함합니다. |


### 2. 스크립트별 핵심 로직

| 파일                | 주요 포인트                                                                          |
| :------------------ | :----------------------------------------------------------------------------------- |
| **`train.py`** | 기본 모델 `SmolLM2`에 LoRA(target `c_attn`, r=16, α=32)를 적용합니다. 20 에포크, 배치 2×16, FP16 훈련을 지원합니다. |
| **`quantization.py`** | 훈련된 LoRA 어댑터를 기본 모델과 병합한 뒤, `BitsAndBytesConfig`를 이용해 NF4 방식으로 4-bit 양자화를 수행하고 저장합니다. |
| **`inference.py`** | 프롬프트 템플릿에 질문을 채운 뒤, `model.generate()`를 호출하여 답변을 생성합니다. (temp=0.7, top-p=0.9, max_length=150) |

## 문제 해결

### 양자화 시 경로 오류

`quantization.py` 실행 시 경로 관련 에러가 발생하면, 먼저 아래 명령어로 실제 체크포인트가 저장된 경로를 확인하세요.

```bash
dir model-checkpoints/SmolLM2-360M-Instruct-lora/
```

그리고 `quantization.py` 파일의 `main` 함수 내에서 `adapter_path` 변수의 경로를 실제 체크포인트 경로로 직접 수정해주세요.

```python
def main():
    # 경로 설정
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "checkpoints/SmolLM2-360M-Instruct-lora/checkpoint-100" # 이 부분을 실제 경로로 수정!
    ...
```

## 라이선스 및 고지 사항

-   **코드**: 별도로 명시되지 않는 한 MIT 라이선스를 따릅니다.
-   **데이터**: 공개된 보험 약관을 재가공한 2차 저작물입니다. **비영리 연구 및 교육 목적으로만** 사용하시기를 권장합니다.
