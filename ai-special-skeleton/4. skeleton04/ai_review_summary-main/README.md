# 리뷰 데이터 요약 챗봇 교육용 프로젝트

이 프로젝트는 GPT-2 모델에 LoRA 기법을 적용하여, 리뷰 데이터를 요약하는 모델을 만드는 파인튜닝 파이프라인입니다. CSV 형식의 데이터만 준비하면 모델 학습부터 4-bit 양자화, 그리고 최종 추론까지의 전체 과정을 경험할 수 있도록 구성되어 있습니다.

## 프로젝트 목표

-   **GPT-2 모델에 LoRA를 적용하여 요약 생성 모델을 파인튜닝하는 방법 학습**
-   **데이터 준비부터 학습, 4-bit 양자화, 추론까지 이어지는 전체 파이프라인 경험**
-   **Alpaca 형식의 프롬프트를 사용하여 모델의 성능을 높이는 방법 이해**
-   **ROUGE, BLEU 등 텍스트 생성 모델의 주요 성능 지표 파악**

## 사전 요구사항

-   **Python**: 3.11.8 이상
-   **GPU**: CUDA 지원 GPU (최소 12GB VRAM 권장, fp16 + grad-accum 32 기준)
-   **운영체제**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

## 프로젝트 구조

```text
AI_interview/
├── train.py              # LoRA 학습 스크립트
├── quantization.py       # 4-bit bitsandbytes 양자화 스크립트
├── inference.py          # 추론 데모 스크립트
├── utils/                  # 유틸리티 모듈
│   ├── data.py           # Alpaca 프롬프트 생성 및 토크나이징
│   ├── prompts.py        # 프롬프트 TEMPLATE 문자열 관리
│   └── metrics.py        # ROUGE, BLEU 지표 계산
├── data/                   # 데이터 준비 위치
│   └── Reviews.csv         # 리뷰·요약 CSV (사용자 준비)
└── checkpoints/            # 학습 및 양자화 결과 저장 위치
```

## 시작하기

### 1. 환경 설정

1. `cd 커맨드 사용 시, 본인이 압축을 푼 디렉토리로 이동해야 합니다.`

2. conda 가상환경을 만듭니다.

3. `pip install -r requirements.txt`와 `python setup.py` 스크립트로 기본 환경을 설정합니다.
    - `pip install -r requirements.txt`: 라이브러리 설치
    - `python setup.py`: 폴더 생성 및 환경 셋팅

```bash
cd AI_interview
conda create -n ai_review_summary python=3.11.8 -y
conda activate ai_review_summary
python setup.py
pip install -r requirements.txt
```


### 2. 데이터 준비

`data/Reviews.csv` 경로에 학습용 데이터를 준비합니다. CSV 파일은 `Text`, `summary` 두 개의 컬럼을 반드시 포함해야 합니다.

| Text        | summary      |
| :---------- | :----------- |
| 리뷰 텍스트 | 요약(정답)   |

### 3. 모델 훈련

다음 명령어로 모델 학습을 시작합니다. 학습 중 1 에포크마다 평가가 진행되며, 체크포인트는 `checkpoints/gpt2-lora/` 경로에 저장됩니다.
학습 시 데이터 사이즈가 크기 때문에, 필요에 따라 `data_size`를 더 낮은 사이즈로 조정하는 것이 필요합니다.
```bash
python train.py
```

-   **학습 설정**: Train/Validation 데이터셋은 99:1 비율로 분할됩니다.

### 4. 모델 최적화 (선택사항)

훈련된 LoRA 어댑터를 원본 모델과 병합한 뒤, 4-bit 양자화를 진행하여 모델을 경량화합니다.

```bash
python quantization.py
```

### 5. 모델 추론

최적화된 모델을 사용하여 새로운 질문과 답변에 대한 요약을 생성합니다.

```bash
python inference.py
```
-   **추론 방식**: 스크립트는 Alpaca 템플릿에 질문과 답변을 삽입하여 최대 120 토큰의 응답을 생성하고, `"TL;DR: "` 뒤의 요약 부분만 출력합니다.

## 주요 기능

-   **모델 학습 (`train.py`)**: `gpt2` 모델에 LoRA를 적용하여 파인튜닝을 수행합니다.
-   **4-bit 양자화 (`quantization.py`)**: `bitsandbytes`를 사용하여 훈련된 모델을 4-bit로 경량화합니다.
-   **데이터 처리 (`utils/data.py`)**: CSV 데이터를 Alpaca 형식의 프롬프트로 변환하고 토크나이징을 수행합니다.
-   **성능 평가 (`utils/metrics.py`)**: ROUGE-1/2/L, BLEU 점수를 계산하여 모델의 요약 성능을 평가합니다. (속도 향상을 위해 BERTScore는 제외)

## 커스터마이징

| 항목                 | 파일                 | 변경 위치                          |
| :------------------- | :------------------- | :--------------------------------- |
| 최대 입력 길이       | `utils/data.py`      | `max_len` 변수                     |
| LoRA `r`, `alpha`      | `train.py`           | `LoraConfig` 객체 생성 부분        |
| 평가/저장 주기       | `train.py`           | `eval_strategy`, `save_strategy` 인자 |
| 8-bit 양자화로 변경  | `quantization.py`    | `load_in_4bit=False`, `load_in_8bit=True` |

## 문제 해결

### 양자화 시 경로 오류

`quantization.py` 실행 시 경로 관련 에러가 발생하면, 먼저 아래 명령어로 실제 체크포인트가 저장된 경로를 확인하세요.

```bash
ls checkpoints/gpt2-lora/
```

그리고 `quantization.py` 파일 상단의 `CONFIG` 딕셔너리에서 `lora_dir` 변수의 경로를 실제 체크포인트 경로로 직접 수정해주세요.

```python
CONFIG = dict(
    base_model = "gpt2",
    lora_dir   = "checkpoints/gpt2-lora/checkpoint-62", # 이 부분을 실제 경로로 수정!
    save_dir   = "checkpoints/gpt2-bnb-4bit",
)
```

## 라이선스

-   **코드**: MIT
-   **데이터**: 해당 데이터셋은 CC BY-SA 4.0 라이선스에 따라 사용할 수 있습니다.
