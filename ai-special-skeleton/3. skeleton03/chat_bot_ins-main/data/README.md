# InsuranceQA v2 (JSONL Local Version)
보험 도메인 질의응답 데이터셋(InsuranceQA v2)을 로컬 환경에서 바로 사용할 수 있도록 train.jsonl, valid.jsonl, test.jsonl 형태로 제공합니다. 각 샘플은 질문(input)과 전문가 답변(output)으로 구성되어 있습니다. 원 데이터셋은 보험 관련 실질적인 Q&A를 포함하며, 고객 질문과 전문가의 상세한 답변이 쌍을 이루고 있습니다.

### 파일 구성
파일명	샘플 수	설명
- train.jsonl	21,325	모델 학습용 데이터
- valid.jsonl	3,354	학습 중 검증용 데이터
- test.jsonl	3,308	최종 평가용 데이터

### 데이터 포맷
각 파일은 JSON Lines(.jsonl) 형식입니다. 한 줄에 하나의 JSON 객체가 기록되어 있으며, 주요 필드는 다음과 같습니다.

```
{
  "input": "What Does Medicare IME Stand For? ",
  "output": "According to the Centers for Medicare and Medicaid Services website , cms.gov , IME stands for Indirect Medical Education and is in regards to payment calculation adjustments for a Medicare discharge of higher cost patients receiving care from teaching hospitals relative to non-teaching hospitals . I would recommend contacting CMS to get more information about IME "
}
```
- input: 보험 관련 질문 텍스트 (영문)
- output: 해당 질문에 대한 전문가 답변 (영문)


### 출처 & 라이선스
- 원본 데이터셋: Hugging Face – deccan-ai/insuranceQA-v2
- 원본 논문: Feng et al., Applying Deep Learning to Answer Selection: A Study and An Open Task, ASRU 2015.
- 본 저장소는 데이터 구조를 JSONL로 변환한 버전이며, 내용은 원본과 동일합니다.
- 사용 전 반드시 원본 데이터셋의 라이선스와 이용 약관을 확인하세요.

### 주의 사항
일부 응답은 법률, 세무, 보험 자문과 유사한 내용을 포함합니다. 실제 서비스 적용 시 반드시 전문가 검토를 거쳐야 합니다.
텍스트 길이가 길어 학습 시 max_length 또는 truncation 설정이 필요할 수 있습니다.
