#!/usr/bin/env python3
"""
모델 양자화 스크립트

이 스크립트는 학습된 한국어 악성 댓글 분류 모델을 4-bit 양자화하여 
메모리 사용량을 줄이고 추론 속도를 향상시킵니다.

LoRA 가중치를 4-bit bitsandbytes 모델로 변환 

사용법:
    python quantization.py
"""
from pathlib import Path
import tempfile
import torch
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)
from peft import PeftModel

CONFIG = {
    "base_model": "skt/kobert-base-v1",
    "lora_dir":   "checkpoints/kobert-lora/checkpoint-15", #해당 부분을 /checkpoints 폴더를 확인하여 변경
    "save_dir":   "checkpoints/kobert-bnb-4bit",
}

def quantize(cfg: dict = CONFIG):
    cfg_hf = AutoConfig.from_pretrained(cfg["base_model"],
                                        num_labels=2,
                                        problem_type="single_label_classification")

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], config=cfg_hf, torch_dtype=torch.float16)

    model = PeftModel.from_pretrained(base, cfg["lora_dir"], torch_dtype=torch.float16).merge_and_unload()


    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        q_model = AutoModelForSequenceClassification.from_pretrained(
            tmp, load_in_4bit=True, device_map="auto")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(cfg["save_dir"])
    q_model.save_pretrained(cfg["save_dir"])
    print(f"4-bit model saved to {cfg['save_dir']}")
    print("="*50)
    print("양자화 완료!!")
    print("inference.py CONFIG에서 옵션(경로)을 수정하여 양자화된 모델로 추론할 수 있습니다!!")


if __name__ == "__main__":
    quantize()
