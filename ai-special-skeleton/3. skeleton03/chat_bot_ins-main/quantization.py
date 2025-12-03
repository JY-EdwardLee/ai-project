# quantization.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os


def main():
    # 경로 설정
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "checkpoints/SmolLM2-360M-Instruct-lora/checkpoint-32" #해당 부분을 /checkpoints 폴더를 확인하여 변경
    quantized_output_dir = "checkpoints/SmolLM2-360M-Instruct-4bit"

    # 1. 기본 모델 및 토크나이저 로드
    print(f"기본 모델 '{base_model_name}'을 로드합니다.")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # 메모리 효율을 위해 float16으로 로드
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. LoRA 어댑터 로드 및 병합
    print(f"LoRA 어댑터를 '{adapter_path}'에서 로드하여 병합합니다.")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print("모델 병합 완료.")

    # 3. 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 4. 병합된 모델을 4-bit로 양자화하여 다시 로드
    print("병합된 모델을 4-bit로 양자화하여 로드합니다...")
    merged_model_path = "checkpoints/gpt2-insurance-merged-temp"
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

    quantized_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, quantization_config=bnb_config, device_map="auto"
    )
    print("4-bit 양자화 완료.")

    # 5. 최종 양자화 모델 저장
    os.makedirs(quantized_output_dir, exist_ok=True)
    quantized_model.save_pretrained(quantized_output_dir)
    tokenizer.save_pretrained(quantized_output_dir)
    print(f"4-bit 양자화 모델이 '{quantized_output_dir}'에 저장되었습니다.")        
    print("="*50)
    print("양자화 완료!!")
    print("inference.py main함수에서 옵션(경로)을 수정하여 양자화된 모델로 추론할 수 있습니다!!")


if __name__ == "__main__":
    main()
