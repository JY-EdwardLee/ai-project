# inference.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_inference(model, tokenizer, instruction):
    """
    주어진 instruction과 input으로 추론을 실행합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": instruction,
            },
        ],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    print("추론 중...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response.strip()


def main():
    # --- 이 경로를 LoRA 체크포인트 또는 4-bit 모델로 변경할 수 있습니다 ---
    # 옵션 1: LoRA 
    model_path = "checkpoints/SmolLM2-360M-Instruct-lora" 
    # # 옵션 2 병합 및 양자화된 모델 
    # model_path = "checkpoints/SmolLM2-360M-Instruct-4bit"  
    
    print(f"'{model_path}'에서 모델을 로드합니다...")

    is_lora_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora_adapter:
        # --- LoRA Loading Logic ---
        model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = PeftModel.from_pretrained(model, model_path)

    else:
        # --- Merged & Quantized Loading Logic ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Universal pad token setting
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("모델 로드 완료!")

    # 2. 추론할 샘플 데이터
    sample_instruction = "Does Medicare Cover Co-Pays?"

    # 3. 추론 실행
    generated_answer = run_inference(model, tokenizer, sample_instruction)

    # 4. 결과 출력
    print("\n" + "=" * 50)
    print(f"질문: {sample_instruction}")
    print(f"생성된 답변: {generated_answer}")
    print("=" * 50)


if __name__ == "__main__":
    main()
