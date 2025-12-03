# train.py
import os
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from utils.data import load_and_prepare_data
from utils.metric import build_compute_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # 기본 설정
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    output_dir = "checkpoints/SmolLM2-360M-Instruct-lora"

    # 1. 데이터 로드 및 전처리
    print("데이터 로딩 및 전처리를 시작합니다...")
    train_dataset, val_dataset, eval_dataset, tokenizer = load_and_prepare_data(
        model_name,
        train_size=1000, # training data의 크기를 조절해주세요.
        val_size=100, # validation data의 크기를 조절해주세요.
        test_size=100 # test data의 크기를 조절해주세요.
    )
    print("데이터 준비 완료!")

    # 2. 기본 모델 로드
    print(f"'{model_name}' 모델을 로드합니다...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id  # pad_token_id 설정

    # 3. LoRA 설정
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 4. 학습 인자 설정
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-5,
        logging_steps=1,
        fp16=torch.cuda.is_available(),  # FP16 학습 활성화 (GPU 사용 시)
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no",
        eval_steps=100,
        dataset_text_field=None,
        max_seq_length=256,
    )

    # 5. SFTTrainer를 사용한 학습
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=build_compute_metrics(tokenizer),
    )
    print("LoRA 파인튜닝을 시작합니다...")
    trainer.train()
    print("학습 완료!")

    # 6. 학습된 LoRA 어댑터 저장
    trainer.save_model(output_dir)
    print(f"학습된 모델이 '{output_dir}'에 저장되었습니다.")
    print("다음 단계:")
    print("1. inference.py를 실행하여 예측을 해보세요")
    print("2. quantization.py를 실행하여 모델을 최적화해보세요")


if __name__ == "__main__":
    main()
