# train.py
"""
GPT-2  LoRA  (decoder-only + DataCollatorForSeq2Seq)
data/Dataset.csv  →  checkpoints/gpt2-lora
"""

import os
from pathlib import Path
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from utils.data import tokenize_summarize
from utils.metrics import build_compute_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CONFIG = dict(
    csv_path=Path("data/Reviews.csv"),
    base_model="gpt2",
    save_dir=Path("checkpoints/gpt2-lora"),
    max_len=256,
    gradient_accumulation_steps=32,
    batch=1,
    epochs=1,
    lr=1e-5,
    val_split=0.01,
    data_size = 2000 # 학습에 활용할 데이터의 크기를 정할 수 있습니다.
)


def main(cfg=CONFIG):
    print("CSV 로드")
    df = (
        pd.read_csv(cfg["csv_path"], index_col=0)
        .dropna(subset=["Text", "Summary"])
        .loc[:cfg["data_size"], ["Text", "Summary"]]
        .rename(columns={"Text": "review", "Summary": "summary"})
    )

    ds_raw = Dataset.from_pandas(df).train_test_split(
        test_size=cfg["val_split"], seed=42
    )

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # for causal LM it's fine

    def _tok(batch):  # ← 래퍼
        return tokenize_summarize(batch, tokenizer=tok, max_len=cfg["max_len"])

    print("프롬프팅+토크나이즈")
    ds_tok = ds_raw.map(
        _tok,  # ← tokenizer 주입
        batched=True,
        batch_size=1,
        num_proc=1,  # temp busy 오류 피하기
        remove_columns=ds_raw["train"].column_names,
        load_from_cache_file=False,  # 이전 캐시 무시
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=None,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=tok.pad_token_id,
    )

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
    base.config.pad_token_id = tok.pad_token_id
    lora = get_peft_model(
        base,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        ),
    )
    lora.print_trainable_parameters()

    compute_metrics = build_compute_metrics(tok)

    args = TrainingArguments(
        output_dir=str(cfg["save_dir"]),
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=1,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        bf16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    )

    Trainer(
        model=lora,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    ).train()

    lora.save_pretrained(cfg["save_dir"])
    tok.save_pretrained(cfg["save_dir"])
    print("LoRA 어댑터 저장: ", cfg["save_dir"])
    print("다음 단계:")
    print("1. inference.py를 실행하여 예측을 해보세요")
    print("2. quantization.py를 실행하여 모델을 최적화해보세요")


if __name__ == "__main__":
    main()
