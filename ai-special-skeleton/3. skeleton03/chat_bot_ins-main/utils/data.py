import os
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

RESPONSE_START_TEMPLATE = "'<|im_start|>assistant\n'"
RESPONSE_START_TEMPLATE_IDS = [1, 520, 9531, 198]

def preprocess(prompt_messages, label_messages, tokenizer):

    label_message = label_messages[0]["content"]
    label_input_ids = tokenizer.encode(
        label_message, add_special_tokens=False, return_tensors="pt"
    ).squeeze(0)

    prompt_input_ids = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=False, tokenize=True, return_tensors="pt"
    ).squeeze(0)

    prompt: str = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )

    input_length = (
        len(prompt_input_ids)
        + len(RESPONSE_START_TEMPLATE_IDS)
        + len(label_input_ids)
        + 1
    )

    input_ids = torch.cat(
        [
            prompt_input_ids,
            torch.tensor(RESPONSE_START_TEMPLATE_IDS),
            label_input_ids,
            torch.tensor([tokenizer.eos_token_id]),
        ],
        dim=0,
    )

    attention_mask = torch.ones(input_length, dtype=torch.int64)
    labels = torch.cat(
        [
            torch.tensor(
                [-100] * (input_length - len(label_input_ids) - 1)
            ),  # prompt + label_start_template
            label_input_ids,  # label
            torch.tensor([tokenizer.eos_token_id]),  # [EOS]
        ],
        dim=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": input_length - len(label_input_ids) - 1,
        "prompt": prompt,
        "label_text": label_message,
    }

def load_and_prepare_data(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
    data_dir="./data/",
    train_size=None,  # training data의 크기를 조절해주세요.
    val_size=None,    # validation data의 크기를 조절해주세요.
    test_size=None    # test data의 크기를 조절해주세요.
):
    """
    로컬 JSONL 파일을 로드하고, 학습/검증/평가용으로 분리한 뒤,
    지정된 크기로 데이터를 조절하고 채팅 형식으로 변환합니다.
    """
    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 로컬 JSONL 파일에서 데이터셋 불러오기
    data_files = {
        "train": os.path.join(data_dir, "train.jsonl"),
        "validation": os.path.join(data_dir, "valid.jsonl"),
        "test": os.path.join(data_dir, "test.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    # 3. 데이터셋 크기 조절
    if train_size:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
    if val_size:
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_size))
    if test_size:
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(test_size))

    def to_chat(example, tokenizer):
        prompt_messages = [
            {"role": "user", "content": example["input"]},
        ]
        label_messages = [
            {"role": "assistant", "content": example["output"]},
        ]
        model_inputs = preprocess(prompt_messages, label_messages, tokenizer)
        return model_inputs

    # 4. 데이터셋을 학습용과 평가용으로 분리
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    eval_dataset = dataset["test"]

    # 5. 데이터셋을 채팅 형식으로 변환
    train_dataset = train_dataset.map(lambda example: to_chat(example, tokenizer))
    val_dataset = val_dataset.map(lambda example: to_chat(example, tokenizer))
    eval_dataset = eval_dataset.map(lambda example: to_chat(example, tokenizer))

    print(
        f"데이터 분리 완료: 학습용 {len(train_dataset)}개, 검증용 {len(val_dataset)}개, 평가용 {len(eval_dataset)}개"
    )

    # 6. 분리된 데이터셋과 토크나이저 반환
    return train_dataset, val_dataset, eval_dataset, tokenizer

# --- Example Usage ---
# To load a smaller portion of the data (e.g., 1000 for train, 100 for val/test)
# train_ds, val_ds, test_ds, tokenizer = load_and_prepare_data(
#     train_size=1000,
#     val_size=100,
#     test_size=100
# )

# To use the full dataset as before, simply don't pass the size arguments
# train_ds_full, val_ds_full, test_ds_full, tokenizer_full = load_and_prepare_data()
