# utils/data.py
from utils.prompts import TEMPLATE


def _make_prompt(
    text: str,
    instruction: str = "Summarize the following text in 1–2 sentences.\n\n",
    response_start_token: str = "\n\nTL;DR: ",
) -> str:
    return TEMPLATE.format(
        instruction=instruction,
        input=text.strip(),
        response_start_token=response_start_token,
    )


def tokenize_summarize(
    batch,
    tokenizer,
    max_len=1024,
    ignore_index=-100,
    padding=False,
):
    ins, masks, labs = [], [], []
    for review, summary in zip(batch["review"], batch["summary"]):
        input_prompt = _make_prompt(text=review)
        tokenized_input_prompt = tokenizer.encode(input_prompt)
        input_prompt_length = len(tokenized_input_prompt)

        tokenized_label = tokenizer.encode(summary) + [tokenizer.eos_token_id]
        response_length = len(tokenized_label)

        input_length = input_prompt_length + response_length

        overflow = input_length - max_len
        if overflow > 0:
            tokenized_input_prompt = tokenized_input_prompt[:-overflow]

        input_ids = tokenized_input_prompt + tokenized_label
        labels = [ignore_index] * len(tokenized_input_prompt) + tokenized_label[:]
        attention_masks = (
            [1] * len(input_ids)
            if not padding
            else [1] * len(input_ids) + [0] * (max_len - len(input_ids))
        )

        if padding:
            if len(input_ids) < max_len:  # padding
                input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
            if len(labels) < max_len:  # padding
                labels += [ignore_index] * (max_len - len(labels))

        if len(input_ids) > max_len:
            import pdb

            pdb.set_trace()

        ins.append(input_ids)
        masks.append(attention_masks)
        labs.append(labels)

    return {"input_ids": ins, "attention_masks": masks, "labels": labs}
