# utils/metrics.py ── BLEU + Perplexity 전용
from evaluate import load
from transformers import PreTrainedTokenizerBase
import numpy as np
import torch
import torch.nn.functional as F

bleu = load("bleu")


def _post(txts):
    return [" ".join(t.strip().split()) for t in txts]


def _nd_to_tokens(arr, tok: PreTrainedTokenizerBase):
    if isinstance(arr, np.ndarray) and arr.ndim == 3:        # [B, L, V] → id
        arr = arr.argmax(-1)
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if isinstance(arr[0], list):
        return tok.batch_decode(arr, skip_special_tokens=True)
    return arr


def _perplexity(logits, labels):
    # ── 1) numpy → torch 변환 ───────────────────────────────────────────
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    logits = logits.to(torch.float32)          # float64로 넘어오는 경우 대비

    # ── 2) shift-one-token ──────────────────────────────────────────────
    shift_logits = logits[..., :-1, :].contiguous()   # (B, L-1, V)
    shift_labels = labels[..., 1:].contiguous()       # (B, L-1)

    # ── 3) cross-entropy + exp ─────────────────────────────────────────
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,          # collator 가 pad 자리 -100 으로 채움
        reduction="mean",
    )
    return torch.exp(loss).item()   # perplexity


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    def compute_metrics(pred):
        logits, labels = pred

        # 1) Perplexity
        ppl = _perplexity(logits, labels)

        # 2) BLEU
        labels_for_bleu = np.where(labels == -100, tokenizer.pad_token_id, labels)
        preds_txt = _post(_nd_to_tokens(logits,  tokenizer))
        refs_txt  = _post(_nd_to_tokens(labels_for_bleu, tokenizer))

        bleu_score = bleu.compute(
            predictions=preds_txt,
            references=[[r] for r in refs_txt]
        )["bleu"]

        return {"bleu": bleu_score,
                "perplexity": ppl}
    return compute_metrics
