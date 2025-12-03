# utils/metrics.py  ── BERTScore 제거 버전
from evaluate import load
from transformers import PreTrainedTokenizerBase
import numpy as np

rouge = load("rouge")
bleu  = load("bleu")


def _post(txts):
    return [" ".join(t.strip().split()) for t in txts]


def _nd_to_tokens(arr, tok: PreTrainedTokenizerBase):
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3:                      # logits [B,L,V] → id
            arr = arr.argmax(-1)
        arr = arr.tolist()
    if isinstance(arr[0], list):               # list[list[int]]
        return tok.batch_decode(arr, skip_special_tokens=True)
    return arr                                 # 이미 str 리스트


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    def compute_metrics(pred):
        preds, labels = pred
        # -100 → pad 토큰으로 바꿔야 디코딩 가능
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        preds_txt = _post(_nd_to_tokens(preds,   tokenizer))
        refs_txt  = _post(_nd_to_tokens(labels, tokenizer))

        r = rouge.compute(predictions=preds_txt,
                          references=refs_txt,
                          use_aggregator=True,
                          use_stemmer=True)
        
        b = bleu.compute(predictions=preds_txt,
                         references=[[t] for t in refs_txt])["bleu"]

        return {"rouge1": r["rouge1"],
                "rouge2": r["rouge2"],
                "rougeL": r["rougeL"],
                "bleu":   b}
    return compute_metrics
