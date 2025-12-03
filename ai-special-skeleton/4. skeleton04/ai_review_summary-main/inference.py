# inference.py
"""
4-bit 양자화 모델 요약 추론
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utils.data import _make_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CONFIG = dict(
    # --- 이 경로를 LoRA 체크포인트 또는 4-bit 모델로 변경할 수 있습니다 ---
    # 옵션 1: LoRA
    ckpt="checkpoints/gpt2-lora/checkpoint-62",  #해당 부분을 /checkpoints 폴더를 확인하여 변경
    # 옵션 2: 병합 및 양자화된 모델
    # ckpt="checkpoints/gpt2-bnb-4bit",           
    review="I purchased this formula but was worried after reading the comments here that my 5 month old baby would suffer from constipation.  He did.  However, I really wanted to use organic formula so I added a few teaspoons of prunes to his cereal and within 12 hours - problem solved.  No constipation since and he has been on this formula for about 2 weeks. I give him some prune/cereal every 4 days.  If your baby is not yet on solids you might consider giving him a little apple or pear juice mixed with water.  This should do the trick also.  Don't let the constipion issue scare you o100",
    summary="If you're worried about consitpation....",
)


def main(cfg=CONFIG):
    print(f"{cfg['ckpt']}에서 모델을 로드합니다...")
    is_lora_adapter = os.path.exists(os.path.join(cfg["ckpt"], "adapter_config.json"))

    if is_lora_adapter:
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
        
        # LoRA 어댑터를 로드하고 병합합니다.
        model = PeftModel.from_pretrained(model, cfg["ckpt"]).merge_and_unload()

    else:
        
        # 4-bit 모델을 로드하기 위한 필수 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 최종 모델 경로에서 모델과 토크나이저를 직접 로드합니다.
        model = AutoModelForCausalLM.from_pretrained(
            cfg["ckpt"],
            quantization_config=bnb_config,
            device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(cfg["ckpt"])

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.config.pad_token_id = tok.pad_token_id
    
    model.eval()

    ids = tok(
        _make_prompt(text=cfg["review"]),
        return_tensors="pt",
    ).to(model.device)
    
    print("요약 생성 중...")
    out = model.generate(
        **ids,
        max_new_tokens=120,
        do_sample=True,           
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,    
        no_repeat_ngram_size=3,   
        eos_token_id=tok.eos_token_id,
    )
    summary = tok.decode(out[0], skip_special_tokens=True).split("TL;DR: ")[-1].strip()
    print("\n요약:\n", summary)


if __name__ == "__main__":
    main()
