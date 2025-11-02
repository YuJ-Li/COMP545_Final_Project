# hf_llm.py
from typing import Tuple, Optional
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


# -------------------------------
# Model loader
# -------------------------------

def load_llm(model_name: str, device: Optional[str] = None, four_bit: bool = False) -> Tuple:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(trust_remote_code=True)
    if four_bit and device == "cuda":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs.update(dict(quantization_config=quant_cfg, device_map="auto"))
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        kwargs.update(dict(dtype=dtype))

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device == "cuda" and not four_bit:
        model = model.to("cuda")
    model.eval()
    return tok, model


# -------------------------------
# Optional: H-aware stopping
# -------------------------------

class JsonHStop(StoppingCriteria):
    def __init__(self, tok, H: int):
        self.tok = tok
        self.H = int(H)
        self.buf = ""

    def __call__(self, input_ids, scores, **kwargs):
        new_txt = self.tok.decode(input_ids[0][-1:], skip_special_tokens=True)
        self.buf += new_txt
        m = re.search(r"\[([^\]]*)$", self.buf)
        if not m:
            return False
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
        return len(nums) >= self.H


# -------------------------------
# Generation wrapper
# -------------------------------

@torch.no_grad()
def llm_call(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    H: Optional[int] = None,
) -> str:
    """
    Generate continuation and return only the generated part (no prompt).
    Handles generation errors gracefully (returns "").
    """
    try:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        do_sample = (temperature and float(temperature) > 0.0) or (top_p and float(top_p) < 1.0)
        if do_sample and seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.pad_token_id,
            do_sample=bool(do_sample),
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=float(temperature), top_p=float(top_p)))

        # optional early stop if enough numbers generated
        if H is not None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([JsonHStop(tok, int(H))])

        out = model.generate(**inputs, **gen_kwargs)
        gen_only = out[0][input_len:]
        text = tok.decode(gen_only, skip_special_tokens=True)
        return text or ""
    except Exception as e:
        print(f"[ERR][GEN] {type(e).__name__}: {e}")
        return ""
