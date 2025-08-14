from dataclasses import asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import set_seed
from .chatfmt import to_chat

class HFRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
        dtype = getattr(torch, cfg.model.dtype) if hasattr(torch, cfg.model.dtype) else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, torch_dtype=dtype, device_map="auto")
        self.model.eval()

    def generate_chat(self, messages):
        cfg = self.cfg
        set_seed(cfg.model.seed)
        packed = to_chat(messages, self.tok, use_harmony=cfg.model.use_harmony_chat_template, add_special_tokens=cfg.model.add_special_tokens)
        inputs = {k: v.to(self.model.device) for k, v in packed["enc"].items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.model.max_new_tokens,
            temperature=cfg.model.temperature,
            top_p=cfg.model.top_p,
            do_sample=cfg.model.temperature > 0,
        )
        full = self.tok.decode(out[0], skip_special_tokens=False)
        gen_only = self.tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        return {
            "prompt_text": packed["text"],
            "full_text": full,
            "gen_text": gen_only,
            "input_ids": inputs["input_ids"].tolist(),
            "generated_ids": out[0].tolist(),
            "model": asdict(cfg.model),
            "gen_params": {"max_new_tokens": cfg.model.max_new_tokens, "temperature": cfg.model.temperature, "top_p": cfg.model.top_p, "seed": cfg.model.seed},
        }
