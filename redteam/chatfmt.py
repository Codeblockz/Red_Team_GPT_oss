from typing import List, Dict
def to_chat(messages: List[Dict], tokenizer, use_harmony: bool=True, add_special_tokens: bool=True) -> Dict:
    if hasattr(tokenizer, "apply_chat_template") and use_harmony:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        return {"text": text, "enc": enc}
    # fallback simple
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    text = "\n".join(parts)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    return {"text": text, "enc": enc}
