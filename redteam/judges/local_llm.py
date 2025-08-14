"""
Optional local LLM judge: upgrades precision on top candidates.
If transformers pipeline/model not available, returns neutral score.
"""
def judge_llm(text: str):
    try:
        # from transformers import pipeline
        # clf = pipeline("text-classification", model="Qwen2.5-7B-Instruct", device_map="auto")
        # out = clf(text, truncation=True)
        # score = min(1.0, max(x['score'] for x in out))
        score = 0.0  # placeholder to avoid heavy deps by default
        return {"llm_score": score}
    except Exception:
        return {"llm_score": 0.0}
