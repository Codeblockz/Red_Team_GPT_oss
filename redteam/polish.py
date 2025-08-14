def polish(meta: dict) -> dict:
    flags = meta.get("judge",{}).get("flags",{})
    topic_hits = [k for k,v in flags.items() if v]
    title = f"{meta['family']} — " + (", ".join(topic_hits) if topic_hits else "candidate")
    return {
        "title": title[:120],
        "category": meta["family"],
        "why_it_matters": "Behavior aligned with a Topic of Interest; investigate impact and reproducibility.",
        "messages": meta["messages"],
        "response_excerpt": meta["response"][:1200],
        "harmony_transcript": meta.get("full_text",""),
        "score": meta.get("judge",{}).get("score",0.0),
        "model": meta.get("model",{}),
        "gen": meta.get("gen",{}),
    }
def top_five(cands: list) -> list:
    ranked = sorted(cands, key=lambda m: m.get("judge",{}).get("score",0.0), reverse=True)
    return [polish(x) for x in ranked[:5]]
