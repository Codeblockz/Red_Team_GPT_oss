from .config import Config
from .hf_backend import HFRunner
from .bandits import UCB1
from .seeds import topic_seed_messages
from .mutators import vary
from .judges.heuristic import judge as h_judge
from .judges.local_llm import judge_llm
from .dedupe import LSHDeduper
from .utils import now_ms, sha

def run_batch(cfg: Config, limit: int = 200):
    runner = HFRunner(cfg)
    fam = topic_seed_messages()
    bandit = UCB1(len(fam))
    ded = LSHDeduper(threshold=0.85)
    out = []
    for _ in range(limit):
        i = bandit.select()
        family, base = fam[i]
        best_score = 0.0
        best_meta = None
        for msgs in vary(base):
            key = sha({"f":family, "m":msgs})
            # Dedupe by near-duplicate of the input transcript (messages text only)
            if not ded.add(key, " ".join(m['content'] for m in msgs)):
                continue
            res = runner.generate_chat(msgs)
            j = h_judge(res["gen_text"])
            llm = judge_llm(res["gen_text"])
            j["score"] = min(1.0, j["score"]*0.7 + llm.get("llm_score",0.0)*0.3)
            meta = {
                "ts": now_ms(),
                "family": family,
                "messages": msgs,
                "prompt_text": res["prompt_text"],
                "full_text": res["full_text"],
                "response": res["gen_text"],
                "model": res["model"],
                "gen": res["gen_params"],
                "judge": j,
            }
            if j["score"] > best_score:
                best_score = j["score"]; best_meta = meta
        bandit.update(i, best_score)
        if best_meta:
            out.append(best_meta)
    return out
