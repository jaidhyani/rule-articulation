"""Single-rule articulation, NONTHINKING (matches the study's model-under-study frame).

For each simple rule: show k labeled in-context examples, ask the model to state the
rule in one sentence with thinking DISABLED (no reasoning), K times. A separate judge
checks whether each statement captures the target rule (same True/False partition).
Reports the articulation rate per rule. Full transcripts via llm.set_run('articulate').

Sits next to the Exp1 classification result (boundary.log: all 5 learn at 100%); this
measures whether the model can also SAY the rule, without reasoning.
"""
import concurrent.futures as cf
import json
import random

import data
import llm

RULES5 = ["lowercase", "has_digit", "question", "animal", "alliteration"]
K_SHOTS, K_ART, SEED = 16, 10, 13
ASK = ("\n\nYou have seen labeled examples above. State, in one sentence, the single rule "
       "that determines each label. Answer directly — do not show any reasoning.")


def fewshot(shots):
    return "\n".join(f"Input: {t}\nLabel: {'True' if y else 'False'}" for t, y in shots)


def run_rule(rid):
    d = data.build_single(rid, per_class=40)
    rng = random.Random(SEED)
    pos, neg = list(d["pos"]), list(d["neg"])
    rng.shuffle(pos); rng.shuffle(neg)
    half = K_SHOTS // 2
    shots = [(s, True) for s in pos[:half]] + [(s, False) for s in neg[:half]]
    rng.shuffle(shots)
    head = fewshot(shots) + ASK

    def articulate_once(_):
        llm.set_tag(experiment="articulate", pair=rid, probe="ARTICULATE")
        return llm.chat_nothink([{"role": "user", "content": head}], max_tokens=200)[0]
    with cf.ThreadPoolExecutor(max_workers=K_ART) as ex:
        texts = list(ex.map(articulate_once, range(K_ART)))

    art = d["articulation"]

    def judge_once(t):
        llm.set_tag(experiment="articulate", pair=rid, probe="JUDGE")
        return llm.judge_articulation(t, art)
    with cf.ThreadPoolExecutor(max_workers=K_ART) as ex:
        judgments = list(ex.map(judge_once, texts))

    correct = sum(1 for j in judgments if j["correct"])
    return {"rule": rid, "articulation": art, "k_shots": K_SHOTS, "n": K_ART,
            "correct": correct, "texts": texts, "judgments": judgments}


def main():
    llm.set_run("articulate")
    res = {}
    for rid in RULES5:
        r = run_rule(rid)
        res[rid] = r
        print(f"{rid:14} articulate {r['correct']}/{r['n']}", flush=True)
    json.dump(res, open("articulate_results.json", "w"), indent=2)
    print("\n=== articulation (nonthinking), k_shots=%d, K=%d ===" % (K_SHOTS, K_ART))
    for rid in RULES5:
        r = res[rid]
        print(f"  {rid:14} {r['correct']}/{r['n']}")


if __name__ == "__main__":
    main()
