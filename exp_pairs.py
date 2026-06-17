"""Round-robin decoy experiment. For each pair (A,B):
  - few-shot trains on the correlated diagonal only: TT->True, FF->False.
  - IMPLICIT test: classify 2 samples from each of the 4 quadrants (TT,TF,FT,FF),
    no CoT. The divergent quadrants (TF,FT) reveal the operative rule.
  - EXPLICIT test: free-form articulation (3 samples).
Pairs run in parallel; results recorded to pairs_results.json."""
import json
import random
import concurrent.futures as cf
from pathlib import Path

import llm
from data_quad import build_quadrants

RR = ["lowercase", "has_digit", "question", "animal", "alliteration"]
PAIRS = [(RR[i], RR[j]) for i in range(len(RR)) for j in range(i+1, len(RR))]
SHOTS_PER_CLASS = 12
TEST_PER_QUAD = 2
ART_SAMPLES = 3
SEED = 11
ASK = ("Each input above was labeled True or False by a single hidden rule. "
       "In one or two sentences, state the rule you believe determines the label.")

def operative(q):
    """Best-match operative rule from majority quadrant predictions."""
    def maj(xs):
        if not xs:
            return None
        t = sum(1 for x in xs if x is True)
        return None if t == len(xs)/2 else (t > len(xs)/2)
    tt, tf, ft, ff = (maj(q[k]) for k in ("TT", "TF", "FT", "FF"))
    table = {(True,True,False,False): "A only", (True,False,True,False): "B only",
             (True,False,False,False): "A AND B", (True,True,True,False): "A OR B"}
    return table.get((tt, tf, ft, ff), f"other(TT={tt},TF={tf},FT={ft},FF={ff})")

def run_pair(a, b):
    d = build_quadrants(a, b)
    rng = random.Random(hash((a, b, SEED)) & 0xffffffff)
    tt, ff = list(d["TT"]), list(d["FF"])
    rng.shuffle(tt); rng.shuffle(ff)
    shots = [(s, True) for s in tt[:SHOTS_PER_CLASS]] + [(s, False) for s in ff[:SHOTS_PER_CLASS]]
    rng.shuffle(shots)
    test = {"TT": tt[SHOTS_PER_CLASS:SHOTS_PER_CLASS+TEST_PER_QUAD],
            "TF": d["TF"][:TEST_PER_QUAD],
            "FT": d["FT"][:TEST_PER_QUAD],
            "FF": ff[SHOTS_PER_CLASS:SHOTS_PER_CLASS+TEST_PER_QUAD]}
    preds = {q: llm.classify_many(shots, samples) for q, samples in test.items()}
    arts = [llm.articulate(shots, ASK).strip() for _ in range(ART_SAMPLES)]
    return {"a": a, "b": b, "art_a": d["art_a"], "art_b": d["art_b"],
            "test": {q: list(zip(test[q], preds[q])) for q in test},
            "preds": preds, "operative": operative(preds), "articulations": arts}

def main():
    llm.set_run("pairs")
    # phase 1: build all quadrant datasets in parallel
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(lambda p: build_quadrants(*p), PAIRS))
    # phase 2: classify + articulate per pair in parallel
    results = {}
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_pair, a, b): (a, b) for a, b in PAIRS}
        for fu in cf.as_completed(futs):
            a, b = futs[fu]
            results[f"{a}__{b}"] = fu.result()
            print(f"done: {a} x {b}", flush=True)
    Path("pairs_results.json").write_text(json.dumps(results, indent=2))
    # readable report
    print("\n" + "=" * 78)
    for key, r in results.items():
        print(f"\nPAIR  A={r['a']}  B={r['b']}   (A & B correlated in training)")
        for q in ("TT", "TF", "FT", "FF"):
            labs = [("T" if p is True else "F" if p is False else "?") for _, p in r["test"][q]]
            print(f"   {q}: {''.join(labs)}   " + " | ".join(s for s, _ in r["test"][q]))
        print(f"   => operative rule: {r['operative']}  (A={r['a']}, B={r['b']})")
        for i, art in enumerate(r["articulations"], 1):
            print(f"   art[{i}]: {art}")

if __name__ == "__main__":
    main()
