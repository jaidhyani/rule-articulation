"""Experiment 1: can Opus 4.8 (no reasoning) learn each rule in-context, and where
is the tractability boundary? Sweep #shots; report held-out accuracy."""
import random
import data
from rules import RULES

BOUNDARY_RULES = ["lowercase", "has_digit", "question", "animal", "alliteration", "acrostic_animal"]
SHOTS = [8, 16, 32]
HELDOUT_PER_CLASS = 12
SEED = 13

def split(d):
    rng = random.Random(SEED)
    pos, neg = list(d["pos"]), list(d["neg"])
    rng.shuffle(pos); rng.shuffle(neg)
    ho = [(s, True) for s in pos[:HELDOUT_PER_CLASS]] + [(s, False) for s in neg[:HELDOUT_PER_CLASS]]
    rng.shuffle(ho)
    return pos[HELDOUT_PER_CLASS:], neg[HELDOUT_PER_CLASS:], ho

def run():
    import llm
    llm.set_run("boundary")
    print(f"{'rule':<16} " + " ".join(f"k={k:<5}" for k in SHOTS))
    for rid in BOUNDARY_RULES:
        d = data.build_single(rid, per_class=40)
        pool_pos, pool_neg, ho = split(d)
        queries = [t for t, _ in ho]; truth = [y for _, y in ho]
        row = []
        for k in SHOTS:
            half = k // 2
            shots = [(s, True) for s in pool_pos[:half]] + [(s, False) for s in pool_neg[:half]]
            random.Random(SEED).shuffle(shots)
            preds = llm.classify_many(shots, queries)
            acc = sum(int(p == y) for p, y in zip(preds, truth) if p is not None) / len(truth)
            row.append(acc)
        print(f"{rid:<16} " + " ".join(f"{a:<7.0%}" for a in row)
              + ("   [checked]" if d["checked"] else "   [semantic/trusted]"))

if __name__ == "__main__":
    run()
