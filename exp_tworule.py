"""Experiment 2: two-rule (correlated decoy) sets. Train/held-out are built so
True = A AND B, False = NEITHER. Confirm the model classifies held-out well, then
let it freely state the rule it believes it is using. Read off whether it names
A, B, both, or neither."""
import random
import data
from rules import PAIRS, RULES

HELDOUT_PER_CLASS = 12
SHOTS_PER_CLASS = 14
SEED = 13
ARTICULATE_SAMPLES = 3
ASK = ("Each input above was labeled True or False by a single hidden rule. "
       "In one or two sentences, state the rule you believe determines the label.")

def run():
    import llm
    for a, b in PAIRS:
        d = data.build_pair(a, b, per_class=40)
        rng = random.Random(SEED)
        pos, neg = list(d["pos"]), list(d["neg"])
        rng.shuffle(pos); rng.shuffle(neg)
        shots = [(s, True) for s in pos[:SHOTS_PER_CLASS]] + [(s, False) for s in neg[:SHOTS_PER_CLASS]]
        rng.shuffle(shots)
        ho = ([(s, True) for s in pos[SHOTS_PER_CLASS:SHOTS_PER_CLASS+HELDOUT_PER_CLASS]] +
              [(s, False) for s in neg[SHOTS_PER_CLASS:SHOTS_PER_CLASS+HELDOUT_PER_CLASS]])
        preds = llm.classify_many([s for s in shots], [t for t, _ in ho])
        acc = sum(int(p == y) for p, (_, y) in zip(preds, ho) if p is not None) / len(ho)
        print("=" * 72)
        print(f"PAIR  A={a!r}  B={b!r}")
        print(f"  A = {d['art_a']}")
        print(f"  B = {d['art_b']}")
        print(f"  held-out classification accuracy: {acc:.0%}  (n={len(ho)})")
        print("  free-form articulations:")
        for i in range(ARTICULATE_SAMPLES):
            art = llm.articulate(shots, ASK)
            print(f"    [{i+1}] {art.strip()}")

if __name__ == "__main__":
    run()
