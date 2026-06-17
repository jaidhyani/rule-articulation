"""Rebuild the de-confounded lowercase single-rule dataset, verify the confound is
gone, and re-check classification accuracy (nonthinking). Logged transcripts.

The old lowercase set confounded case with punctuation/start-position (all positives were
unpunctuated lowercase fragments; all negatives were capitalized+punctuated sentences). The
new generators vary punctuation and capital position independently so only "any uppercase?"
predicts the label.
"""
import random

import data
import llm
from exp_boundary import split

SEED = 13


def term_punct(s):
    return s.strip().endswith((".", "?", "!"))


def starts_lower(s):
    s = s.strip()
    return bool(s) and s[0].islower()


def main():
    llm.set_run("lowercase_redo")
    llm.set_tag(experiment="lowercase_redo", probe="GENERATE")
    d = data.build_single("lowercase", force=True)
    pos, neg = d["pos"], d["neg"]
    print(f"positives: {len(pos)} | with terminal punctuation: {sum(term_punct(s) for s in pos)}")
    print(f"negatives: {len(neg)} | starting lowercase: {sum(starts_lower(s) for s in neg)} | "
          f"without terminal punctuation: {sum(not term_punct(s) for s in neg)}")
    print("sample pos:", [pos[i] for i in range(min(4, len(pos)))])
    print("sample neg:", [neg[i] for i in range(min(4, len(neg)))])

    pool_pos, pool_neg, ho = split(d)
    queries = [t for t, _ in ho]
    truth = [y for _, y in ho]
    shots = [(s, True) for s in pool_pos[:8]] + [(s, False) for s in pool_neg[:8]]
    random.Random(SEED).shuffle(shots)
    llm.set_tag(experiment="lowercase_redo", probe="CLASSIFY")
    preds = llm.classify_many(shots, queries)
    acc = sum(int(p == y) for p, y in zip(preds, truth) if p is not None) / len(truth)
    print(f"\nlowercase classification accuracy (k=16, held-out {len(truth)}): {acc:.0%}")


if __name__ == "__main__":
    main()
