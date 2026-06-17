"""Induction vs execution. For rules that failed in-context (Exp/search), measure:
  ICL  = infer rule from 32 labeled examples, then classify (no CoT)
  TOLD = rule stated explicitly, then classify (no CoT)
A big TOLD>>ICL gap means the failure was induction (couldn't find the rule);
TOLD still low means a genuine execution limit (can't compute it even when told)."""
import random
import llm
from families import FAMILIES

N = 60
SEED = 5

RULE_TEXT = {
    "pos_vowel": lambda k: f"Label True if and only if character number {k} (counting from 1) of the input is a vowel (a, e, i, o, or u).",
    "count_mod": lambda k: f"Label True if and only if the number of vowels in the input is divisible by {k}.",
    "palindrome_ends": lambda k: f"Label True if and only if the first {k} characters of the input equal the last {k} characters reversed.",
    "sorted_prefix": lambda k: f"Label True if and only if the first {k} characters of the input are in strictly increasing alphabetical order.",
    "run_len": lambda k: f"Label True if and only if the input contains a run of at least {k} identical characters in a row.",
}

# (family, k) probe points — where ICL failed
POINTS = [
    ("pos_vowel", 2), ("pos_vowel", 7),
    ("palindrome_ends", 3),
    ("sorted_prefix", 2), ("sorted_prefix", 4),
    ("run_len", 2), ("run_len", 4),
    ("count_mod", 2), ("count_mod", 3),
]

def make_eval(fam, k, rng):
    return ([(fam.make(k, True, rng), True) for _ in range(N//2)] +
            [(fam.make(k, False, rng), False) for _ in range(N//2)])

def acc(preds, he):
    ok = [p == y for p, (_, y) in zip(preds, he) if p is not None]
    return sum(ok)/len(ok)

def main():
    llm.set_run("execution")
    print(f"{'family':<16} {'k':>2}  {'ICL':>5}  {'TOLD':>5}  gap")
    for nm, k in POINTS:
        fam = FAMILIES[nm]
        rng = random.Random(hash((nm, k, SEED)) & 0xffffffff)
        he = make_eval(fam, k, rng)
        # ICL: 32 fresh labeled examples as few-shot
        half = 16
        shots = ([(fam.make(k, True, rng), True) for _ in range(half)] +
                 [(fam.make(k, False, rng), False) for _ in range(half)])
        rng.shuffle(shots)
        icl = acc(llm.classify_many(shots, [s for s, _ in he]), he)
        told = acc(llm.classify_told_many(RULE_TEXT[nm](k), [s for s, _ in he]), he)
        print(f"{nm:<16} {k:>2}  {icl:>5.0%}  {told:>5.0%}  {told-icl:+.0%}", flush=True)

if __name__ == "__main__":
    main()
