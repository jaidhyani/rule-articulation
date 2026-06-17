"""Content-controlled lowercase dataset (round 3, per Jai).

Changes from round 2:
  - LONGER sentences (>=10 words), each with MULTIPLE would-be-capitalized words
    (>=2 internal proper nouns, plus the sentence-initial position).
  - Negatives kept in NATURAL form: first letter uppercase too (round 2 had lowercased it).
    So the rule is the true "entirely lowercase" (ANY capital, including sentence-initial),
    with redundant capital cues rather than a single buried internal capital.

One common pool (longer, multi-proper-noun sentences), disjoint halves: one half fully
lowercased -> positives; the other half natural (first letter + proper nouns capitalized)
-> negatives. Content distribution matched; not case-toggled twins.
"""
import random

import data
import llm

SEED = 13
PER_CLASS = 40
POOL_INSTR = (
    "Write LONGER sentences, about 16 to 30 words each. Each sentence must mention at least "
    "THREE specific proper nouns from different categories — e.g. a person's name, a city or "
    "country, a company or brand, a day of the week, or a month. Vary the grammatical person "
    "across the set (first person I/we, second person you, and third person he/she/they). Vary "
    "the ending punctuation."
)


def usable(s):
    toks = s.split()
    if len(toks) < 10:
        return False
    internal_caps = sum(1 for w in toks[1:] if w[:1].isupper())
    return internal_caps >= 2


def main():
    llm.set_run("lowercase_redo3")
    llm.set_tag(experiment="lowercase_redo3", probe="GENERATE")
    pool, tries = [], 0
    while len(pool) < 2 * PER_CLASS + 30 and tries < 12:
        for s in data._gen_lines(POOL_INSTR, 50):
            if usable(s) and s not in pool:
                pool.append(s)
        tries += 1
    rng = random.Random(SEED)
    rng.shuffle(pool)
    half = len(pool) // 2
    src_pos, src_neg = pool[:half], pool[half:]

    check = data.RULES["lowercase"].check
    pos = [s.lower() for s in src_pos]
    pos = [s for s in pos if check(s)][:PER_CLASS]
    neg = [s[0].upper() + s[1:] for s in src_neg]      # natural: capitalize first letter
    neg = [s for s in neg if not check(s)][:PER_CLASS]

    d = {"rid": "lowercase", "pos": pos, "neg": neg,
         "checked": True, "articulation": data.RULES["lowercase"].articulation}
    (data.RAW / "single_lowercase.json").write_text(__import__("json").dumps(d, indent=2))

    def caps_count(s):
        return sum(1 for c in s if c.isupper())
    def words(s):
        return len(s.split())
    print(f"pool usable: {len(pool)} | positives: {len(pos)} | negatives: {len(neg)}")
    print(f"avg words  pos={sum(words(s) for s in pos)/len(pos):.1f}  neg={sum(words(s) for s in neg)/len(neg):.1f}")
    print(f"avg capitals in negatives: {sum(caps_count(s) for s in neg)/len(neg):.1f} (want >1)")
    print("sample pos:", pos[:2])
    print("sample neg:", neg[:2])

    # stable classification over several shot draws
    accs = []
    for seed in range(1, 6):
        rng2 = random.Random(seed)
        p, n = list(pos), list(neg)
        rng2.shuffle(p); rng2.shuffle(n)
        shots = [(s, True) for s in p[:8]] + [(s, False) for s in n[:8]]
        rng2.shuffle(shots)
        q = [(s, True) for s in p[8:]] + [(s, False) for s in n[8:]]
        rng2.shuffle(q)
        queries = [t for t, _ in q]; truth = [y for _, y in q]
        llm.set_tag(experiment="lowercase_redo3", probe="CLASSIFY", cond=f"seed{seed}")
        preds = llm.classify_many(shots, queries)
        acc = sum(int(pr == y) for pr, y in zip(preds, truth) if pr is not None) / len(truth)
        accs.append(acc)
        print(f"  seed{seed}: {acc:.1%} (n={len(truth)})")
    print(f"classification mean over {len(accs)} draws: {sum(accs)/len(accs):.1%}  "
          f"(min {min(accs):.0%}, max {max(accs):.0%})")


if __name__ == "__main__":
    main()
