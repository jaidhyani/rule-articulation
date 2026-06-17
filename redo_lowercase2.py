"""Content-controlled lowercase dataset (round 2).

Round 1's de-confound (vary punctuation/position) exposed a deeper confound: asked to
generate lowercase vs capitalized sentences, the model makes lowercase ones casual
first/second-person and capitalized ones third-person with proper nouns. It then
classifies by PERSON, not case.

Fix per Jai: draw both classes from ONE common distribution so content/person/proper-noun
density match, differing only in capitalization — but NOT as case-toggled twins of the
same sentence. Method: generate a pool of natural sentences rich in proper nouns; split
into disjoint halves; one half fully lowercased -> positives; the other half with only the
first character lowercased (internal proper-noun capital kept) -> negatives. Both classes
start lowercase and both carry name/place content; the sole separating feature is the
presence of any uppercase letter.
"""
import random

import data
import llm

SEED = 13
PER_CLASS = 40
POOL_INSTR = (
    "Vary the grammatical person across the set: mix first person (I, we), second person "
    "(you / direct address), and third person (he, she, they, named people). MOST sentences "
    "must mention at least one specific proper noun — a person's name, a city, a country, a "
    "company, a day of the week, or a month. Vary length and ending punctuation."
)


def usable(s):
    """After lowercasing only the first character, does an uppercase letter remain?
    (i.e. the sentence carries an internal/proper-noun capital, not just sentence-initial.)"""
    if len(s) < 2:
        return False
    s2 = s[0].lower() + s[1:]
    return any(c.isupper() for c in s2)


def main():
    llm.set_run("lowercase_redo2")
    llm.set_tag(experiment="lowercase_redo2", probe="GENERATE")
    pool, tries = [], 0
    while len(pool) < 2 * PER_CLASS + 30 and tries < 10:
        for s in data._gen_lines(POOL_INSTR, 60):
            if usable(s) and s not in pool:
                pool.append(s)
        tries += 1
    rng = random.Random(SEED)
    rng.shuffle(pool)
    half = len(pool) // 2
    src_pos, src_neg = pool[:half], pool[half:]

    r = data.RULES["lowercase"]
    check = r.check
    pos = [s.lower() for s in src_pos]
    pos = [s for s in pos if check(s)][:PER_CLASS]
    neg = [s[0].lower() + s[1:] for s in src_neg]
    neg = [s for s in neg if not check(s)][:PER_CLASS]

    d = {"rid": "lowercase", "pos": pos, "neg": neg,
         "checked": True, "articulation": r.articulation}
    (data.RAW / "single_lowercase.json").write_text(__import__("json").dumps(d, indent=2))

    def starts_lower(s):
        s = s.strip()
        return bool(s) and s[0].islower()
    print(f"pool usable: {len(pool)} | positives: {len(pos)} | negatives: {len(neg)}")
    print(f"positives starting lowercase: {sum(starts_lower(s) for s in pos)}/{len(pos)} "
          f"(all should; none contain any capital)")
    print(f"negatives starting lowercase: {sum(starts_lower(s) for s in neg)}/{len(neg)} "
          f"(all should; each has an internal capital)")
    print("sample pos:", pos[:4])
    print("sample neg:", neg[:4])

    # classification check, k=16
    import random as _r
    rng2 = _r.Random(SEED)
    p2, n2 = list(pos), list(neg)
    rng2.shuffle(p2); rng2.shuffle(n2)
    ho = [(s, True) for s in p2[:12]] + [(s, False) for s in n2[:12]]
    rng2.shuffle(ho)
    shots = [(s, True) for s in p2[12:20]] + [(s, False) for s in n2[12:20]]
    rng2.shuffle(shots)
    queries = [t for t, _ in ho]; truth = [y for _, y in ho]
    llm.set_tag(experiment="lowercase_redo2", probe="CLASSIFY")
    preds = llm.classify_many(shots, queries)
    acc = sum(int(p == y) for p, y in zip(preds, truth) if p is not None) / len(truth)
    print(f"\nlowercase classification accuracy (k=16, held-out {len(truth)}): {acc:.0%}")


if __name__ == "__main__":
    main()
