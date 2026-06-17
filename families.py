"""Parameterized rule families with an integer complexity knob `k`.
Each family constructs balanced, distribution-matched inputs via make(k, label):
positives and negatives are built the same way, flipping only the rule feature.
Inputs are synthetic strings (exact labels, no generator LLM needed)."""
import random
import string

LOWER = string.ascii_lowercase
VOWELS = "aeiou"
CONS = "".join(c for c in LOWER if c not in VOWELS)

def _maxrun(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i-1] else 1
        best = max(best, run)
    return best if seq else 0

class Family:
    def __init__(self, name, make, kmin, kmax, desc):
        self.name = name; self.make = make; self.kmin = kmin; self.kmax = kmax; self.desc = desc

def _pos_vowel(k, label, rng):
    L = k + rng.randint(2, 12)
    cs = [rng.choice(LOWER) for _ in range(L)]
    cs[k-1] = rng.choice(VOWELS if label else CONS)
    return "".join(cs)

def _count_mod(m, label, rng):
    L = 24
    ok = [c for c in range(L+1) if (c % m == 0) == label]
    c = rng.choice(ok)
    pos = set(rng.sample(range(L), c))
    return "".join(rng.choice(VOWELS) if i in pos else rng.choice(CONS) for i in range(L))

def _palindrome_ends(k, label, rng):
    pre = [rng.choice(LOWER) for _ in range(k)]
    mid = [rng.choice(LOWER) for _ in range(rng.randint(0, 8))]
    suf = list(reversed(pre))
    if not label:
        i = rng.randrange(k)
        suf[i] = rng.choice([x for x in LOWER if x != suf[i]])
    return "".join(pre + mid + suf)

def _sorted_prefix(k, label, rng):
    suf = [rng.choice(LOWER) for _ in range(rng.randint(2, 10))]
    if label:
        pre = sorted(rng.sample(LOWER, k))
    else:
        while True:
            pre = [rng.choice(LOWER) for _ in range(k)]
            if any(pre[i] >= pre[i+1] for i in range(k-1)):
                break
    return "".join(pre + suf)

def _run_len(w, label, rng):
    L = 22
    if label:
        cs = [rng.choice(LOWER) for _ in range(L)]
        ch = rng.choice(LOWER); start = rng.randint(0, L-w)
        for i in range(start, start+w): cs[i] = ch
        if start > 0 and cs[start-1] == ch: cs[start-1] = rng.choice([x for x in LOWER if x != ch])
        if start+w < L and cs[start+w] == ch: cs[start+w] = rng.choice([x for x in LOWER if x != ch])
    else:
        while True:
            cs = [rng.choice(LOWER) for _ in range(L)]
            if _maxrun(cs) < w:
                break
    return "".join(cs)

FAMILIES = {f.name: f for f in [
    Family("pos_vowel", _pos_vowel, 1, 40,
           "char at position k is a vowel (knob = position)"),
    Family("count_mod", _count_mod, 2, 12,
           "number of vowels is divisible by m (knob = modulus)"),
    Family("palindrome_ends", _palindrome_ends, 1, 16,
           "first k chars equal the last k reversed (knob = k)"),
    Family("sorted_prefix", _sorted_prefix, 2, 16,
           "first k chars are in strictly increasing order (knob = k)"),
    Family("run_len", _run_len, 2, 14,
           "contains a run of >= w identical chars (knob = w)"),
]}
