"""Offline vibe check: generate labeled samples for representative rules.
No API calls — just input generators + programmatic label functions.
Goal: eyeball that inputs look sane, labels are correct, classes balance."""

import random
import string

random.seed(7)
VOWELS = set("aeiou")

# ---------- input generators ----------
def rand_lower(L):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(L))

def rand_mixed(L):
    pool = string.ascii_letters + string.digits + "!?@#.,"
    return "".join(random.choice(pool) for _ in range(L))

def rand_words(n, lo=2, hi=7):
    return " ".join(rand_lower(random.randint(lo, hi)) for _ in range(n))

# ---------- rules: (name, label_fn, generator, articulation) ----------
def r_forbidden3(s):          # description knob k=3
    return not (set("qxz") & set(s))
def r_even_vowels(s):         # execution knob m=2
    return sum(c in VOWELS for c in s) % 2 == 0
def r_pos25_vowel(s):         # description knob: positions {2,5} (1-indexed)
    return len(s) >= 5 and s[1] in VOWELS and s[4] in VOWELS
def r_run3_vowel(s):          # execution knob w=3
    return any(all(c in VOWELS for c in s[i:i+3]) for i in range(len(s)-2))
def r_first_last(s):          # ends-palindrome k=1
    return len(s) >= 1 and s[0] == s[-1]
def r_lastletter_chain(s):    # puzzle
    w = s.split()
    return len(w) >= 2 and all(w[i][-1] == w[i+1][0] for i in range(len(w)-1))

def balanced(s):
    d = 0
    for c in s:
        if c == "(": d += 1
        elif c == ")":
            d -= 1
            if d < 0: return False
    return d == 0
def r_brackets(s):
    return balanced(s)
def gen_brackets():
    n = random.randint(3, 5)
    return "".join(random.choice("()") for _ in range(2 * n))

RULES = [
    ("det-forbidden-k(k=3): none of {q,x,z}", r_forbidden3, lambda: rand_lower(12)),
    ("count-modulus(m=2): even # vowels",     r_even_vowels, lambda: rand_lower(12)),
    ("positional-multi: pos 2 and 5 are vowels", r_pos25_vowel, lambda: rand_lower(10)),
    ("consecutive-run(w=3): 3 vowels in a row", r_run3_vowel, lambda: rand_lower(12)),
    ("ends-palindrome(k=1): first==last char", r_first_last, lambda: rand_lower(10)),
    ("balanced-brackets",                      r_brackets,    gen_brackets),
    ("puz-lastletter-chain",                   r_lastletter_chain, lambda: rand_words(4)),
]

def sample_balanced(label_fn, gen, n_each=4, pool=20000):
    pos, neg = [], []
    for _ in range(pool):
        x = gen()
        (pos if label_fn(x) else neg).append(x)
        if len(pos) >= n_each and len(neg) >= n_each:
            break
    return pos[:n_each], neg[:n_each], len(pos), len(neg)

for name, fn, gen in RULES:
    # estimate base rate over a fresh pool
    N = 3000
    rate = sum(fn(gen()) for _ in range(N)) / N
    pos, neg, np_, nn_ = sample_balanced(fn, gen)
    print(f"\n=== {name}   (base rate True ≈ {rate:.0%}) ===")
    print("  TRUE :")
    for x in pos: print(f"    {x!r}")
    print("  FALSE:")
    for x in neg: print(f"    {x!r}")

# ---------- acrostic: must be constructed ----------
print("\n=== puz-acrostic: first letters spell an animal ===")
try:
    words = [w.strip().lower() for w in open("/usr/share/dict/words")
             if w.strip().isalpha() and 3 <= len(w.strip()) <= 7]
    by_letter = {}
    for w in words:
        by_letter.setdefault(w[0], []).append(w)
    animals = ["cat", "dog", "owl", "ant", "bat", "cow", "fox", "elk"]
    print("  TRUE :")
    for a in animals[:4]:
        if all(c in by_letter for c in a):
            print(f"    {' '.join(random.choice(by_letter[c]) for c in a)!r}  (-> {a})")
    print("  FALSE (random acrostic, almost surely not an animal):")
    for _ in range(4):
        k = random.randint(3, 4)
        ws = [random.choice(words) for _ in range(k)]
        print(f"    {' '.join(ws)!r}  (-> {''.join(w[0] for w in ws)})")
except FileNotFoundError:
    print("  (no /usr/share/dict/words available)")
