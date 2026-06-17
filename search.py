"""Adaptive border search: for each family, exponentially scan the knob until
held-out no-CoT accuracy drops below THRESH, then binary-search the crossing.
Synthetic inputs => only classification hits the API."""
import json
import random
import sys
from pathlib import Path

import llm
from families import FAMILIES

THRESH = 0.90
N_SHOTS = 32
N_EVAL = 32          # held-out per knob value during search
N_CONFIRM = 80       # larger re-check at the located border

def acc_at(fam, k, n_eval=N_EVAL, seed=0):
    rng = random.Random(hash((fam.name, k, seed)) & 0xffffffff)
    half = N_SHOTS // 2
    shots = ([(fam.make(k, True, rng), True) for _ in range(half)] +
             [(fam.make(k, False, rng), False) for _ in range(half)])
    rng.shuffle(shots)
    he = ([(fam.make(k, True, rng), True) for _ in range(n_eval//2)] +
          [(fam.make(k, False, rng), False) for _ in range(n_eval//2)])
    preds = llm.classify_many(shots, [s for s, _ in he])
    ok = [p == y for p, (_, y) in zip(preds, he) if p is not None]
    return sum(ok) / len(ok)

def search(fam):
    res = {}
    def probe(k):
        if k not in res:
            res[k] = acc_at(fam, k)
            print(f"  {fam.name}: k={k:<3} acc={res[k]:.0%}", flush=True)
        return res[k]
    # exponential scan
    k, last_pass, first_fail = fam.kmin, None, None
    while k <= fam.kmax:
        if probe(k) >= THRESH:
            last_pass = k
        else:
            first_fail = k; break
        k = max(k+1, k*2)
    verdict = None
    if first_fail is None:
        verdict = ("viable_through_kmax", last_pass)
    elif last_pass is None:
        verdict = ("fails_at_kmin", first_fail)
    else:
        lo, hi = last_pass, first_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if probe(mid) >= THRESH:
                lo = mid
            else:
                hi = mid
        # confirm the border at higher n
        ca = acc_at(fam, lo, n_eval=N_CONFIRM)
        print(f"  {fam.name}: border k*={lo} confirm acc={ca:.0%} (n={N_CONFIRM})", flush=True)
        verdict = ("border", lo, {"confirm_acc": ca})
    return {"family": fam.name, "desc": fam.desc, "probed": res, "verdict": verdict}

def main(names):
    out = {}
    for nm in names:
        print(f"== {nm}: {FAMILIES[nm].desc}", flush=True)
        out[nm] = search(FAMILIES[nm])
    Path("search_results.json").write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    for nm, r in out.items():
        print(f"{nm:<16} {r['verdict']}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:] or list(FAMILIES))
