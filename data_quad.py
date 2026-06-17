"""Build the four (A,B) quadrants for a pair: TT=A&B, TF=A&~B, FT=~A&B, FF=~A&~B.
Generator knows the target combo; labels verified programmatically where checkable."""
import json
import re
import threading
from pathlib import Path

import llm
from rules import RULES

RAW = Path(__file__).with_name("data") / "raw"
RAW.mkdir(parents=True, exist_ok=True)
_io_lock = threading.Lock()

def _gen_lines(instruction, n):
    prompt = (f"Write {n} short, natural, everyday English sentences.\n{instruction}\n"
              "Output exactly one sentence per line. No numbering, no bullets, no quotation "
              "marks, no blank lines, no commentary.")
    txt = llm.generate(prompt, effort="medium")
    out = []
    for ln in txt.splitlines():
        ln = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip().strip('"')
        if ln:
            out.append(ln)
    return out

def _instr(ra, wa, rb, wb):
    ia = ra.gen_pos if wa else ra.gen_neg
    ib = rb.gen_pos if wb else rb.gen_neg
    return f"Every sentence must satisfy BOTH of these conditions at once:\n  (1) {ia}\n  (2) {ib}"

import functools

@functools.lru_cache(maxsize=8192)
def _sat_semantic(sentence, articulation):
    return llm.judge_satisfies(sentence, articulation)

def _sat(rule, s):
    # programmatic check where available; judge-verify semantic rules
    return rule.check(s) if rule.check is not None else _sat_semantic(s, rule.articulation)

def _ok(ra, wa, rb, wb, s):
    return _sat(ra, s) == wa and _sat(rb, s) == wb

def _collect(ra, wa, rb, wb, want):
    got, tries = [], 0
    instr = _instr(ra, wa, rb, wb)
    while len(got) < want and tries < 8:
        for s in _gen_lines(instr, max(want - len(got), 6) + 4):
            if _ok(ra, wa, rb, wb, s) and s not in got:
                got.append(s)
        tries += 1
    return got[:want]

def build_quadrants(a, b, n_tt_ff=14, n_div=4, force=False):
    import concurrent.futures as cf
    f = RAW / f"quad_{a}__{b}.json"
    if f.exists() and not force:
        d = json.loads(f.read_text())
        if (len(d["TT"]) >= n_tt_ff and len(d["FF"]) >= n_tt_ff and
                len(d["TF"]) >= n_div and len(d["FT"]) >= n_div):
            return d  # cache is large enough
    ra, rb = RULES[a], RULES[b]
    specs = {"TT": (True, True, n_tt_ff), "FF": (False, False, n_tt_ff),
             "TF": (True, False, n_div), "FT": (False, True, n_div)}
    out = {}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_collect, ra, wa, rb, wb, n): q for q, (wa, wb, n) in specs.items()}
        for fu in futs:
            out[futs[fu]] = fu.result()
    data = {"a": a, "b": b, "art_a": ra.articulation, "art_b": rb.articulation, **out}
    with _io_lock:
        f.write_text(json.dumps(data, indent=2))
    return data
