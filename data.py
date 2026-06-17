"""Generate + cache natural-text datasets (generator knows the rule, labels verified
programmatically where possible)."""
import json
import re
from pathlib import Path

import llm
from rules import RULES

RAW = Path(__file__).with_name("data") / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def _gen_lines(instruction, n):
    prompt = (
        f"Write {n} short, natural, everyday English sentences.\n{instruction}\n"
        "Output exactly one sentence per line. No numbering, no bullets, no quotation marks, "
        "no blank lines, no commentary."
    )
    txt = llm.generate(prompt, effort="medium")
    lines = []
    for ln in txt.splitlines():
        ln = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip().strip('"')
        if ln:
            lines.append(ln)
    return lines

def _collect(instruction, want, check, accept_label):
    """Generate until `want` examples whose check()==accept_label (or all, if check is None)."""
    got, tries = [], 0
    while len(got) < want and tries < 6:
        for s in _gen_lines(instruction, max(want - len(got), 6) + 4):
            if check is None or check(s) == accept_label:
                got.append(s)
        tries += 1
    return got[:want]

def build_single(rid, per_class=40, force=False):
    f = RAW / f"single_{rid}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    r = RULES[rid]
    pos = _collect(r.gen_pos, per_class, r.check, True)
    neg = _collect(r.gen_neg, per_class, r.check, False)
    data = {"rid": rid, "pos": pos, "neg": neg,
            "checked": r.check is not None, "articulation": r.articulation}
    f.write_text(json.dumps(data, indent=2))
    return data

def build_pair(a, b, per_class=40, force=False):
    f = RAW / f"pair_{a}__{b}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    ra, rb = RULES[a], RULES[b]
    pos_instr = (f"Every sentence must satisfy BOTH of these conditions at once:\n"
                 f"  (1) {ra.gen_pos}\n  (2) {rb.gen_pos}")
    neg_instr = (f"Every sentence must satisfy NEITHER of these — violate both:\n"
                 f"  (1) opposite of: {ra.gen_pos}\n  (2) opposite of: {rb.gen_pos}\n"
                 f"Concretely: {ra.gen_neg} {rb.gen_neg}")
    chk_pos = (lambda s: (ra.check is None or ra.check(s)) and (rb.check is None or rb.check(s)))
    chk_neg = (lambda s: (ra.check is None or not ra.check(s)) and (rb.check is None or not rb.check(s)))
    pos = _collect(pos_instr, per_class, (None if (ra.check is None and rb.check is None) else (lambda s: chk_pos(s))), True)
    # for neg we want chk_neg True; wrap so accept_label semantics line up
    neg, tries = [], 0
    while len(neg) < per_class and tries < 6:
        for s in _gen_lines(neg_instr, max(per_class - len(neg), 6) + 4):
            if chk_neg(s):
                neg.append(s)
        tries += 1
    neg = neg[:per_class]
    data = {"a": a, "b": b, "pos": pos, "neg": neg,
            "art_a": ra.articulation, "art_b": rb.articulation}
    f.write_text(json.dumps(data, indent=2))
    return data
