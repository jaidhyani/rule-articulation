# In-context rule articulation & faithfulness

Exploratory study: when a language model learns a binary **classification rule from
few-shot examples alone (no chain-of-thought)**, can it *articulate* that rule — and are
its self-reports *faithful* to how it actually classifies?

Model under study: `claude-opus-4-8` with **reasoning disabled** (verified: zero thinking
tokens). A separate instance of the same model is used as generator/judge.

## What's here

| file | role |
|---|---|
| `llm.py` | API layer; all calls route through one logging chokepoint that writes full request+response transcripts to `data/transcripts/<run>.jsonl` |
| `rules.py`, `rules_catalog.md` | natural-text rules (label fn, ground-truth articulation, generator prompts) |
| `families.py` | parameterized synthetic-string rule families (integer complexity knob) |
| `data.py`, `data_quad.py` | dataset builders (single-rule; correlated two-rule quadrants) |
| `exp_boundary.py` | Step 1 — which rules are learnable in-context (accuracy sweep) |
| `search.py` | adaptive border search over rule complexity |
| `exp_execution.py` | induction-vs-execution control (learn-from-examples vs told-the-rule) |
| `exp_pairs.py` | two-rule decoy: behavior vs articulation on divergent quadrants |
| `exp_suggest.py` | commit-then-reveal suggestion experiment (neutral vs congratulatory) |
| `findings.md` | running results, framing notes, methodological endnotes |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install anthropic
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
.venv/bin/python exp_suggest.py    # or any exp_*.py
```

Generated datasets, transcripts, and logs land under `data/` (gitignored).

_Exploratory work; see `findings.md` for caveats and uncertainties._
