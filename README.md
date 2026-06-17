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
| `exp_articulate.py` | single-rule articulation, no-CoT, judged for correctness |
| `reproduce.sh` | one-command reproduction harness (sets up the venv, runs any/all stages) |
| `findings.md` | running results, framing notes, methodological endnotes |

## Reproduce

Given an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or: echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
./reproduce.sh                        # prints the stage list + cost estimates (runs nothing)
./reproduce.sh boundary articulate    # run specific stages
./reproduce.sh all                    # full pipeline (the `suggest` stage alone is ~2200 Opus calls)
```

`reproduce.sh` creates `.venv` and installs `requirements.txt` on first run. Input datasets
are committed under `data/raw/`, so reproduction re-runs the model **calls** on fixed inputs;
full request+response transcripts are written to `data/transcripts/<run>.jsonl`. (`.env`,
`.venv/`, and `*.log` are gitignored; `data/` is intentionally committed.)

_Exploratory work; see `findings.md` for caveats and uncertainties._
