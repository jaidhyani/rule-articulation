# HANDOFF — rule-articulation study (for a fresh instance)

Personal work-test exploration for Truthful AI (Owain Evans). Question: can an LLM
articulate the classification rule it learns in-context, and are its self-reports faithful?
Model under study: **claude-opus-4-8, thinking DISABLED** (verified `thinking_tokens==0`).
Judge/generator: same model, thinking adaptive/low–medium.

Working dir: `/Users/jai/projects/rule-articulation` · venv: `.venv/bin/python` · key in `.env`.
Read `findings.md` for full results so far (Exp 1–5 + framing note + determinism endnote).

## NEW REQUIREMENT (just added, must hold for everything from here)
**Save full raw branching transcripts for every test, locally.** Implemented as a logging
chokepoint in `llm.py`: every API call's full request (model/system/messages/thinking/
output_config/max_tokens) + response (content/usage/stop_reason) is appended to
`data/transcripts/<run>.jsonl`. Each experiment must call `llm.set_run("<name>")` once, and
tag calls via `llm.set_tag(experiment=, pair=, probe=, cond=)` inside worker fns. Already
wired in `exp_suggest.py`; **the older scripts (exp_boundary/exp_pairs/exp_execution/search)
still need `llm.set_run(...)` added before re-running** — they otherwise log to `misc.jsonl`.
Also: `exp_suggest.py` now persists `raw_state` (all K=20 texts per condition) and `raw_sample`
into `suggest_results.json` — earlier runs did NOT keep raw free-form text (only category counts
+ OTHER/NEITHER raw), which is why the contrast question below needs a re-run.

## IMMEDIATE PLAN — run these now (in order)
1. **Re-run the suggestion experiment with logging** (the centerpiece, now instrumented):
   `cd /Users/jai/projects/rule-articulation && .venv/bin/python exp_suggest.py > suggest_final.log 2>&1 &`
   ~2200 calls, ~10–15 min, 6 pairs (datasets cached at n_div=12). Produces
   `data/transcripts/suggest_final.jsonl` (full transcripts) + `suggest_results.json` (now with raw_state).
2. **Answer the open question** (see below) from the saved `raw_state` / transcripts — do NOT
   re-sample ad hoc; read the saved data.
3. (Optional, for completeness of the raw record) add `llm.set_run("boundary"|"pairs"|"execution"|"search")`
   to those scripts and re-run to capture their transcripts too. Datasets are cached so they're fast.

## OPEN QUESTION blocking interpretation (Jai's, important)
In the "genuine contrast" neutral-reveal cases — **question×animal reveal-animal → claims
"question" (20/20)**; **question×alliteration reveal-alliteration → claims "question" (20/20)** —
is the model *actually changing its self-claim*, or is it *accurately describing the contrast
between its own belief and our claim* ("I had question-AND-animal in mind; you revealed animal,
so the part that's mine is question")? The judge coded these as "A/question" and can't tell the
two apart. **Read the raw STATE text** (from `raw_state["neutral_B"]` for those pairs, or the
transcripts) and decide. If it's faithful contrast-description, the "anti-conformity" framing in
the report + findings.md is WRONG and should be softened to "faithfully reports its belief and
notes the discrepancy" — this likely *strengthens* the honesty story. Update `findings.md` Exp5
and the report (`/Users/jai/projects/clai/claude-space/workshop/rule-articulation.html`, §4b) accordingly.

## EXPERIMENT DESIGN (current, locked)
6 pairs over {lowercase, has_digit, question, animal, alliteration}:
lowercase×animal, question×animal, lowercase×has_digit, has_digit×alliteration,
question×alliteration, lowercase×alliteration.
- In-context = correlated diagonal ONLY (TT=A∧B→True, FF=¬A∧¬B→False), so A/B/AND/OR reveals
  are all genuinely consistent. 12 shots/class. Datasets verified clean (programmatic + judge
  for semantic `animal`). Built at n_tt_ff=24, n_div=12 (10 samples/quad available).
- COMMIT (structured): "have you locked into one rule? knows=t/f" — K=5 for the rate; the branch
  prefix is resampled (≤10) to a knows=true realization (so the congratulatory reveal is congruent).
- BASELINE: STATE (free-form, K=20, judged A/B/AND/OR/NEITHER/OTHER) + SAMPLE (structured,
  10/quad×3 → operative rule via `exp_pairs.operative`).
- REVEALS, STATE only, K=20: {neutral, congratulatory} × {A,B,AND,OR}. Congrats only when prefix knows=true.
- Non-STATE turns use structured (JSON-schema) responses so no hints leak into the prefix.

## KEY RESULTS SO FAR (see findings.md for detail)
- Exp1–4: simple natural-text rules learn to ~100% no-CoT and are easily articulated (no
  classify-but-can't-articulate gap for learnable rules). Synthetic-string failures split into
  EXECUTION limits (counting/indexing, need CoT) vs INDUCTION limits (not findable in noise).
- Exp5 (suggestion): COMMIT always knows=5/5. Baseline mostly consistent + faithful. NEUTRAL
  reveal: robust / often names the other rule. CONGRATULATORY reveal: sharp conformity jump
  incl. adopting an "A or B" rule it doesn't actually use (0%→100% on some pairs); resistance
  scales with salience of the true rule. Three regimes: honest / pressured-dishonest / no-belief.

## FILE INVENTORY
- `llm.py` — API layer + logging chokepoint (`_create`, `set_run`, `set_tag`). All calls route through `_create`.
- `rules.py` — natural-text rule defs (label fn, articulation, generator prompts).
- `families.py` — parameterized synthetic-string families (knobbed) for the border search.
- `data_quad.py` — builds the 4 (A,B) quadrants; size-aware cache; semantic verify for `animal`.
- `data.py` — single-rule + simple two-rule dataset builders (Exp1/2).
- `exp_boundary.py` (Exp1) · `exp_execution.py` (Exp4) · `search.py` (Exp3 border search)
  · `exp_pairs.py` (Exp2 decoy + `operative()` helper, imported elsewhere) · `exp_suggest.py` (Exp5, FINAL).
- `findings.md` — running results + framing note + determinism endnote (Jai writes the actual report; do NOT AI-write the submission prose).
- `data/raw/quad_*.json`, `single_*.json`, `pair_*.json` — cached datasets. `data/transcripts/*.jsonl` — full transcripts.
- Report (internal, served on tailnet): `/Users/jai/projects/clai/claude-space/workshop/rule-articulation.html`
  → https://mini.jai.one/workshop/rule-articulation.html (Caddy serves the dir directly; edits are live, no rebuild).

## GOTCHAS
- Background `python` stdout is buffered; `print(..., flush=True)` or read the task .output file. exp_suggest prints "done: a x b" flushed.
- Don't trust noisy reads: 2/quad gave false operatives earlier; current design uses 10/quad×3.
- `git`: repo is `git init`'d, nothing committed yet. Jai hasn't asked to commit — ask before committing.
- Report is NOT the work-test submission; it's internal scaffolding. The submission prose must be Jai's own (assignment rule).
- Cost: each full exp_suggest run is ~2200 Opus calls. Don't re-run casually.
