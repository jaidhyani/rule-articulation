#!/usr/bin/env bash
# Reproduce the rule-articulation study from a clean clone.
#
# Requires an Anthropic API key (model under study: claude-opus-4-8).
# Provide it either by exporting ANTHROPIC_API_KEY, or by creating a .env file
# next to this script containing:  ANTHROPIC_API_KEY=sk-ant-...
#
# Usage:
#   ./reproduce.sh                 # print this help + stage list + cost estimates (runs nothing)
#   ./reproduce.sh <stage> [...]   # run specific stages
#   ./reproduce.sh all             # run every stage in order
#
# Datasets are committed under data/raw/ and are reused as-is (the model CALLS are
# what get reproduced), so results should match up to API non-determinism. Full
# request/response transcripts are written to data/transcripts/<run>.jsonl.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${ANTHROPIC_API_KEY:-}" && ! -f .env ]]; then
  echo "ERROR: no API key. Export ANTHROPIC_API_KEY, or put it in a .env file next to this script." >&2
  exit 1
fi

ALL=(boundary pairs search execution articulate lowercase suggest)

usage() {
  cat <<'EOF'
Stages (pass one or more, or `all`):

  boundary     Exp1 — per-rule held-out classification accuracy, no-CoT      (~450 calls, ~2 min)
  pairs        Exp2 — two-rule correlated-decoy articulation                  (~250 calls, ~3 min)
  search       Exp3 — synthetic-string difficulty border search              (~adaptive, ~3 min)
  execution    Exp4 — induction vs execution limits (ICL vs told-the-rule)   (~250 calls, ~3 min)
  articulate   single-rule articulation, no-CoT, judged                       (~100 calls, ~2 min)
  lowercase    content-controlled "entirely lowercase" set + classification   (~400 calls, ~3 min)
  suggest      Exp5 — commit-then-reveal (CENTERPIECE)                         (~2200 calls, ~10-15 min)

Examples:
  ./reproduce.sh boundary articulate
  ./reproduce.sh all

Note: `lowercase` regenerates its dataset fresh (its content-controlled construction is not
seeded), so its exact sentences differ run-to-run; all other stages reuse committed datasets.
EOF
}

PY=.venv/bin/python
ensure_venv() {
  if [[ ! -x "$PY" ]]; then
    echo ">> creating .venv and installing dependencies"
    python3 -m venv .venv
    .venv/bin/pip install -q -U pip
    .venv/bin/pip install -q -r requirements.txt
  fi
}

run_stage() {
  case "$1" in
    boundary)   echo "== Exp1 boundary";                       $PY exp_boundary.py   2>&1 | tee boundary.log ;;
    pairs)      echo "== Exp2 pairs (decoy)";                  $PY exp_pairs.py      2>&1 | tee pairs.log ;;
    search)     echo "== Exp3 border search";                 $PY search.py         2>&1 | tee search.log ;;
    execution)  echo "== Exp4 execution vs induction";        $PY exp_execution.py  2>&1 | tee execution.log ;;
    articulate) echo "== single-rule articulation";           $PY exp_articulate.py 2>&1 | tee articulate.log ;;
    lowercase)  echo "== content-controlled lowercase";       $PY redo_lowercase3.py 2>&1 | tee lowercase.log ;;
    suggest)    echo "== Exp5 commit-then-reveal (~2200 Opus calls)"; $PY exp_suggest.py 2>&1 | tee suggest_final.log ;;
    *)          echo "unknown stage: $1" >&2; usage; exit 1 ;;
  esac
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

STAGES=("$@")
[[ "${1:-}" == "all" ]] && STAGES=("${ALL[@]}")

ensure_venv
for s in "${STAGES[@]}"; do
  run_stage "$s"
done
echo ">> done. Results: *_results.json + *.log ; full transcripts: data/transcripts/*.jsonl"
