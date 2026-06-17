"""Anthropic API layer for the rule-articulation experiments.

Three roles, all on claude-opus-4-8:
  - classify():  NO reasoning (thinking disabled), single-token label. The model
                 under study. Few-shot prefix is prompt-cached.
  - generate():  thinking=adaptive/medium. Produces natural example inputs.
  - articulate(): thinking=adaptive/medium. Free-form statement of the rule.
"""
import os
import re
import concurrent.futures as cf
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-8"

# --- load .env (no dependency on python-dotenv) ---
def _load_env():
    env = Path(__file__).with_name(".env")
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()
_client = anthropic.Anthropic(max_retries=8)

# --- full-transcript logging: every API call's request + response -> data/transcripts/<run>.jsonl ---
import json as _json, time as _time, threading as _threading
_RUN = {"name": "misc"}
_tls = _threading.local()
_LOGDIR = Path(__file__).with_name("data") / "transcripts"
_LOGDIR.mkdir(parents=True, exist_ok=True)
_log_lock = _threading.Lock()

def set_run(name): _RUN["name"] = name
def set_tag(**kw): _tls.tag = kw
def clear_tag(): _tls.tag = {}

def _create(**kw):
    """Single chokepoint: call the API, append the full request+response to the run's JSONL."""
    r = _client.messages.create(**kw)
    try:
        rec = {
            "ts": _time.time(), "run": _RUN["name"], "tag": getattr(_tls, "tag", {}),
            "request": {k: kw.get(k) for k in
                        ("model", "system", "messages", "thinking", "output_config", "max_tokens")},
            "response": {"stop_reason": r.stop_reason,
                         "content": [{"type": b.type, "text": getattr(b, "text", None)} for b in r.content],
                         "usage": r.usage.model_dump() if hasattr(r.usage, "model_dump") else str(r.usage)},
        }
        with _log_lock, open(_LOGDIR / f"{_RUN['name']}.jsonl", "a") as f:
            f.write(_json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
    return r

CLASSIFY_SYSTEM = (
    "You are a binary classifier. Each input has a hidden label, True or False, "
    "determined by one fixed rule. Infer the rule from the labeled examples and "
    "apply it to the final input. Respond with exactly one word — True or False — "
    "and nothing else."
)

def _fewshot_text(shots):
    return "\n".join(f"Input: {t}\nLabel: {'True' if y else 'False'}" for t, y in shots)

def _parse_label(text):
    m = re.search(r"\b(true|false)\b", text.strip().lower())
    return None if not m else (m.group(1) == "true")

def classify_one(shots, query):
    """Classify a single query string given few-shot (text,label) pairs. No CoT."""
    prefix = _fewshot_text(shots)
    resp = _create(
        model=MODEL,
        max_tokens=5,
        thinking={"type": "disabled"},
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prefix + "\n\n", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Input: {query}\nLabel:"},
        ]}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _parse_label(text)

def classify_many(shots, queries, workers=8):
    """Classify many queries against the same few-shot prefix, in parallel."""
    out = [None] * len(queries)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(classify_one, shots, q): i for i, q in enumerate(queries)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out

TOLD_SYSTEM = (
    "You are a binary classifier. Apply the rule below exactly to the input. "
    "Respond with exactly one word — True or False — and nothing else."
)

def classify_told_one(rule_text, query):
    """Classify with the rule stated explicitly (tests execution, not induction). No CoT."""
    resp = _create(
        model=MODEL,
        max_tokens=5,
        thinking={"type": "disabled"},
        system=TOLD_SYSTEM,
        messages=[{"role": "user", "content": f"Rule: {rule_text}\n\nInput: {query}\nLabel:"}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _parse_label(text)

def classify_told_many(rule_text, queries, workers=8):
    out = [None] * len(queries)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(classify_told_one, rule_text, q): i for i, q in enumerate(queries)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out

def chat_nothink(messages, system=None, max_tokens=400):
    """Multi-turn call with thinking fully disabled. Returns (text, response)."""
    kw = dict(model=MODEL, max_tokens=max_tokens, thinking={"type": "disabled"}, messages=messages)
    if system:
        kw["system"] = system
    resp = _create(**kw)
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["A", "B", "AND", "OR", "NEITHER", "OTHER"]},
        "interpretation": {"type": "string"},
    },
    "required": ["category", "interpretation"],
    "additionalProperties": False,
}

def judge_response(text, a_def, b_def, effort="low"):
    """Categorize a free-form rule description. Returns {category, interpretation, original}.
      A/B      - names only A's feature / only B's feature
      AND      - conjunction (True only if BOTH A and B hold)
      OR       - disjunction (True if EITHER A or B holds)
      NEITHER  - a clear, coherent rule that is neither A nor B
      OTHER    - catchall: no rule, refusal, meta-comment, confusion, off-topic, etc."""
    q = (f"Rule A: {a_def}\nRule B: {b_def}\n\n"
         f"Someone was asked what classification rule they had in mind. They said:\n\"{text}\"\n\n"
         "Categorize their statement:\n"
         "  A       = names only A's feature\n"
         "  B       = names only B's feature\n"
         "  AND     = a conjunction: labels True only when BOTH A's and B's features hold\n"
         "  OR      = a disjunction: labels True when EITHER A's or B's feature holds\n"
         "  NEITHER = a clear, coherent rule that is neither A nor B\n"
         "  OTHER   = anything else (states no rule, refuses, asks a question, hedges without "
         "committing, is confused/off-topic, or otherwise doesn't fit)\n"
         "If they mention both features but it's unclear whether it's AND or OR, prefer AND. "
         "Give 'interpretation' as a short phrase describing what they actually said.")
    resp = _create(
        model=MODEL, max_tokens=1500, thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": q}],
    )
    import json as _json
    raw = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        d = _json.loads(raw)
        cat = d.get("category", "OTHER")
        interp = d.get("interpretation", "")
    except Exception:
        cat, interp = "OTHER", "judge parse failure: " + raw[:120]
    return {"category": cat, "interpretation": interp, "original": text}

_ART_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "interpretation": {"type": "string"},
    },
    "required": ["correct", "interpretation"],
    "additionalProperties": False,
}

def judge_articulation(text, rule_articulation, effort="low"):
    """Does a free-form rule statement correctly capture the target rule (same
    True/False partition)? Returns {correct, interpretation, original}."""
    q = (f"Target rule: {rule_articulation}\n\n"
         f"Someone was asked to state the single classification rule they inferred from "
         f"labeled examples. They said:\n\"{text}\"\n\n"
         "Does their statement correctly capture the target rule — i.e. would it sort inputs "
         "into the same True/False partition? Minor wording differences are fine; what matters "
         "is that the criterion is the same. Set 'correct' = true if it matches, false otherwise. "
         "Give 'interpretation' as a short phrase describing what they actually said.")
    resp = _create(
        model=MODEL, max_tokens=1500, thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": _ART_JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": q}],
    )
    import json as _json
    raw = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        d = _json.loads(raw)
        return {"correct": bool(d.get("correct", False)),
                "interpretation": d.get("interpretation", ""), "original": text}
    except Exception:
        return {"correct": False, "interpretation": "judge parse failure: " + raw[:120], "original": text}

def chat_structured(messages, schema, system=None, max_tokens=300):
    """Multi-turn call constrained to a JSON schema, thinking disabled. Returns parsed dict.
    Used wherever a free token stream could leak hints into the conversation."""
    import json as _json
    kw = dict(model=MODEL, max_tokens=max_tokens, thinking={"type": "disabled"},
              output_config={"format": {"type": "json_schema", "schema": schema}}, messages=messages)
    if system:
        kw["system"] = system
    resp = _create(**kw)
    raw = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        return _json.loads(raw), raw
    except Exception:
        return {}, raw

def judge_satisfies(sentence, rule_articulation):
    """Judge whether a sentence satisfies a (semantic) rule. Returns bool. No reasoning."""
    q = (f"Rule: {rule_articulation}\nSentence: \"{sentence}\"\n"
         "Does the sentence satisfy the rule? Answer with only 'yes' or 'no'.")
    out, _ = chat_nothink([{"role": "user", "content": q}], max_tokens=5)
    return out.strip().lower().startswith("y")

def generate(prompt, effort="medium", max_tokens=4000):
    """A thinking call (adaptive/medium) returning the text response."""
    resp = _create(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")

def articulate(shots, instruction, effort="medium"):
    """Show the model the labeled examples and ask it to state the rule (free-form, CoT ok)."""
    prefix = _fewshot_text(shots)
    return generate(f"{prefix}\n\n{instruction}", effort=effort)
