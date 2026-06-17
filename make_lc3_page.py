"""Diagnostic page: round-3 lowercase classification errors (long, multi-proper-noun
sentences; natural capitalization on negatives). Aggregates per-query error rate across
the 5 shot draws from the transcript."""
import collections
import html
import json

from llm import _parse_label

check = lambda s: any(c.isalpha() for c in s) and s == s.lower()
recs = [json.loads(l) for l in open("data/transcripts/lowercase_redo3.jsonl")]

agg = collections.defaultdict(lambda: {"wrong": 0, "total": 0, "truth": None})
for r in recs:
    c = r["request"]["messages"][0].get("content")
    if not (isinstance(c, list) and len(c) == 2):
        continue
    q = c[1]["text"]
    if not q.startswith("Input: "):
        continue
    query = q[len("Input: "):].rsplit("\nLabel:", 1)[0]
    rtext = "".join(b.get("text", "") for b in r["response"].get("content", []) if b.get("type") == "text")
    pred = _parse_label(rtext)
    truth = check(query)
    a = agg[query]
    a["truth"] = truth
    a["total"] += 1
    if pred != truth:
        a["wrong"] += 1

n_queries = len(agg)
n_calls = sum(v["total"] for v in agg.values())
n_wrong = sum(v["wrong"] for v in agg.values())
wrongs = sorted([(q, v) for q, v in agg.items() if v["wrong"] > 0],
                key=lambda kv: -kv[1]["wrong"] / kv[1]["total"])
# split by direction
miss_cap = [(q, v) for q, v in wrongs if not v["truth"]]   # has-capital called lowercase
false_alarm = [(q, v) for q, v in wrongs if v["truth"]]     # lowercase called has-capital


def hl(s):
    return "".join(f'<span class="cap">{html.escape(ch)}</span>' if ch.isupper() else html.escape(ch) for ch in s)


def block(rows):
    out = []
    for q, v in rows:
        rate = f'{v["wrong"]}/{v["total"]}'
        ncaps = sum(1 for ch in q if ch.isupper())
        out.append(
            f'<div class="item"><div class="meta"><span class="rate">{rate} wrong</span>'
            f'<span class="caps">{ncaps} real capital{"s" if ncaps != 1 else ""}</span></div>'
            f'<div class="q">{hl(q)}</div></div>')
    return "\n".join(out)


doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lowercase — what Opus 4.8 gets wrong</title>
<style>
:root{{--paper:#f6f1e7;--panel:#fffdf8;--ink:#22201c;--muted:#6f675b;--line:#e2d9c8;
--terra:#c2613f;--teal:#2f7e7b;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.55}}
.wrap{{max-width:920px;margin:0 auto;padding:40px 24px 80px}}
.kicker{{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.08em;
text-transform:uppercase;color:var(--terra);margin-bottom:10px}}
h1{{font-size:29px;margin:0 0 6px}}
.dek{{color:var(--muted);font-size:16.5px;margin:0 0 22px}}
.stats{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0 8px}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:130px}}
.stat .big{{font-size:28px;font-weight:700;color:var(--terra);line-height:1}}
.stat .lbl{{font-size:12.5px;color:var(--muted);margin-top:4px}}
h2{{font-size:18px;margin:32px 0 6px}}
.sub{{color:var(--muted);font-size:14.5px;margin:0 0 14px}}
.item{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:11px 14px;margin:9px 0}}
.item .meta{{display:flex;gap:12px;margin-bottom:6px}}
.rate{{font-family:ui-monospace,Menlo,monospace;font-size:12px;font-weight:700;color:var(--terra)}}
.caps{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted)}}
.item .q{{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;word-break:break-word}}
.cap{{background:var(--terra);color:#fff;border-radius:3px;padding:0 2px;font-weight:700}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--teal);
border-radius:8px;padding:14px 16px;margin:16px 0;font-size:15px}}
</style></head><body><div class="wrap">
<div class="kicker">rule-articulation · diagnostic</div>
<h1>"entirely lowercase" — what Opus 4.8 (no reasoning) gets wrong</h1>
<p class="dek">Round-3 set: long sentences (~23 words), each with multiple would-be-capitalized
words; negatives in natural form. 5 shot draws, k=16, single-token label. Red = a real
uppercase letter in the text.</p>

<div class="stats">
<div class="stat"><div class="big">{n_wrong}/{n_calls}</div><div class="lbl">classifications wrong ({n_wrong/n_calls:.1%})</div></div>
<div class="stat"><div class="big">{len(wrongs)}/{n_queries}</div><div class="lbl">distinct sentences wrong &ge;1&times;</div></div>
<div class="stat"><div class="big">{len(miss_cap)}</div><div class="lbl">capital-bearing sentences called "lowercase"</div></div>
<div class="stat"><div class="big">{len(false_alarm)}</div><div class="lbl">all-lowercase sentences called "has-capital"</div></div>
</div>

<div class="note"><b>The model isn't reading the glyphs — it's judging by content.</b> On long,
proper-noun-dense sentences it (a) <b>overlooks real capital letters</b> that are right there,
and (b) <b>false-alarms on all-lowercase sentences</b> packed with names/places it expects to be
capitalized. Both are consistent with pattern-matching on register, not a character-level scan.</div>

<h2>① Misses capitals that are right there</h2>
<p class="sub">Truth = has-capital (red letters present), but the model said "entirely lowercase."</p>
{block(miss_cap)}

<h2>② False-alarms on genuinely all-lowercase text</h2>
<p class="sub">Truth = entirely lowercase (zero capitals), but the model said "has a capital" —
fooled by lowercased proper nouns.</p>
{block(false_alarm)}
</div></body></html>"""

open("lowercase_errors.html", "w").write(doc)
print(f"wrote lowercase_errors.html | {n_wrong}/{n_calls} wrong | miss-capital={len(miss_cap)} | false-alarm={len(false_alarm)}")
