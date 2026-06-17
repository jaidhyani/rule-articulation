"""Build a self-contained HTML diagnostic for the seed2 (56%) lowercase shot draw.
Recovers shots/queries/predictions from the logged transcript (calls are untagged because
set_tag is thread-local and didn't reach the ThreadPool workers; we match by shot-prefix)."""
import html
import json
import random

from llm import _fewshot_text, _parse_label

check = lambda s: any(c.isalpha() for c in s) and s == s.lower()
d = json.load(open("data/raw/single_lowercase.json"))
pos, neg = d["pos"], d["neg"]


def shots_for(seed):
    rng = random.Random(seed)
    p, n = list(pos), list(neg)
    rng.shuffle(p); rng.shuffle(n)
    shots = [(s, True) for s in p[:8]] + [(s, False) for s in n[:8]]
    rng.shuffle(shots)
    return shots


SEED = 2
shots = shots_for(SEED)
prefix = _fewshot_text(shots) + "\n\n"

recs = [json.loads(l) for l in open("data/transcripts/lowercase_redo2.jsonl")]
items = []
for r in recs:
    c = r["request"]["messages"][0].get("content")
    if not (isinstance(c, list) and len(c) == 2):
        continue
    if c[0]["text"] != prefix:
        continue
    query = c[1]["text"][len("Input: "):].rsplit("\nLabel:", 1)[0]
    rtext = "".join(b.get("text", "") for b in r["response"].get("content", []) if b.get("type") == "text")
    pred = _parse_label(rtext)
    truth = check(query)
    items.append({"query": query, "pred": pred, "truth": truth, "correct": pred == truth})

n = len(items)
correct = sum(x["correct"] for x in items)
# confusion by truth class
tp = sum(1 for x in items if x["truth"] and x["pred"])          # truly lowercase, called lowercase
fn = sum(1 for x in items if x["truth"] and not x["pred"])      # truly lowercase, called has-capital
tn = sum(1 for x in items if not x["truth"] and not x["pred"])  # has-capital, called has-capital
fp = sum(1 for x in items if not x["truth"] and x["pred"])      # has-capital, called lowercase (MISSED capital)
n_pos = sum(1 for x in items if x["truth"])
n_neg = n - n_pos


def hl(s):
    """Escape, then wrap every uppercase letter in a red marker."""
    out = []
    for ch in s:
        e = html.escape(ch)
        out.append(f'<span class="cap">{e}</span>' if ch.isupper() else e)
    return "".join(out)


def lab(b):
    return "lowercase" if b else "has-capital"


shot_rows = "\n".join(
    f'<div class="shot {"t" if y else "f"}"><span class="tag">{ "True" if y else "False"}</span>'
    f'<span class="txt">{hl(t)}</span></div>' for t, y in shots)

item_rows = []
for i, x in enumerate(items, 1):
    cls = "ok" if x["correct"] else "bad"
    mark = "✓" if x["correct"] else "✗"
    item_rows.append(
        f'<tr class="{cls}"><td class="num">{i}</td><td class="q">{hl(x["query"])}</td>'
        f'<td class="c">{lab(x["truth"])}</td><td class="c">{lab(x["pred"])}</td>'
        f'<td class="m">{mark}</td></tr>')
item_rows = "\n".join(item_rows)

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lowercase — the 56% draw (seed2)</title>
<style>
:root{{--paper:#f6f1e7;--panel:#fffdf8;--ink:#22201c;--muted:#6f675b;--line:#e2d9c8;
--terra:#c2613f;--teal:#2f7e7b;--good:#2f7e7b;--bad:#c2613f;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.5}}
.wrap{{max-width:1000px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:30px;margin:0 0 6px}}
.dek{{color:var(--muted);font-size:17px;margin:0 0 24px}}
.kicker{{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.08em;
text-transform:uppercase;color:var(--terra);margin-bottom:10px}}
h2{{font-size:19px;margin:34px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.stats{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:120px}}
.stat .big{{font-size:30px;font-weight:700;color:var(--terra);line-height:1}}
.stat .lbl{{font-size:12.5px;color:var(--muted);margin-top:4px}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--teal);
border-radius:8px;padding:14px 16px;margin:14px 0;font-size:15px}}
.shots{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.shot{{display:flex;gap:10px;align-items:baseline;background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:8px 11px;font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.shot .tag{{font-weight:700;font-size:11px;min-width:42px}}
.shot.t .tag{{color:var(--teal)}} .shot.f .tag{{color:var(--terra)}}
.shot .txt{{word-break:break-word}}
table{{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-family:ui-monospace,Menlo,monospace;font-size:11px;text-transform:uppercase;
letter-spacing:.05em;color:var(--muted)}}
td.num{{color:var(--muted);font-variant-numeric:tabular-nums;width:32px}}
td.q{{font-family:ui-monospace,Menlo,monospace;font-size:13px;word-break:break-word}}
td.c{{white-space:nowrap;color:var(--muted);font-size:13px}}
td.m{{text-align:center;font-weight:700;width:36px}}
tr.bad{{background:#f7e7df}} tr.bad td.m{{color:var(--bad)}} tr.ok td.m{{color:var(--good)}}
tr.bad td.q{{font-weight:600}}
.cap{{background:var(--terra);color:#fff;border-radius:3px;padding:0 2px;font-weight:700}}
.legend{{font-size:13px;color:var(--muted);margin:6px 0 0}}
</style></head><body><div class="wrap">
<div class="kicker">rule-articulation · diagnostic</div>
<h1>"entirely lowercase" — the shot draw that collapsed to 56%</h1>
<p class="dek">Content-controlled lowercase set (case isolated from register/content). Nonthinking
Opus 4.8, k=16 few-shot, single token label. Seed-2 draw. Red = an actual uppercase letter.</p>

<div class="stats">
<div class="stat"><div class="big">{correct}/{n}</div><div class="lbl">overall correct ({correct/n:.0%})</div></div>
<div class="stat"><div class="big">{tp}/{n_pos}</div><div class="lbl">lowercase items called lowercase</div></div>
<div class="stat"><div class="big">{fp}/{n_neg}</div><div class="lbl">has-capital items it <b>missed</b> (called lowercase)</div></div>
<div class="stat"><div class="big">{tn}/{n_neg}</div><div class="lbl">has-capital items caught</div></div>
</div>

<div class="note"><b>Failure mode:</b> of the {n-correct} errors, <b>{fp}</b> are has-capital sentences
the model called "lowercase" (it <b>missed an internal capital</b>) and <b>{fn}</b> are lowercase
sentences it called "has-capital." When the rule is isolated to "is there <i>any</i> uppercase
letter," nonthinking single-pass classification mostly fails by <b>overlooking a single capital
letter</b> buried in a proper noun — exactly the kind of serial scan that benefits from reasoning.</div>

<h2>The 16 shots it learned from (seed 2)</h2>
<div class="shots">
{shot_rows}
</div>

<h2>All {n} held-out queries — prediction vs. truth</h2>
<p class="legend">Rows shaded when wrong. "truth" = does it contain any uppercase letter (red).</p>
<table><thead><tr><th>#</th><th>query</th><th>truth</th><th>predicted</th><th></th></tr></thead>
<tbody>
{item_rows}
</tbody></table>
</div></body></html>"""

open("lowercase_56_draw.html", "w").write(doc)
print(f"wrote lowercase_56_draw.html  ({correct}/{n} = {correct/n:.1%}; "
      f"missed-capital errors={fp}, false-neg={fn})")
