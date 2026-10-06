"""Self-contained static report for viewing off the GPU box (no server, no tunnel):

    python -m datagen report      # -> data/reports.tar.gz
    scp <host>:<repo>/data/reports.tar.gz . && tar xzf reports.tar.gz && open reports/index.html

Contains the stats plots and tables, the render-audit contact sheets, per-generator galleries
(each item with ground-truth points + masks, its pair partner, question, answer and checks) and
the CI / build / blind-check / length reports.
"""

import html
import json
import tarfile
from collections import defaultdict

import numpy as np

from .checks import audit_sheets, load_pools
from .common import DATA, REPORTS
from .stats import GRID, INK, INK2, SURFACE, run_stats

PER_GENERATOR = 30


def galleries():
    from .viewer import overlay
    out = REPORTS / "gallery"
    (out / "img").mkdir(parents=True, exist_ok=True)
    pools = load_pools()
    by_id = {r["id"]: r for recs in pools.values() for r in recs}
    partner = defaultdict(list)
    for r in by_id.values():
        if r["pair_id"]:
            partner[r["pair_id"]].append(r["id"])
    by_gen = defaultdict(list)
    for r in by_id.values():
        if r["role"] in ("canonical", "original", "neutral"):  # one entry per pair; partner shown beside it
            by_gen[r["render"]["generator"]].append(r)
    pages = []
    from tqdm import tqdm
    for gen, recs in tqdm(sorted(by_gen.items()), desc="galleries", unit="page"):
        rng = np.random.default_rng(0)
        pick = [recs[i] for i in sorted(rng.choice(len(recs), size=min(PER_GENERATOR, len(recs)), replace=False))]
        rows = []
        for r in pick:
            cells = []
            for rec in [r] + [by_id[i] for i in partner.get(r["pair_id"], []) if i != r["id"]]:
                name = f"{rec['id']}.jpg"
                overlay(rec).resize((336, 336)).save(out / "img" / name, quality=85)
                cells.append(f"<figure><img src='img/{name}' width=336><figcaption><b>{html.escape(rec['role'])}</b> · "
                             f"answer {html.escape(str(rec['answer']))} · familiar {html.escape(str(rec['familiar_answer']))}"
                             f" · Δ {rec['delta']}<br>recount {rec['checks']['independent_recount']} · color "
                             f"{rec['checks']['color_name']}<br><small>{html.escape(rec['id'])}</small></figcaption></figure>")
            rows.append(f"<div class=item><p>{html.escape(r['question'])}</p><div class=row>{''.join(cells)}</div></div>")
        page = f"{gen}.html"
        (out / page).write_text(_html(f"{gen} — {len(pick)} of {len(recs)} items (points: red, masks: tinted)", "".join(rows)))
        pages.append((gen, f"gallery/{page}", len(recs)))
    return pages


def _html(title, body):
    return (f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
            f"body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};margin:24px}}"
            f".item{{border-bottom:1px solid {GRID};padding:8px 0}}.row{{display:flex;gap:12px;flex-wrap:wrap}}"
            f"figure{{margin:0}}figcaption{{font-size:12px;color:{INK2};max-width:336px}}a{{color:#2a78d6}}</style></head>"
            f"<body><h1>{html.escape(title)}</h1>{body}</body></html>")


def build_report():
    run_stats()
    audit_sheets()
    pages = galleries()
    links = "".join(f"<li><a href='{p}'>{html.escape(g)}</a> ({n:,} items)</li>" for g, p, n in pages)
    sheets = "".join(f"<li><a href='audit/{p.name}'>{p.stem}</a></li>" for p in sorted((REPORTS / "audit").glob("*.jpg")))
    reports = "".join(f"<li><a href='{p.name}'>{p.name}</a></li>" for p in sorted(REPORTS.glob("*.json")))
    ci = REPORTS / "ci_report.json"
    ci_rows = ""
    if ci.exists():
        ci_rows = "".join(f"<li>{'PASS' if v['ok'] else '<b>FAIL</b>'} — {html.escape(k)} <small>{html.escape(str(v.get('summary', '')))}</small></li>"
                          for k, v in json.loads(ci.read_text()).items())
    body = (f"<h2><a href='stats/index.html'>Statistics and plots</a></h2><h2>CI checks</h2><ul>{ci_rows or '<li>not run</li>'}</ul>"
            f"<h2>Galleries (item + pair partner, with overlays)</h2><ul>{links}</ul>"
            f"<h2>Render-audit contact sheets (50 per generator)</h2><ul>{sheets}</ul><h2>Raw reports</h2><ul>{reports}</ul>")
    (REPORTS / "index.html").write_text(_html("Generated data report", body))
    tar = DATA / "reports.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(REPORTS, arcname="reports")
    print(f"Report: {REPORTS / 'index.html'}\nBundle: {tar} ({tar.stat().st_size / 1e6:.1f} MB) — "
          f"scp it to your machine, untar, open reports/index.html")
