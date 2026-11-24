"""Tiny fake data directory for smoke-testing analysis tools on a laptop (not for results).

    python tools/dev/make_smoke_data.py ~/Documents/research/smoke_data

Pools with the same record fields the tools read: count pools (circles to count; conflict pools have canonical /
counterfactual roles and a familiar answer) and a color pool. Images are 448x448 PNGs."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path(sys.argv[1]).expanduser()
rng = np.random.default_rng(0)
COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 70), "yellow": (230, 210, 40)}


def circles(n, path):
    im = Image.new("RGB", (448, 448), (235, 235, 230)); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rng.integers(40, 408, 2); d.ellipse([x - 22, y - 22, x + 22, y + 22], fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True); im.save(path)


def pool(name, n, conflict=False, color=False, roles=("canonical", "counterfactual")):
    recs = []
    for i in range(n):
        if color:
            c = list(COLORS)[i % len(COLORS)]
            img = f"{name}/images/{i:04d}.png"; p = root / img; p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (448, 448), COLORS[c]).save(p)
            recs.append({"id": f"{name}-{i}", "image": img, "prior": "color", "answer": c, "familiar_answer": "white",
                         "question": "What color is the square? Answer with one color word in curly brackets, e.g., {red}.", "role": "counterfactual"})
            continue
        fam = 4 if conflict else None
        for role in (roles if conflict else ("neutral",)):
            k = (fam if role == "canonical" else int(rng.choice([3, 5]))) if conflict else int(rng.integers(1, 10))
            img = f"{name}/images/{i:04d}-{role}.png"; circles(k, root / img)
            recs.append({"id": f"{name}-{i}-{role}", "image": img, "prior": "count", "answer": k, "familiar_answer": fam, "role": role,
                         "dataset": name.replace("/", "_"), "question": "How many circles are there? Answer with a number in curly brackets, e.g., {9}."})
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


for name, n, conflict in (("parts/conflict_v2", 30, True), ("parts/count", 30, False), ("e1/conflict", 30, True), ("e1/neutral", 30, False),
                          ("e1/val_conflict", 15, True), ("tests/P/conflict", 15, True), ("tests/P/count", 15, False),
                          ("tests/T0/count", 15, True), ("tests/R/count", 15, False)):
    pool(name, n, conflict)
pool("tests/T2/color", 12, color=True)
(root / "params.json").write_text("{}")
print("wrote", root)
