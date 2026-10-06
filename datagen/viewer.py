"""Browser app for looking at generated data and verifying it by eye.

    python -m datagen view [port]   # listens on localhost only

On the hosted GPU box, tunnel the port over SSH and open it in your local browser:
    ssh -L 7860:localhost:7860 <user>@<gpu-host>      then  http://localhost:7860
(VS Code Remote-SSH forwards it automatically.) For viewing without a server, see `python -m datagen report`.

Tabs:
  Browse  filter by pool / family / role, step through items; overlays for ground-truth points,
          unit masks and the object box; the other image of a pair side by side with the pixel
          difference of the edit (before compression); full record. "Flag" logs a problem to
          data/reports/review_flags.jsonl.
  Audit   blind human audit (§10.3): the question without the answer; click every counted unit
          (or pick the color); the derived answer is compared with the label and logged to
          data/reports/audit_log.jsonl, with running accuracy against the 97% bar.
  Stats   the plots from `python -m datagen stats`.
"""

import json
import random
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from .checks import load_pools
from .common import DATA, REPORTS, rle_decode

COLORS = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]
MASK_TINTS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (74, 58, 167)]


def overlay(rec, show_points=True, show_masks=True, show_bbox=True, clicks=()):
    im = Image.open(DATA / rec["image"]).convert("RGB")
    arr = np.asarray(im).astype(float)
    if show_masks and rec["masks_rle"]:
        for k, m in enumerate(rec["masks_rle"]):
            mk = rle_decode(m)
            arr[mk] = arr[mk] * 0.55 + np.array(MASK_TINTS[k % len(MASK_TINTS)]) * 0.45
    im = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(im)
    if show_bbox and rec["object_bbox"]:
        d.rectangle(rec["object_bbox"], outline=(255, 255, 255), width=1)
    if show_points:
        for k, (x, y) in enumerate(rec["points"] or []):
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 0), width=2)
            if len(rec["points"]) <= 60:
                d.text((x + 6, y - 6), str(k + 1), fill=(255, 0, 0))
    for x, y in clicks:
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 255, 255), width=3)
    return im.resize((672, 672), Image.NEAREST)


def edit_diff(a, b):
    from .items import render_clean
    ia, ib = np.asarray(render_clean(a)[0], int), np.asarray(render_clean(b)[0], int)
    diff = np.abs(ia - ib).max(axis=2) > 0
    base = (ia * 0.35).astype(np.uint8)
    base[diff] = (255, 64, 64)
    return Image.fromarray(base).resize((448, 448), Image.NEAREST), int(diff.sum())


def describe(rec):
    lines = [f"**{rec['id']}**", f"pool item: `{rec['dataset']}` · family `{rec['family']}` · {rec['data_type']} / {rec['role']}",
             f"**Q:** {rec['question']}", f"**answer:** {rec['answer']} · familiar: {rec['familiar_answer']} · Δ: {rec['delta']}",
             f"edit: `{rec['edit']['op']}` {json.dumps(rec['edit']['params'])} · null edit: {rec['edit']['null_edit']}",
             f"checks: recount {rec['checks']['independent_recount']} · color name {rec['checks']['color_name']}",
             f"unit size {rec['unit_size_px']} px · background {rec['render']['background']['kind']} · "
             f"distractors {len(rec['render']['distractors'])} · jpeg q{rec['compression']['jpeg_quality']} "
             f"· downscale {rec['compression']['downscale']}"]
    return "\n\n".join(lines)


def build_app():
    pools = load_pools()
    if not pools:
        raise SystemExit("No pools built yet: run `python -m datagen build`.")
    by_id = {r["id"]: r for recs in pools.values() for r in recs}
    pairs = {}
    for r in by_id.values():
        if r["pair_id"]:
            pairs.setdefault(r["pair_id"], []).append(r["id"])

    def filtered(pool, family, role):
        return [r for r in pools[pool] if (family == "all" or r["family"] == family) and (role == "all" or r["role"] == role)]

    def families(pool):
        return ["all"] + sorted({r["family"] for r in pools[pool]})

    def roles(pool):
        return ["all"] + sorted({r["role"] for r in pools[pool]})

    def show(pool, family, role, idx, pts, masks, bbox):
        items = filtered(pool, family, role)
        if not items:
            return None, None, None, "no items", "{}", gr.update(maximum=0, value=0), ""
        idx = int(min(max(idx, 0), len(items) - 1))
        rec = items[idx]
        partner, diff_img, diff_note = None, None, ""
        if rec["pair_id"]:
            other = [by_id[i] for i in pairs[rec["pair_id"]] if i != rec["id"]]
            if other:
                partner = overlay(other[0], pts, masks, bbox).resize((448, 448))
                a, b = sorted([rec, other[0]], key=lambda r: r["role"] not in ("canonical", "original"))
                diff_img, n = edit_diff(a, b)
                diff_note = f"pair partner: {other[0]['role']} (answer {other[0]['answer']}) · {n} px changed by the edit"
        light = {k: v for k, v in rec.items() if k != "masks_rle"}
        return (overlay(rec, pts, masks, bbox), partner, diff_img, describe(rec) + "\n\n" + diff_note,
                json.dumps(light, indent=1), gr.update(maximum=len(items) - 1, value=idx), f"{idx + 1} / {len(items)}")

    def flag(pool, family, role, idx, note):
        items = filtered(pool, family, role)
        if not items:
            return "nothing to flag"
        rec = items[int(idx)]
        REPORTS.mkdir(parents=True, exist_ok=True)
        with (REPORTS / "review_flags.jsonl").open("a") as f:
            f.write(json.dumps({"id": rec["id"], "note": note, "time": time.time()}) + "\n")
        return f"flagged {rec['id']}"

    # ---- audit ----
    def audit_start(pool, n):
        items = pools[pool][:]
        random.Random(f"audit-{pool}-{time.time()}").shuffle(items)
        queue = [r["id"] for r in items[: int(n)]]
        return audit_render({"queue": queue, "pos": 0, "clicks": [], "results": [], "pool": pool})

    def audit_render(st):
        if not st or st["pos"] >= len(st["queue"]):
            done = st["results"] if st else []
            acc = 100 * sum(r["correct"] for r in done) / len(done) if done else 0
            return st, None, f"Audit finished: {len(done)} items, label accuracy {acc:.1f}% (pass ≥ 97%)", gr.update(), ""
        rec = by_id[st["queue"][st["pos"]]]
        img = overlay(rec, False, False, False, clicks=st["clicks"])
        done = st["results"]
        acc = f"{100 * sum(r['correct'] for r in done) / len(done):.1f}%" if done else "—"
        info = f"item {st['pos'] + 1} / {len(st['queue'])} · running label accuracy {acc} (pass ≥ 97%)\n\n**{rec['question']}**"
        if rec["prior"] == "count":
            info += f"\n\nClick every counted unit. Clicked: {len(st['clicks'])}"
        return st, img, info, gr.update(visible=rec["prior"] == "color", value=None), ""

    def audit_click(st, evt: gr.SelectData):
        if st and st["pos"] < len(st["queue"]):
            x, y = evt.index
            st["clicks"].append((x * 448 / 672, y * 448 / 672))
        return audit_render(st)

    def audit_undo(st):
        if st and st["clicks"]:
            st["clicks"].pop()
        return audit_render(st)

    def audit_submit(st, color):
        if not st or st["pos"] >= len(st["queue"]):
            return audit_render(st)
        rec = by_id[st["queue"][st["pos"]]]
        if rec["prior"] == "count":
            given = len(st["clicks"])
            # Each click should land on a distinct unit mask.
            masks = [rle_decode(m) for m in rec["masks_rle"]]
            hit = {k for x, y in st["clicks"] for k, m in enumerate(masks) if m[min(447, int(y)), min(447, int(x))]}
            correct = given == rec["answer"]
            extra = {"clicks": st["clicks"], "units_hit": len(hit)}
        else:
            given, correct, extra = color, color == rec["answer"], {}
        result = {"id": rec["id"], "pool": st["pool"], "label": rec["answer"], "given": given, "correct": correct,
                  "time": time.time(), **extra}
        REPORTS.mkdir(parents=True, exist_ok=True)
        with (REPORTS / "audit_log.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        st["results"].append(result)
        st["pos"] += 1
        st["clicks"] = []
        out = audit_render(st)
        note = "" if correct else f"previous item: you gave {given}, label is {rec['answer']} ({rec['id']})"
        return out[:4] + (note,)

    def stats_images():
        d = REPORTS / "stats"
        if not (d / "index.html").exists():
            from .stats import run_stats
            run_stats()
        return [str(p) for p in sorted(d.glob("*.png"))]

    first = list(pools)[0]
    with gr.Blocks(title="counterpoint data viewer") as app:
        with gr.Tab("Browse"):
            with gr.Row():
                pool = gr.Dropdown(list(pools), value=first, label="pool")
                family = gr.Dropdown(families(first), value="all", label="family")
                role = gr.Dropdown(roles(first), value="all", label="role")
                pts, masks, bbox = gr.Checkbox(True, label="points"), gr.Checkbox(True, label="masks"), gr.Checkbox(False, label="object box")
            with gr.Row():
                prev_b, next_b, rand_b = gr.Button("◀ prev"), gr.Button("next ▶"), gr.Button("random")
                idx = gr.Slider(0, len(pools[first]) - 1, value=0, step=1, label="index")
                pos = gr.Markdown()
            with gr.Row():
                main_img = gr.Image(label="item (overlays)", type="pil", height=672)
                with gr.Column():
                    partner_img = gr.Image(label="pair partner", type="pil", height=448)
                    diff_img = gr.Image(label="edit difference, before compression (red = changed)", type="pil", height=448)
            info = gr.Markdown()
            with gr.Row():
                note = gr.Textbox(label="problem note", scale=4)
                flag_b = gr.Button("Flag item", scale=1)
                flag_out = gr.Markdown()
            with gr.Accordion("record JSON", open=False):
                rec_json = gr.Code(language="json")
            inputs = [pool, family, role, idx, pts, masks, bbox]
            outputs = [main_img, partner_img, diff_img, info, rec_json, idx, pos]
            pool.change(lambda p: (gr.update(choices=families(p), value="all"), gr.update(choices=roles(p), value="all"), 0),
                        pool, [family, role, idx]).then(show, inputs, outputs)
            for c in (family, role):
                c.change(lambda: 0, None, idx).then(show, inputs, outputs)
            for c in (idx, pts, masks, bbox):
                c.change(show, inputs, outputs)
            prev_b.click(lambda i: max(0, i - 1), idx, idx)
            next_b.click(lambda i: i + 1, idx, idx)
            rand_b.click(lambda p, f, r: random.randrange(max(1, len(filtered(p, f, r)))), [pool, family, role], idx)
            flag_b.click(flag, [pool, family, role, idx, note], flag_out)
            app.load(show, inputs, outputs)

        with gr.Tab("Audit"):
            state = gr.State(None)
            with gr.Row():
                a_pool = gr.Dropdown(list(pools), value=first, label="pool to audit")
                a_n = gr.Number(100, label="items", precision=0)
                a_start = gr.Button("Start audit")
            with gr.Row():
                a_img = gr.Image(label="click each unit", type="pil", height=672, interactive=False)
                with gr.Column():
                    a_info = gr.Markdown()
                    a_color = gr.Radio(COLORS, label="color", visible=False)
                    with gr.Row():
                        a_undo, a_submit = gr.Button("Undo click"), gr.Button("Submit", variant="primary")
                    a_note = gr.Markdown()
            outs = [state, a_img, a_info, a_color, a_note]
            a_start.click(audit_start, [a_pool, a_n], outs)
            a_img.select(audit_click, state, outs)
            a_undo.click(audit_undo, state, outs)
            a_submit.click(audit_submit, [state, a_color], outs)

        with gr.Tab("Stats"):
            gallery = gr.Gallery(columns=1, height="auto", label="python -m datagen stats")
            refresh = gr.Button("Regenerate stats")
            refresh.click(lambda: (__import__("datagen.stats", fromlist=["run_stats"]).run_stats(), stats_images())[1],
                          None, gallery)
            app.load(stats_images, None, gallery)
    return app


def view(port=7860):
    import socket
    print(f"Viewer on 127.0.0.1:{port}. From your laptop:\n"
          f"    ssh -N -L {port}:localhost:{port} <user>@{socket.gethostname()}\n"
          f"then open http://localhost:{port}")
    # Localhost only: the data must not be exposed on the hosted machine's public interface.
    build_app().launch(server_name="127.0.0.1", server_port=port, allowed_paths=[str(DATA)], share=False)
