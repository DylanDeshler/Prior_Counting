"""Update the status deck in place (2026-11-24): new strategy/dataset/diagnosis slides, steering round 2, and rebuilt
story / next / summary / appendix slides. Charts: tools/deck/charts_v3.py; diagnosis figures from reports/diagnose/.

    uv run --with python-pptx python tools/deck/update_v3.py
"""
import copy
import os
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

DECK = Path(os.path.expanduser("~/Documents/research/counterpoint_status_2026-11-10.pptx"))
A = Path(os.path.expanduser("~/Documents/research/deck_assets"))
DIAG = Path(os.path.expanduser("~/Downloads/reports/diagnose/part2-conflict-point-points-qwen3.5-s0-ckpt222-merged"))
INK, INK2, MUTED, ACC, LIGHT = (RGBColor(0x0b, 0x0b, 0x0b), RGBColor(0x52, 0x51, 0x4e), RGBColor(0x8a, 0x89, 0x84),
                                RGBColor(0x2a, 0x78, 0xd6), RGBColor(0xf3, 0xf2, 0xee))
ORANGE, GREEN = RGBColor(0xeb, 0x68, 0x34), RGBColor(0x1b, 0xaf, 0x7a)
FONT = "Arial"

prs = Presentation(DECK)
BLANK = prs.slide_layouts[6]


# ---- helpers (same look as the original builder) ----------------------------------------------------------------
def text(s, x, y, w, h, paras, size=16, color=INK, bold=False):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, p in enumerate(paras if isinstance(paras, list) else [paras]):
        p = {"t": p} if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = para.add_run(); r.text = ("•  " if p.get("bullet") else "") + p["t"]
        r.font.name = FONT; r.font.size = Pt(p.get("size", size)); r.font.bold = p.get("bold", bold)
        r.font.color.rgb = p.get("color", color); para.space_after = Pt(p.get("after", 6))
    return tb


def B(t, **k):
    return {"t": t, "bullet": True, **k}


def new_slide(title, kicker, notes=""):
    s = prs.slides.add_slide(BLANK)
    text(s, Inches(0.6), Inches(0.35), Inches(12), Inches(0.4), kicker.upper(), size=13, color=ACC, bold=True)
    text(s, Inches(0.6), Inches(0.65), Inches(12.1), Inches(1.0), title, size=28, bold=True)
    s.notes_slide.notes_text_frame.text = notes
    pn = text(s, Inches(12.2), Inches(7.0), Inches(0.9), Inches(0.35), "0", size=11, color=MUTED)
    pn.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    return s


def verdict(s, hyp, res, nxt=None, y=Inches(1.45)):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), y, Inches(12.1), Inches(0.62 if not nxt else 0.9))
    r.fill.solid(); r.fill.fore_color.rgb = LIGHT; r.line.fill.background(); r.adjustments[0] = 0.06
    lines = [{"t": "Hypothesis: " + hyp, "size": 13, "color": INK2, "after": 2}, {"t": "Result: " + res, "size": 13, "bold": True, "after": 2}]
    if nxt:
        lines.append({"t": "Led to: " + nxt, "size": 13, "color": ACC})
    text(s, Inches(0.8), y + Inches(0.04), Inches(11.8), Inches(0.9), lines, size=13)


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), x, y, width=w, height=h)


def table(s, rows, x, y, w, h, widths, size=12, first_bold=True, accent_col=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, w, h).table
    for c, wd in enumerate(widths):
        t.columns[c].width = Inches(wd)
    for r_, row in enumerate(rows):
        t.rows[r_].height = Emu(int(h / len(rows)))
        for c, v in enumerate(row):
            cell = t.cell(r_, c); cell.text = str(v)
            cell.margin_top = cell.margin_bottom = Emu(20000)
            p = cell.text_frame.paragraphs[0]; p.font.name = FONT; p.font.size = Pt(size)
            p.font.bold = (r_ == 0 or (first_bold and c == 0))
            p.font.color.rgb = ACC if (accent_col is not None and c == accent_col and r_) else INK
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if r_ == 0 else RGBColor(255, 255, 255)
    return t


def title_of(slide):
    texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    return texts[1] if len(texts) > 1 else (texts[0] if texts else "")


def index_of(title_start):
    for i, sl in enumerate(prs.slides):
        if title_of(sl).startswith(title_start):
            return i
    raise KeyError(title_start)


def move_last_to(idx):
    lst = prs.slides._sldIdLst
    el = lst[-1]; lst.remove(el); lst.insert(idx, el)


PENDING_DELETE = []


def delete(idx):
    """Mark a slide for deletion. Deletion happens after all new slides exist: deleting first made python-pptx reuse
    an existing slide's part name for a new slide (duplicate zip entries, blank slides)."""
    PENDING_DELETE.append(prs.slides._sldIdLst[idx])


def index_of_live(title_start):
    for i, sl in enumerate(prs.slides):
        if prs.slides._sldIdLst[i] in PENDING_DELETE:
            continue
        if title_of(sl).startswith(title_start):
            return i
    raise KeyError(title_start)


for f in ("tug_animals.png", "blocks_animals.png", "flow_animals.png", "patch_animals.png", "swap_ablation.png", "card_00.png"):
    shutil.copy(DIAG / f, A / f)

# ---- STRATEGIES: mixed-format training (after the format slide) -------------------------------------------------
s = new_slide("Mixed training doesn't carry pointing into direct answers", "Strategies · distillation (data fixed per panel)",
    notes=("Half the training targets are point lists (with the pointing prompt), half are bare numbers (plain question). If pointing were "
           "'distilled', the direct answers would improve over number-only training. They don't: synthetic data 11% on animals (numbers 13%), "
           "real-photo objects 22% (numbers 15%); other topics unchanged. With the pointing prompt at test time the synthetic mixed model reaches 34.9 "
           "overall (animals 54), still below the untrained model with the same prompt (39.0)."))
verdict(s, "training on both point lists and numbers teaches the model to count even when it answers directly (H3c, distillation)",
        "rejected: direct answers stay at number-training level (animals 11–22%); pointing helps only when the model points")
pic(s, A / "s_distill.png", Inches(0.45), Inches(2.25), w=Inches(12.4))
move_last_to(index_of_live("Format: numbers learn the data") + 1)

# ---- STRATEGIES: steering round 2 (replaces round 1) -------------------------------------------------------------
old = index_of_live("Steering: no effect")
s = new_slide("Steering: strong pushes break the familiar answer, not the prior", "Strategies · activation steering (no training)",
    notes=("A 'true count vs familiar count' direction from synthetic and leg counterfactuals (no VLMBias data), added at every text position "
           "after the image at layer 12, 16, 20 or 24, from 0.1x to 10x the hidden-state size, plus the reversed direction. Up to 0.3x: no effect. At "
           "about 1x the familiar answer collapses (animals: familiar 97 to 2-61%) but the true count barely appears (best: 26% on animals at layer 24) "
           "and unedited originals fall from 91 to 25-54%. At 3x and above the model stops answering with a number. The reversed direction also "
           "damages originals. So the direction moves answers away from the familiar count without making the model count: no single steering "
           "direction fixes it."))
verdict(s, "one 'count, don't recall' direction, learned without VLMBias data, makes the model give the true count",
        "rejected: ~1× removes the familiar answer but the true count barely appears (≤26%) and originals collapse")
pic(s, A / "s_steer2.png", Inches(0.3), Inches(2.3), w=Inches(12.7))
text(s, Inches(0.6), Inches(5.6), Inches(12.2), Inches(1.4), [
    B("Round 1 (≤0.2×, answer position only) had no effect; round 2 spans 0.1×–10× at every text position, plus the reversed direction", size=14),
    B("Steering the leg-trained model makes it worse in both directions", size=14),
    B("Consistent with the diagnosis: the familiar count is written by several late MLPs, not one direction", size=14, color=ORANGE)], size=14)
move_last_to(old + 1); delete(old)

# ---- DATASETS: synthetic neutral vs conflict (after the synthetic slide) -----------------------------------------
s = new_slide("Synthetic neutral images do as well as conflict pairs", "Datasets · 1b · synthetic: conflict pairs vs neutral images (point lists)",
    notes=("Same rendered families, point-list targets. Conflict pairs show a familiar object with its canonical and an edited count; neutral images "
           "show objects with no familiar count. On VLMBias the neutral model is no worse (animals 49.5 vs 40.5, other topics 25.6 vs 23.9) and both "
           "stay below the pointing prompt. The conflict structure is not what helps on synthetic data."))
verdict(s, "seeing canonical-vs-edited pairs (conflict) is what teaches the model to override its prior",
        "not on synthetic data: neutral images do as well on VLMBias (49.5 vs 40.5 on animals; 25.6 vs 23.9 on other topics)")
pic(s, A / "d_neutral.png", Inches(0.9), Inches(2.3), w=Inches(11.4))
move_last_to(index_of_live("Synthetic conflict pairs: solved") + 1)

# ---- DIAGNOSIS: three slides after the count-probe slide ----------------------------------------------------------
pos = index_of_live("The true count is formed, then overwritten") + 1
s = new_slide("The familiar count is written by the last MLP layers", "Diagnosis · per-item: where the answer turns (30 VLMBias animals)",
    notes=("Items the untrained model gets wrong and the leg-trained model gets right, forced direct answer. Left: running total of every block's "
           "direct push on the answer. All three lines stay near zero until layer ~18; the untrained model then drifts toward the familiar count and "
           "drops sharply in layers 28-31, ending at -5.8. The edited image and its unedited original give almost the same curve: the late layers "
           "write '4' whether or not the image was edited. The trained model stays flat and its last attention layer (31) pushes +2.5 toward the true "
           "count. Right: the blocks that push hardest toward the familiar count are the MLPs of layers 28, 29, 30 (and 27)."))
verdict(s, "the familiar count is retrieved somewhere specific late in the model",
        "yes: MLPs in layers 27–30 write it, the same for the edited image and its original; training adds a late counter-push")
pic(s, A / "tug_animals.png", Inches(0.4), Inches(2.3), w=Inches(6.6))
pic(s, A / "blocks_animals.png", Inches(7.0), Inches(2.3), w=Inches(6.0))
text(s, Inches(0.6), Inches(5.3), Inches(12.2), Inches(1.6), [
    B("Untrained: preference ≈ 0 until layer ~18, then −0.8 (L20), −1.6 (L24), −5.3 (L28), −5.8 at the answer", size=14),
    B("Edited and unedited images end the same (−5.8 vs −6.6): the late layers ignore the edit", size=14),
    B("Other topics show the same pattern, weaker (final −2.5; MLPs of layers 27, 28, 30)", size=14)], size=14)
move_last_to(pos)

s = new_slide("The fix lives in the text positions, not the subject words", "Diagnosis · attention and patching (30 VLMBias animals)",
    notes=("Left: share of the answer position's attention, standard-attention layers only (Qwen3.5 is single-stream; 8 of 32 layers are standard "
           "attention). Attention to the image rises to 34-38% at layers 19-23 and falls to 8% at layer 31; attention to the subject words ('legs', "
           "'animal') peaks at 12-15% in layers 15-19 and is ~2-3% in the last layers, where the template and other question words dominate. So the "
           "hypothesis that the late, overwriting layers look at the subject words is rejected. The trained model looks at the image more (49% at layer "
           "23). Knockouts of these routes move the preference by at most ~1 logit (vs -5.8). Right: copying one layer from the trained model into the "
           "untrained one. Image-token states carry none of the fix (~0); text-position states carry it from the middle layers (50% of the gap by layer "
           "14, 90% by layer 19); the answer position alone only fully from the last layers."))
verdict(s, "early layers look at the image; the late layers that overwrite the count look at the subject words in the text",
        "half right: image attention peaks mid-depth (34–38%); subject words peak mid (12–15%), only ~2% in the late layers")
pic(s, A / "flow_animals.png", Inches(0.3), Inches(2.3), w=Inches(7.4))
pic(s, A / "patch_animals.png", Inches(7.8), Inches(2.5), w=Inches(5.3))
text(s, Inches(0.6), Inches(5.6), Inches(12.2), Inches(1.4), [
    B("Knocking out attention to the image or the subject words moves the answer by ≤1 logit (vs −5.8): no single attention route carries the prior", size=14),
    B("Training changes how the TEXT positions read the image (mid layers), not the image tokens themselves", size=14, color=ORANGE)], size=14)
move_last_to(pos + 1)

s = new_slide("Removing the culprits only weakens the prior", "Diagnosis · causal tests",
    notes=("Left: replacing 'this animal' with 'this object' shifts the preference by +2.4 logits toward the true count on animals but flips no item; "
           "no effect on other topics. Right: zeroing the blocks that push hardest toward the familiar count (chosen on half the items, tested on the "
           "other half): with the top 10 removed, 13% of held-out animal items flip to the true count (other topics 40%), and every unedited original "
           "still gets 4. The prior is spread across many late MLPs rather than a few components. Bottom: one item card (lion, 5 legs)."))
verdict(s, "a few late components hold the prior; removing them makes the model count",
        "partly: removing 10 blocks flips 13% of animals (40% of other topics), originals intact; the prior is spread out")
pic(s, A / "swap_ablation.png", Inches(0.5), Inches(2.3), w=Inches(7.8))
pic(s, A / "card_00.png", Inches(2.1), Inches(4.95), w=Inches(9.0))
text(s, Inches(8.5), Inches(2.4), Inches(4.3), Inches(2.4), [
    B("'this animal' → 'this object': +2.4 logits toward the true count, 0 items flip", size=13),
    B("Top-10 ablation: animals 13% flip, other topics 40%; originals unchanged", size=13),
    B("Item card: attention overlays (early/late), question-word attention, running preference", size=13, color=INK2)], size=13)
move_last_to(pos + 2)

# ---- rebuilt: story, next, summary, appendix ---------------------------------------------------------------------
old = index_of_live("How the question evolved")
s = new_slide("How the question evolved: each result set the next step", "The search so far",
    notes="Each row: a hypothesis we tested, what came back, and what it changed. Strategies and datasets are reported in separate sections after this.")
rows = [("Question / hypothesis", "Result", "What it changed"),
        ("Does the model have a count prior?", "Yes: 12.7% on edited images vs 96.9% on originals", "Look for a fix"),
        ("Prompting it to point breaks the prior", "Partly: 39%, at −18 on originals", "Pointing is the bar to beat"),
        ("Instructions alone are enough", "No: animals stay at 0–1%", "Only pointing unlocks counting"),
        ("Synthetic conflict pairs transfer", "Synthetic yes (89%); VLMBias no (28%); neutral images do as well", "Try real photos"),
        ("Real-photo object counts transfer", "Real photos +21; VLMBias no (30%)", "VLMBias counts parts: animal legs"),
        ("Real animal-leg edits transfer", "v1 ties prompting, worse on animals", "Diagnose the data"),
        ("The leg data was badly mixed", "v2: animals 90.5 vs 78.8; other topics no gain", "Ask where the count is lost"),
        ("Mixed point/number training distills pointing", "No: direct answers stay at number level", "Pointing must happen at answer time"),
        ("The model can't see the edited part", "Rejected: every leg decodable from image tokens", "Count probes"),
        ("The model never forms the true count", "Rejected (animals): formed mid-depth, overwritten at the end", "Per-item diagnosis"),
        ("Late layers read the subject words", "Rejected: MLPs in layers 27–30 write the familiar count", "Ablation, steering"),
        ("A few components / one direction hold the prior", "No: ablation flips 13%; steering breaks answers", "RL; better data"),
        ("Training only the late layers fixes it", "Rejected: worse everywhere (originals 62)", "—")]
table(s, rows, Inches(0.5), Inches(1.45), Inches(12.3), Inches(5.6), (4.4, 4.9, 3.0), size=11, first_bold=False, accent_col=2)
move_last_to(old + 1); delete(old)

old = index_of_live("Next:")
s = new_slide("Next: RL from the pointing prompt", "Next steps",
    notes=("RL failed at startup (a missing package for ms-swift's GRPO trainer, now installed automatically) and reruns next. The diagnosis says the "
           "prior is written broadly by late MLPs and the trained model's fix sits in how text positions read the image; RL that rewards correct points "
           "and totals targets exactly that reading."))
steps = [("✓", "Strategies, datasets, prompts, distillation, steering", "Pointing at answer time is the lever; training helps only the trained category"),
         ("✓", "Probes and per-item diagnosis", "Count formed mid-depth; late MLPs (27–30) write the familiar count; prior spread out"),
         ("1", "RL from the pointing prompt (fixed, rerunning)", "Rewards: correct total + points on real legs; never the familiar answer"),
         ("2", "Diagnosis of more trained models", "Do RL and the mixed models change the late MLPs or the text-position reading?"),
         ("3", "Remaining controls, then seeds", "Synthetic neutral/mixed arms done; dice-only, leg-data seeds last")]
for i, (n, name, desc) in enumerate(steps):
    y = Inches(1.8 + i * 0.98)
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), y + Inches(0.05), Inches(0.55), Inches(0.55))
    c.fill.solid(); c.fill.fore_color.rgb = GREEN if n == "✓" else (ACC if i < 4 else MUTED); c.line.fill.background()
    c.text_frame.text = n; p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; p.font.size = Pt(16); p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255); p.font.name = FONT
    text(s, Inches(1.5), y, Inches(3.9), Inches(0.9), name, size=16, bold=True)
    text(s, Inches(5.4), y + Inches(0.03), Inches(7.4), Inches(0.8), desc, size=15, color=INK2)
move_last_to(old + 1); delete(old)

old = index_of_live("Summary: hypotheses so far")
s = new_slide("Summary: hypotheses so far", "Takeaways")
rows = [("Hypothesis", "Verdict"),
        ("The model has a strong count prior", "Supported: 12.7% on edited vs 96.9% on originals"),
        ("Prompting it to point breaks the prior", "Partly: 39%, at −18 on originals; the bar to beat"),
        ("Instructions alone are enough", "Rejected: animals stay at 0–1%"),
        ("Point-list training beats number training", "Mixed: numbers learn the data, points add a little transfer"),
        ("Mixed point/number training distills pointing", "Rejected: direct answers stay at number-training level"),
        ("Conflict pairs matter (vs neutral images)", "No (synthetic): neutral images do as well"),
        ("Some training dataset transfers beyond its category", "Not yet: best gain is in-category (animals 90.5)"),
        ("Training only the late layers works better", "Rejected: worse on every measure"),
        ("The model can't see the edited part / never forms the count", "Rejected (animals): seen, formed mid-depth, overwritten late"),
        ("Late layers read the subject words to recall the prior", "Rejected: late MLPs (27–30) write it; late attention is on the template"),
        ("A few components or one direction hold the prior", "Rejected: ablation flips 13%; steering breaks answers instead")]
table(s, rows, Inches(0.6), Inches(1.45), Inches(12.1), Inches(5.6), (5.8, 6.3), size=12)
move_last_to(old + 1); delete(old)

old = index_of_live("Appendix: all results")
s = new_slide("Appendix: all results", "Appendix",
    notes="Thinking off, greedy decoding, seed 0 for trained models. Run ids in the last column for cross-reference with RESULTS.md.")
hdr = ["Model", "VLMBias", "Other topics", "Animals", "Originals", "Balanced", "Synth. held-out (edited)", "Color", "Real photos", "Run id"]
data = [["No training, plain prompt", 12.7, 17.4, 0.0, 96.9, 54.8, 47.3, 95.2, 37.7, "Qwen3.5-4B"],
        ["No training, direct-answer prompt", 9.0, 12.4, 0.0, 95.1, 52.1, 52.4, 95.2, 37.2, "+direct"],
        ["No training, de-bias prompt", 19.5, 26.3, 1.3, 87.1, 53.3, "–", "–", "–", "+debias"],
        ["No training, pointing prompt", 39.0, 24.0, 78.8, 79.1, 59.1, 61.1, 95.2, 47.1, "+pointT"],
        ["No training, pointing + de-bias prompt", 40.2, 27.2, 74.5, 81.6, 60.9, "–", "–", "–", "+pointDebias"],
        ["Synthetic conflict pairs, numbers", 20.6, 23.6, 12.6, 92.0, 56.3, 75.7, 95.2, 45.6, "conflict-count"],
        ["Synthetic pairs, images grayed (control)", 9.8, 13.5, 0.0, 86.5, 48.1, 45.6, 93.1, 44.3, "conflict-blind"],
        ["Synthetic conflict pairs, point lists", 28.4, 23.9, 40.5, 72.4, 50.4, 88.7, 82.1, 44.4, "conflict-point"],
        ["Synthetic conflict pairs, mixed", 19.9, 23.7, 9.9, 85.3, 52.6, 73.4, 91.8, 37.3, "conflict-mixed"],
        ["Synthetic conflict pairs, mixed + pointing prompt", 34.9, 27.5, 54.4, 74.8, 54.9, "–", "–", "–", "conflict-mixed+pointT"],
        ["Synthetic neutral images, point lists", 32.1, 25.6, 49.5, 65.0, 48.6, 80.0, 85.9, 42.1, "neutral-point"],
        ["Real-photo objects, numbers", 16.1, 16.5, 14.8, 82.2, 49.1, 42.8, 94.8, 68.2, "real-count"],
        ["Real-photo objects, mixed", 17.0, 15.2, 21.8, 79.1, 48.1, 33.9, 91.4, 67.6, "real-mixed"],
        ["Real-photo objects, point lists", 30.4, 22.5, 51.1, 71.8, 51.1, 62.2, 91.8, 68.5, "real-point"],
        ["Natural animal legs, point lists", 21.4, 26.1, 9.0, 80.4, 50.9, 74.3, 59.8, 41.4, "part-neutral-point"],
        ["Animal-leg edits v1, numbers", 30.1, 15.8, 67.8, 92.0, 61.0, 55.6, 94.5, 49.8, "part-conflict-count"],
        ["Animal-leg edits v1, point lists", 38.8, 29.5, 63.2, 78.5, 58.6, 78.7, 37.5, 47.8, "part-conflict-point"],
        ["Animal-leg edits v1, point lists + de-bias prompt", 40.5, 28.8, 71.2, 79.8, 60.1, "–", "–", "–", "part-conflict-point+debias"],
        ["Animal-leg edits v2, numbers", 38.4, 16.4, 96.5, 86.5, 62.4, 57.2, 97.9, 45.6, "part2-conflict-count"],
        ["Animal-leg edits v2, point lists", 43.9, 26.3, 90.5, 78.5, 61.2, 69.8, 96.9, 48.2, "part2-conflict-point"],
        ["Animal-leg edits v2, point lists, layers 16–31 only", 37.7, 20.7, 82.6, 62.0, 49.8, 68.8, 96.2, 45.9, "part2-...-L16-31"]]
table(s, [hdr] + data, Inches(0.35), Inches(1.3), Inches(12.6), Inches(5.6), (3.5, 0.8, 0.9, 0.8, 0.85, 0.85, 1.2, 0.7, 0.8, 2.2), size=8.5)
text(s, Inches(0.35), Inches(6.95), Inches(12.6), Inches(0.5), [{"t":
    "Other topics: VLMBias without animals. Originals: unedited images (guardrail). Balanced: mean of VLMBias and originals. Mixed: half point lists, "
    "half numbers. Real photos: unaudited LVIS labels.", "size": 9, "color": INK2}], size=9)
move_last_to(old + 1); delete(old)

lst = prs.slides._sldIdLst
for el in PENDING_DELETE:
    prs.part.drop_rel(el.rId); lst.remove(el)

# ---- renumber -----------------------------------------------------------------------------------------------------
for i, sl in enumerate(prs.slides, start=1):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().isdigit() and sh.left > Inches(11.5) and sh.top > Inches(6.8):
            runs = sh.text_frame.paragraphs[0].runs
            if runs:
                runs[0].text = str(i)
prs.save(DECK)
print("saved", DECK, len(prs.slides), "slides")
