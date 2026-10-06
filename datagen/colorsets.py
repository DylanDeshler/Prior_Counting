"""Color object sets (§5.2): CoDa ∩ Noto Emoji ∩ LVIS/COCO, exclusions, held-out split, recolor targets.

Writes data/sources/color_objects.json:
    {"objects": [{name, group ("single"|"any"), emoji, cps, coda {term: p}, modal, current, targets {term: [a, b]},
                  target_probs {term: p} (any only), reach {term: frac}, split ("train"|"heldout")}],
     "dropped": {name: reason}, "vcf_objects": [...], "lvis_count": n}
"""

import hashlib
import io
import json
import re
import zipfile

import numpy as np
import requests

from . import emoji as E
from .colornames import CHROMATIC, TERMS, lab_to_srgb, name_indices, srgb_to_lab, w2c
from .common import SOURCES

CODA_URL = "https://huggingface.co/datasets/corypaik/coda/resolve/main/data/default_{split}.jsonl"
LVIS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
VCF_REPO = "datasets/mgolov/Visual-Counterfact"
OUT = SOURCES / "color_objects.json"
LVIS_CATS = SOURCES / "lvis_categories.json"

CODA_GROUPS = {0: "single", 1: "multi", 2: "any"}
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]

# CoDa object -> Noto/CLDR emoji name, where the names differ but the object is the same.
ALIASES = {"orange": "tangerine", "grape": "grapes", "blueberry": "blueberries", "corn": "ear of corn",
           "milk": "glass of milk", "honey": "honey pot", "pancake": "pancakes", "football": "american football",
           "tennis ball": "tennis", "car": "automobile", "shirt": "t-shirt", "tie": "necktie", "book": "closed book",
           "sock": "socks", "glove": "gloves", "suitcase": "luggage", "ski": "skis", "cherry": "cherries",
           "truck": "delivery truck"}

# §4.2 / §9 exclusions for colored objects: animals and body parts, flags, rings and stripe logos,
# chess pieces and board games, and held-out classes.
EXCLUDED_GROUPS = {"People & Body", "Flags"}
EXCLUDED_SUBGROUP_PREFIXES = ("animal", "game", "person", "hand", "body", "face")
# CoDa's object names are mechanically singularized ("scissor", "mangoe", "taxis"); fix them before matching.
CODA_FIX = {"scissor": "scissors", "mangoe": "mango", "radishe": "radish", "taxis": "taxi", "kiwis": "kiwi fruit",
            "burritos": "burrito", "flamingos": "flamingo", "hippopotamuse": "hippopotamus", "rhinocero": "rhinoceros",
            "binocular": "binoculars", "chopstick": "chopsticks"}
# Emoji that depict the object under another Unicode name (only where the emoji shows just that object).
ALIASES.update({"box": "package", "chocolate": "chocolate bar", "chopsticks": "chopsticks", "clock": "mantelpiece clock",
                "pumpkin": "jack-o-lantern", "toilet paper": "roll of paper", "tin can": "canned food",
                "nut": "chestnut", "pea": "pea pod", "wheelchair": "manual wheelchair", "houseplant": "potted plant",
                "cheese": "cheese wedge", "coffee": "hot beverage", "beer": "beer mug", "picnic basket": "basket"})
# LVIS/COCO category for objects whose name differs there.
LVIS_ALIASES = {"chopsticks": "chopstick", "chocolate": "chocolate bar", "houseplant": "potted plant",
                "picnic basket": "basket"}
EXCLUDED_NAMES = {"ring", "traffic light", "snowflake", "chess pawn", "game die", "playing card"}
# Emoji whose dominant colored region is not the named object (necktie: mostly a shirt).
NOT_THE_OBJECT = {"tie"}

HELDOUT_FRAC = 0.2
CONFLICT_MAX_P = 0.02
REACH_MIN = 0.9    # share of recolored pixels named the target
CONF_MIN = 0.6     # mean w2c probability of the target over those pixels (rejects e.g. cream labeled "brown")
DOMINANT_MIN = 0.7  # the recolored (answer) color must cover >= 70% of the object, so "what color is X" has one answer
VERIFY_MIN = 0.65   # same rule per rendered image (a little slack for resampling)
CHROMATIC_MIN = 0.25
# Lightness guards (mean L* of the recolored pixels): yellow/pink/orange read as olive/mauve/brown when dark,
# brown reads as beige when light, even where w2c still names them.
L_RANGE = {"yellow": (70, 101), "pink": (60, 101), "orange": (50, 101), "brown": (0, 55)}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_coda():
    objs = {}
    for split in ("train", "validation", "test"):
        for line in requests.get(CODA_URL.format(split=split), timeout=60).text.splitlines():
            r = json.loads(line)
            objs[r["ngram"]] = {"group": CODA_GROUPS[r["object_group"]], "dist": r["label"]}
    return objs


def fetch_lvis():
    if not LVIS_CATS.exists():
        data = requests.get(LVIS_URL, timeout=600).content
        with zipfile.ZipFile(io.BytesIO(data)) as z, z.open(z.namelist()[0]) as f:
            cats = json.load(f)["categories"]
        LVIS_CATS.write_text(json.dumps([{"name": c["name"], "synonyms": c["synonyms"]} for c in cats]))
    names = set()
    for c in json.loads(LVIS_CATS.read_text()):
        for n in [c["name"]] + c["synonyms"]:
            names.add(re.sub(r"\s*\(.*?\)", "", n.replace("_", " ")).strip().lower())
    return names


def fetch_vcf_objects():
    """Objects of Visual CounterFact's color split. Its size split asks about relative size, not
    color, so those objects stay eligible as color objects (scope decision, 2026-10-05)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"{VCF_REPO}/data/color-00000-of-00001.parquet", "rb") as fh:
        return sorted({str(v).lower() for v in pq.read_table(fh, columns=["object"]).column("object").to_pylist()})


def excluded(emoji_name, info):
    if info["group"] in EXCLUDED_GROUPS or info["subgroup"].startswith(EXCLUDED_SUBGROUP_PREFIXES):
        return f"emoji group {info['group']}/{info['subgroup']}"
    if emoji_name in EXCLUDED_NAMES:
        return "excluded name"
    return None


def best_ab(lab_px, target, k=0.5, step=6):
    """ab maximizing the fraction of pixels named `target` after recolor with L fixed (§5.3)."""
    mean = lab_px[:, 1:].mean(axis=0)
    dev = k * (lab_px[:, 1:] - mean)
    grid = np.arange(-96, 97, step, dtype=float)
    ti = TERMS.index(target)
    table = w2c()
    best = (-1.0, None, 0.0, 0.0)
    for a in grid:
        cand = np.stack(np.meshgrid([a], grid, indexing="ij"), -1).reshape(-1, 2)  # (G, 2)
        ab = cand[:, None, :] + dev[None]                                          # (G, N, 2)
        lab = np.concatenate([np.broadcast_to(lab_px[None, :, :1], ab.shape[:2] + (1,)), ab], -1)
        raw = lab_to_srgb(lab, clip=False)
        clip = np.abs(raw - np.clip(raw, 0, 255)).mean(axis=(1, 2))
        px = np.round(np.clip(raw, 0, 255)).astype(np.int64)
        probs = table[px[..., 0] // 8 + 32 * (px[..., 1] // 8) + 1024 * (px[..., 2] // 8)]
        frac = (probs.argmax(-1) == ti).mean(axis=1)
        conf = probs[..., ti].mean(axis=1)
        score = frac + conf - 0.002 * clip
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), (float(cand[i, 0]), float(cand[i, 1])), float(frac[i]), float(conf[i]))
    return best[1], best[2], best[3]


def current_confidence(cps, term):
    """Mean w2c probability of `term` over the emoji pixels named `term`."""
    rgba = E.base_rgba(cps)
    px = rgba[..., :3][E.term_mask(rgba, term)].astype(np.int64)
    probs = w2c()[px[:, 0] // 8 + 32 * (px[:, 1] // 8) + 1024 * (px[:, 2] // 8)]
    return float(probs[:, TERMS.index(term)].mean())


def analyze(cps):
    """Object color: the dominant chromatic term if it covers >= CHROMATIC_MIN of the object, else the
    majority term. Returns (term, LAB samples of its pixels, its share of the object)."""
    rgba = E.base_rgba(cps)
    opaque = rgba[..., 3] > 127
    share = np.bincount(name_indices(rgba[..., :3][opaque]), minlength=len(TERMS)) / opaque.sum()
    chrom = max(CHROMATIC, key=lambda t: share[TERMS.index(t)])
    current = chrom if share[TERMS.index(chrom)] >= CHROMATIC_MIN else TERMS[int(np.argmax(share))]
    px = rgba[..., :3][E.term_mask(rgba, current)]
    px = px[:: max(1, len(px) // 1500)]
    return current, srgb_to_lab(px), float(share[TERMS.index(current)])


def build(coda, lvis, vcf):
    index = E.emoji_index()
    vcf_norm = {norm(v) for v in vcf} | {norm(v.rstrip("s")) for v in vcf}
    lvis_coco = lvis | set(COCO)
    candidates, dropped = {}, {}
    def in_lvis(n):
        return any(v in lvis_coco for v in (n, LVIS_ALIASES.get(n), n + "s", n[:-1] if n.endswith("s") else None))

    for coda_name, o in sorted(coda.items()):
        if o["group"] == "multi":
            continue
        name = CODA_FIX.get(coda_name, coda_name)
        ename = ALIASES.get(name, name)
        if ename not in index:
            continue
        if not in_lvis(name):
            dropped[name] = "not in LVIS/COCO"
            continue
        why = excluded(ename, index[ename]) or ("emoji shows more than the object" if name in NOT_THE_OBJECT else None)
        if why or norm(name) in vcf_norm:
            dropped[name] = why or "in Visual CounterFact"
            continue
        candidates[name] = {"emoji": ename, "cps": index[ename]["cps"], "group": o["group"],
                            "coda": dict(zip(TERMS, [round(p, 4) for p in o["dist"]]))}
    have = E.fetch_assets([c["cps"] for c in candidates.values()])
    objects = []
    for name, c in candidates.items():
        if c["cps"] not in have:
            dropped[name] = "no Noto asset"
            continue
        coda_p = c["coda"]
        modal = max(coda_p, key=coda_p.get)
        current, lab_px, share = analyze(c["cps"])
        if share < DOMINANT_MIN:
            dropped[name] = f"{current} covers only {share:.0%} of the emoji (need {DOMINANT_MIN:.0%})"
            continue
        # Every chromatic color the main region can be recolored to with a confident name (§5.3).
        # Both images of a pair are recolored to one of these, so labels never rest on how the emoji
        # happens to be drawn (e.g. a dark orange chair that people call brown).
        mean_l = float(lab_px[:, 0].mean())
        reach_ab, reach = {}, {}
        for t in CHROMATIC:
            lo, hi = L_RANGE.get(t, (0, 101))
            if not lo <= mean_l <= hi:
                continue
            ab, frac, conf = best_ab(lab_px, t)
            reach[t] = [round(frac, 3), round(conf, 3)]
            if frac >= REACH_MIN and conf >= CONF_MIN:
                reach_ab[t] = [round(ab[0], 1), round(ab[1], 1)]
        if c["group"] == "single":
            if modal in CHROMATIC:
                if modal not in reach_ab:
                    dropped[name] = f"modal {modal} not reachable at L*={mean_l:.0f} ({reach})"
                    continue
                start = modal  # canonical: recolored to the familiar color
            elif current == modal and current_confidence(c["cps"], current) >= CONF_MIN:
                start = None   # achromatic familiar color (white snowman): identity recolor
            else:
                dropped[name] = f"achromatic modal {modal}, emoji reads {current}"
                continue
            targets = {t: ab for t, ab in reach_ab.items() if t != modal and coda_p[t] < CONFLICT_MAX_P}
        else:
            start = None       # sampled per item from target_probs
            targets = {t: ab for t, ab in reach_ab.items() if coda_p[t] > 0}
            if len(targets) < 2:
                targets = {}
        if not targets:
            dropped[name] = f"no reachable target colors ({reach})"
            continue
        rec = {"name": name, **c, "modal": modal, "current": current, "share": round(share, 3),
               "mean_l": round(mean_l, 1), "start": start, "start_ab": reach_ab.get(start), "targets": targets,
               "reach": reach}
        if c["group"] == "any":
            z = sum(coda_p[t] for t in targets)
            rec["target_probs"] = {t: round(coda_p[t] / z, 4) for t in targets}
        objects.append(rec)
    # Deterministic 20% held-out split of the Single objects (by hash, not by list position).
    singles = sorted((o for o in objects if o["group"] == "single"),
                     key=lambda o: hashlib.sha256(o["name"].encode()).hexdigest())
    n_held = round(HELDOUT_FRAC * len(singles))
    held = {o["name"] for o in singles[:n_held]}
    for o in objects:
        o["split"] = "heldout" if o["name"] in held else "train"
    return objects, dropped


def fetch():
    E.fetch_index()
    coda, lvis, vcf = fetch_coda(), fetch_lvis(), fetch_vcf_objects()
    objects, dropped = build(coda, lvis, vcf)
    OUT.write_text(json.dumps({"objects": objects, "dropped": dropped, "vcf_objects": vcf,
                               "lvis_count": len(lvis)}, indent=1))
    n = lambda g, s=None: sum(o["group"] == g and (s is None or o["split"] == s) for o in objects)
    print(f"color objects: single train {n('single', 'train')}, single held-out {n('single', 'heldout')}, "
          f"any {n('any')}; dropped {len(dropped)}")
    return objects


def load():
    return json.loads(OUT.read_text())
