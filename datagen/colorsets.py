"""Color objects for the photo levels (§5.2): CoDa objects matched to LVIS categories.

  conflict ("single" in CoDa): a strong typical color; counterfactual colors have CoDa p < 0.02
  neutral  ("any" in CoDa):    no typical color; recolored to colors sampled from their own CoDa distribution
Excluded at every level (§4.2): animals and body parts, flags, rings, chess pieces, board games, the
held-out count classes, and every Visual CounterFact color object (evaluation set). 20% of the conflict
objects are held out for the color test sets.

Writes data/sources/color_objects.json:
    {"objects": [{name, group, lvis_cats [id], coda {term: p}, modal, cf_colors [term], split}],
     "dropped": {name: reason}, "vcf_color_objects": [...]}
"""

import hashlib
import json
import re

import requests

from .colornames import CHROMATIC, TERMS
from .common import SOURCES

CODA_URL = "https://huggingface.co/datasets/corypaik/coda/resolve/main/data/default_{split}.jsonl"
VCF_REPO = "datasets/mgolov/Visual-Counterfact"
OUT = SOURCES / "color_objects.json"
CODA_GROUPS = {0: "single", 1: "multi", 2: "any"}
HELDOUT_FRAC = 0.2
CONFLICT_MAX_P = 0.02

# CoDa's object names are mechanically singularized ("scissor", "mangoe", "taxis"); fix before matching.
CODA_FIX = {"scissor": "scissors", "mangoe": "mango", "radishe": "radish", "taxis": "taxi", "kiwis": "kiwi fruit",
            "burritos": "burrito", "flamingos": "flamingo", "hippopotamuse": "hippopotamus", "rhinocero": "rhinoceros",
            "binocular": "binoculars", "chopstick": "chopsticks", "houseplant": "potted plant",
            "television": "television set"}
# Excluded as counted/colored objects (§4.2), beyond the WordNet animal/body-part test.
EXCLUDED_WORDS = {"flag", "ring", "chess", "checkerboard", "board game", "dice", "domino", "playing card",
                  "traffic light", "snowflake", "stop sign", "clock", "logo",
                  # display / print surfaces: recoloring them tints the picture they show
                  "billboard", "poster", "sign", "painting", "picture", "television screen", "monitor"}
EXCLUDED_HYPERNYMS = {"animal.n.01", "body_part.n.01", "person.n.01", "flag.n.01"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_coda():
    objs = {}
    for split in ("train", "validation", "test"):
        for line in requests.get(CODA_URL.format(split=split), timeout=60).text.splitlines():
            r = json.loads(line)
            objs[r["ngram"]] = {"group": CODA_GROUPS[r["object_group"]], "dist": r["label"]}
    return objs


def fetch_vcf_color_objects():
    """Objects of Visual CounterFact's color split (its size split asks about size, so it doesn't collide)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"{VCF_REPO}/data/color-00000-of-00001.parquet", "rb") as fh:
        return sorted({str(v).lower() for v in pq.read_table(fh, columns=["object"]).column("object").to_pylist()})


def _wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    return wn


def wordnet_excluded(synset_name, wn):
    """True if the LVIS synset is an animal, body part, person or flag (WordNet hypernym closure)."""
    try:
        syn = wn.synset(synset_name)
    except Exception:
        return False
    closure = {s.name() for s in syn.closure(lambda s: s.hypernyms())} | {syn.name()}
    return bool(closure & EXCLUDED_HYPERNYMS)


def lvis_name_index(cats):
    idx = {}
    for cid, c in cats.items():
        for n in [c["name"]] + c["synonyms"]:
            n = re.sub(r"\s*\(.*?\)", "", n.replace("_", " ")).strip().lower()
            idx.setdefault(norm(n), set()).add(cid)
    return idx


def match_lvis(name, idx):
    for v in (name, name + "s", name + "es", name[:-1] if name.endswith("s") else None):
        if v and norm(v) in idx:
            return sorted(idx[norm(v)])
    return []


def build(coda, cats, vcf):
    wn = _wordnet()
    idx = lvis_name_index(cats)
    vcf_norm = {norm(v) for v in vcf} | {norm(v.rstrip("s")) for v in vcf}
    objects, dropped = [], {}
    for coda_name, o in sorted(coda.items()):
        if o["group"] == "multi":
            continue
        name = CODA_FIX.get(coda_name, coda_name)
        lvis_ids = match_lvis(name, idx)
        if not lvis_ids:
            dropped[name] = "not an LVIS category"
            continue
        if norm(name) in vcf_norm:
            dropped[name] = "in Visual CounterFact (color)"
            continue
        if any(w in name for w in EXCLUDED_WORDS) or any(wordnet_excluded(cats[c]["synset"], wn) for c in lvis_ids):
            dropped[name] = "excluded category (§4.2)"
            continue
        coda_p = dict(zip(TERMS, [round(p, 4) for p in o["dist"]]))
        modal = max(coda_p, key=coda_p.get)
        rec = {"name": name, "group": o["group"], "lvis_cats": lvis_ids, "coda": coda_p, "modal": modal}
        if o["group"] == "single":
            rec["cf_colors"] = [t for t in CHROMATIC if t != modal and coda_p[t] < CONFLICT_MAX_P]
        objects.append(rec)
    singles = sorted((o for o in objects if o["group"] == "single"),
                     key=lambda o: hashlib.sha256(o["name"].encode()).hexdigest())
    held = {o["name"] for o in singles[:round(HELDOUT_FRAC * len(singles))]}
    for o in objects:
        o["split"] = "heldout" if o["name"] in held else "train"
    return objects, dropped


def fetch():
    from . import coco
    coco.fetch_lvis("val")
    cats = coco.lvis("val")["cats"]
    coda, vcf = fetch_coda(), fetch_vcf_color_objects()
    objects, dropped = build(coda, cats, vcf)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"objects": objects, "dropped": dropped, "vcf_color_objects": vcf}, indent=1))
    n = lambda g, s: sum(o["group"] == g and o["split"] == s for o in objects)
    print(f"color objects: conflict train {n('single', 'train')}, conflict held-out {n('single', 'heldout')}, "
          f"neutral {n('any', 'train')}; dropped {len(dropped)}")


def load():
    return {o["name"]: o for o in json.loads(OUT.read_text())["objects"]}
