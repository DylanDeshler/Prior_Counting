"""External inputs for L0 (§8): Open Images background photos and VLMBias statistics/exclusions.
Photo-level sources live in coco.py, colorsets.py and exclusions.py."""

import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from .common import DATA, SOURCES, item_seed, update_params

OI_CLASSES = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_TRAIN_BOXES = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
# Backgrounds must not contain held-out classes or any target family (§9.5).
OI_EXCLUDED = ["Traffic light", "Traffic sign", "Stop sign", "Clock", "Alarm clock", "Wall clock", "Digital clock",
               "Watch", "Dice", "Piano", "Musical keyboard", "Snowman"]
N_BACKGROUNDS = 3000
EXCLUSIONS = DATA / "exclusions"

# VLMBias topics are excluded as counted/colored objects everywhere (§4.2).
VLMBIAS_EXCLUDED_TERMS = ["animal", "leg", "dog", "cat", "bird", "horse", "logo", "ring", "stripe", "flag",
                          "chess", "xiangqi", "board", "go grid", "sudoku", "dice pattern", "tally", "grid pattern"]


def fetch_openimages_backgrounds(n=N_BACKGROUNDS, refresh=False):
    out = SOURCES / "openimages_bg"
    manifest = SOURCES / "openimages_bg_manifest.json"
    if not refresh and manifest.exists() and len(list(out.glob("*.jpg"))) >= len(json.loads(manifest.read_text())["image_ids"]):
        print(f"Open Images backgrounds: up to date ({manifest})")
        return
    out.mkdir(parents=True, exist_ok=True)
    classes = pd.read_csv(OI_CLASSES, header=None, names=["mid", "name"])
    excluded_mids = set(classes[classes["name"].isin(OI_EXCLUDED)]["mid"])
    missing = set(OI_EXCLUDED) - set(classes[classes["mid"].isin(excluded_mids)]["name"])
    assert not missing, f"Open Images classes not found: {missing}"

    # Stream the 2.3 GB train box file once; keep only image-id sets.
    all_ids, bad_ids = set(), set()
    with requests.get(OI_TRAIN_BOXES, stream=True, timeout=60) as r, \
            tqdm(total=int(r.headers.get("content-length", 0)) or None, unit="B", unit_scale=True,
                 desc="Open Images train boxes (stream)") as bar:
        r.raise_for_status()
        r.raw.decode_content = True
        raw_read = r.raw.read

        def read(*a, **k):  # count bytes as pandas pulls them
            data = raw_read(*a, **k)
            bar.update(len(data))
            return data

        r.raw.read = read
        for chunk in pd.read_csv(r.raw, usecols=["ImageID", "LabelName"], chunksize=2_000_000):
            all_ids.update(chunk["ImageID"])
            bad_ids.update(chunk.loc[chunk["LabelName"].isin(excluded_mids), "ImageID"])
    pool = sorted(all_ids - bad_ids)
    rng = np.random.default_rng(item_seed("openimages_bg"))
    chosen = sorted(rng.choice(pool, size=min(len(pool), int(n * 1.1)), replace=False).tolist())

    def download(image_id):
        path = out / f"{image_id}.jpg"
        if path.exists():
            return image_id
        try:
            resp = requests.get(OI_IMAGE.format(image_id), timeout=30)
            resp.raise_for_status()
            im = Image.open(io.BytesIO(resp.content)).convert("RGB")
            im.thumbnail((640, 640), Image.LANCZOS)
            im.save(path, quality=92)
            return image_id
        except Exception:
            return None

    with ThreadPoolExecutor(32) as ex:
        got = [i for i in tqdm(ex.map(download, chosen), total=len(chosen), desc="backgrounds") if i]
    keep = set(sorted(got)[:n])
    for p in out.glob("*.jpg"):  # exactly n files, deterministic choice
        if p.stem not in keep:
            p.unlink()
    (SOURCES / "openimages_bg_manifest.json").write_text(json.dumps({
        "excluded_classes": OI_EXCLUDED, "candidates": len(pool), "image_ids": sorted(keep),
        "license": "CC-BY 2.0 (Open Images V7 train)"}, indent=1))
    print(f"Open Images backgrounds: {len(keep)} of {len(pool)} clean train images")


def fetch_vlmbias_stats(refresh=False):
    """Shared-block upper bound U = max(40, largest VLMBias counting label) (§4.3, §12)."""
    from .common import load_params
    if not refresh and load_params().get("vlmbias_max_count") and (EXCLUSIONS / "vlmbias_categories.json").exists():
        print("VLMBias stats: up to date")
        return
    from datasets import load_dataset
    ds = load_dataset("anvo25/vlms-are-biased", split="main").remove_columns(["image"])
    counts = [int(g) for g, t in zip(ds["ground_truth"], ds["topic"]) if t != "Optical Illusion" and str(g).isdigit()]
    largest = max(counts)
    topics = sorted(set(zip(ds["topic"], ds["sub_topic"])))
    EXCLUSIONS.mkdir(parents=True, exist_ok=True)
    (EXCLUSIONS / "vlmbias_categories.json").write_text(json.dumps(
        {"topics": [list(t) for t in topics], "excluded_terms": VLMBIAS_EXCLUDED_TERMS}, indent=1))
    p = update_params(vlmbias_counting_items=len(counts), vlmbias_max_count=largest,
                      shared_block_max=max(40, largest), shared_block_max_source="max(40, VLMBias max count)")
    print(f"VLMBias: {len(counts)} counting items, largest count {largest} -> U = {p['shared_block_max']}")


def fetch_all(refresh=False):
    """Each source is skipped if already present (pass refresh=True / `sources refresh` to redo)."""
    fetch_vlmbias_stats(refresh)
    fetch_openimages_backgrounds(refresh=refresh)
    # Photo levels (L2 color): COCO/LVIS annotations, CoDa x LVIS color objects, eval-set exclusions.
    from . import coco, colornames, colorsets, exclusions
    colornames.fetch()
    coco.fetch_all()
    colorsets.fetch()
    exclusions.fetch_all(refresh)
