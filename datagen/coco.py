"""COCO 2017 / LVIS v1 annotations and an on-demand image cache for the photo levels (§5.5, §8.1).

LVIS v1 annotates COCO 2017 images with 1,203 categories and polygon masks. Only the images we
actually use are downloaded (from each image's coco_url), never the 18 GB archives.

Index files (pickles in data/sources/coco/):
    lvis_<split>.pkl  {"images": {id: {url, w, h, not_exhaustive, neg}}, "anns": {image_id: [ann...]},
                       "cats": {id: {name, synonyms, synset}}}
    coco_extra.pkl    per image: COCO crowd category names, traffic-light boxes (COCO is exhaustive for
                      its 80 classes, LVIS is federated)
"""

import io
import json
import pickle
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
from PIL import Image
from tqdm import tqdm

from .common import SOURCES

LVIS_URLS = {"train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
             "val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"}
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DIR = SOURCES / "coco"
IMAGES = DIR / "images"


def _download(url, desc):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        with tqdm(total=int(r.headers.get("content-length", 0)) or None, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(1 << 20):
                buf.write(chunk)
                bar.update(len(chunk))
    buf.seek(0)
    return buf


def fetch_lvis(split):
    out = DIR / f"lvis_{split}.pkl"
    if out.exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_download(LVIS_URLS[split], f"LVIS {split} annotations")) as z, z.open(z.namelist()[0]) as f:
        data = json.load(f)
    cats = {c["id"]: {"name": c["name"], "synonyms": c["synonyms"], "synset": c.get("synset")} for c in data["categories"]}
    images = {im["id"]: {"url": im["coco_url"], "w": im["width"], "h": im["height"],
                         "not_exhaustive": im.get("not_exhaustive_category_ids", []),
                         "neg": im.get("neg_category_ids", [])} for im in data["images"]}
    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append({"id": a["id"], "cat": a["category_id"], "bbox": a["bbox"],
                                    "seg": a["segmentation"], "area": a["area"]})
    with out.open("wb") as f:
        pickle.dump({"images": images, "anns": dict(anns), "cats": cats}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"LVIS {split}: {len(images)} images, {len(data['annotations'])} annotations")


def fetch_coco_extra():
    out = DIR / "coco_extra.pkl"
    if out.exists():
        return
    extra = defaultdict(lambda: {"crowd": set(), "traffic_light": []})
    with zipfile.ZipFile(_download(COCO_ANN_ZIP, "COCO 2017 annotations")) as z:
        for split in ("train", "val"):
            with z.open(f"annotations/instances_{split}2017.json") as f:
                data = json.load(f)
            names = {c["id"]: c["name"] for c in data["categories"]}
            for a in data["annotations"]:
                name = names[a["category_id"]]
                if a.get("iscrowd"):
                    extra[a["image_id"]]["crowd"].add(name)
                if name == "traffic light":
                    extra[a["image_id"]]["traffic_light"].append(a["bbox"])
    with out.open("wb") as f:
        pickle.dump({k: {"crowd": sorted(v["crowd"]), "traffic_light": v["traffic_light"]} for k, v in extra.items()}, f)


def fetch_all():
    for split in LVIS_URLS:
        fetch_lvis(split)
    fetch_coco_extra()


@lru_cache(maxsize=2)
def lvis(split):
    with (DIR / f"lvis_{split}.pkl").open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def coco_extra():
    with (DIR / "coco_extra.pkl").open("rb") as f:
        return pickle.load(f)


def image_file(image_id):
    return IMAGES / f"{image_id:012d}.jpg"


def download_images(items, desc="COCO images"):
    """items: [(image_id, url)]. Returns the set of ids available locally."""
    IMAGES.mkdir(parents=True, exist_ok=True)

    def get(item):
        image_id, url = item
        path = image_file(image_id)
        if path.exists():
            return image_id
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(path)
            return image_id
        except Exception:
            return None

    items = sorted(set(items))
    with ThreadPoolExecutor(32) as ex:
        return {i for i in tqdm(ex.map(get, items), total=len(items), desc=desc, unit="img") if i is not None}


@lru_cache(maxsize=64)
def load_image(image_id):
    return Image.open(image_file(image_id)).convert("RGB")
