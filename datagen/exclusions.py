"""Leakage controls for photo-derived data (§9).

    coco_image_ids.json   COCO image ids used by eval sets (POPE, ORIC). Excluded by id, not split name:
                          COCO val2014 ids reappear in train2017 (§9.2).
    eval_phash.npz        64-bit pHashes of evaluation / probe images (VLMBias, Visual CounterFact,
                          CountBenchQA, PixMo-Count val+test, MMStar, POPE). Photo-derived training
                          images within Hamming distance <= 6 of any of them are dropped (§9.4).
"""

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import imagehash
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from .common import DATA

EXCLUSIONS = DATA / "exclusions"
POPE = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{}.json"
ORIC = "https://raw.githubusercontent.com/ZhaoyangLi-1/ORIC/main/dataset/oric_{}.json"
COCO_VAL2014 = "http://images.cocodataset.org/val2014/{}"
PHASH_MAX_DIST = 6


def coco_id(filename):
    return int(re.search(r"(\d{6,})", filename).group(1))


def fetch_coco_ids():
    ids, pope_files = {}, set()
    for kind in ("random", "popular", "adversarial"):
        for line in requests.get(POPE.format(kind), timeout=60).text.splitlines():
            if line.strip():
                f = json.loads(line)["image"]
                pope_files.add(f)
                ids[coco_id(f)] = "pope"
    for kind in ("bench", "train"):
        for item in requests.get(ORIC.format(kind), timeout=60).json():
            ids.setdefault(coco_id(item["image"]), "oric")
    EXCLUSIONS.mkdir(parents=True, exist_ok=True)
    (EXCLUSIONS / "coco_image_ids.json").write_text(json.dumps({"ids": sorted(ids), "source": {str(k): v for k, v in sorted(ids.items())}}))
    print(f"excluded COCO image ids: {len(ids)} (POPE {sum(v == 'pope' for v in ids.values())}, ORIC {sum(v == 'oric' for v in ids.values())})")
    return sorted(pope_files)


@lru_cache(maxsize=1)
def excluded_coco_ids():
    return set(json.loads((EXCLUSIONS / "coco_image_ids.json").read_text())["ids"])


def phash_int(img):
    return int(str(imagehash.phash(img.convert("RGB"))), 16)


def _hf_images(repo, split, columns, config=None):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    for row in tqdm(ds, desc=f"{repo}:{split}", unit="img"):
        for c in columns:
            if row.get(c) is not None:
                yield row[c]


def _url_images(urls, desc):
    def get(u):
        try:
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content))
        except Exception:
            return None
    with ThreadPoolExecutor(32) as ex:
        for im in tqdm(ex.map(get, urls), total=len(urls), desc=desc, unit="img"):
            if im is not None:
                yield im


def fetch_eval_phashes(pope_files):
    from datasets import load_dataset
    hashes, sources = [], []

    def add(images, name):
        for im in images:
            hashes.append(phash_int(im))
            sources.append(name)

    add(_hf_images("anvo25/vlms-are-biased", "main", ["image"]), "vlmbias")
    add(_hf_images("anvo25/vlms-are-biased", "original", ["image"]), "vlmbias")
    add(_hf_images("mgolov/Visual-Counterfact", "color", ["original_image", "counterfact_image"]), "visual_counterfact")
    add(_hf_images("mgolov/Visual-Counterfact", "size", ["original_image", "counterfact_image"]), "visual_counterfact")
    add(_hf_images("vikhyatk/CountBenchQA", "test", ["image"]), "countbenchqa")
    add(_hf_images("Lin-Chen/MMStar", "val", ["image"], config="val"), "mmstar")
    for split in ("validation", "test"):
        urls = load_dataset("allenai/pixmo-count", split=split)["image_url"]
        add(_url_images(urls, f"pixmo-count:{split}"), "pixmo_count")
    add(_url_images([COCO_VAL2014.format(f) for f in pope_files], "POPE images"), "pope")
    arr = np.array(hashes, dtype=np.uint64)
    np.savez_compressed(EXCLUSIONS / "eval_phash.npz", hashes=arr, sources=np.array(sources))
    print(f"eval pHashes: {len(arr)} images")


def fetch_all(refresh=False):
    if not refresh and (EXCLUSIONS / "coco_image_ids.json").exists() and (EXCLUSIONS / "eval_phash.npz").exists():
        print("eval-set exclusions + pHashes: up to date")
        return
    pope_files = fetch_coco_ids()
    fetch_eval_phashes(pope_files)


@lru_cache(maxsize=1)
def _eval_hashes():
    path = EXCLUSIONS / "eval_phash.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m datagen sources` (photo levels need it, §9.4)")
    return np.load(path)["hashes"]


def nearest_eval_distance(img):
    """Hamming distance from img's pHash to the closest evaluation image."""
    h = np.uint64(phash_int(img))
    x = np.bitwise_xor(_eval_hashes(), h)
    # popcount of 64-bit ints
    bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
    return int(bits.min())
