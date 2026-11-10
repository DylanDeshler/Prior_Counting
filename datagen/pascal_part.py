"""Pascal-Part (Chen et al., CVPR 2014): real photos with every visible body part / wheel segmented
separately, so countable parts (legs, wheels, wings, arms) have one mask each.

Used for real-photo PART counting, the case VLMBias tests (legs of an animal) and neither the synthetic
families (rendered shapes) nor the LVIS pool (separate whole objects) covers:
  - parts/count     unedited crops of one animal; answer = number of VISIBLE legs (often fewer than 4)
  - parts/conflict  edit pairs on fully visible animals: one image gets a leg added (copy-paste of the
                    animal's own leg) or removed (inpainted); its partner gets the same kind of edit on the
                    background (a null edit), so "edited" never predicts "count changed" (C-blind lesson)
  - tests/P/...     the same, from held-out images (20% by image-id hash)
Questions ask how many legs are VISIBLE: in an unedited photo an occluded leg still exists, so "how many
legs does it have" would have two right answers.

Images: PASCAL VOC 2010 trainval (Flickr; VOC terms). Annotations: Pascal-Part trainval.tar.gz (no license
stated; cite Chen et al. 2014).
"""

import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from .common import REPORTS, SOURCES

DIR = SOURCES / "pascal_part"
ANN = DIR / "Annotations_Part"
IMAGES = DIR / "JPEGImages"
ANN_URL = "http://roozbehm.info/pascal-parts/trainval.tar.gz"
VOC_URLS = ["https://thor.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"]

# Countable part types: class -> unit -> (canonical count, {position: [segment names that make up that unit]}).
# A unit is visible if any of its segments is annotated (e.g. only the hoof of a leg still counts as a leg).
_QUAD_POS = ("lf", "rf", "lb", "rb")
UNITS = {
    "cat": {"leg": (4, {p: [f"{p}leg", f"{p}pa"] for p in _QUAD_POS}), "eye": (2, {"l": ["leye"], "r": ["reye"]}),
            "ear": (2, {"l": ["lear"], "r": ["rear"]})},
    "dog": {"leg": (4, {p: [f"{p}leg", f"{p}pa"] for p in _QUAD_POS}), "eye": (2, {"l": ["leye"], "r": ["reye"]}),
            "ear": (2, {"l": ["lear"], "r": ["rear"]})},
    "cow": {"leg": (4, {p: [f"{p}uleg", f"{p}lleg"] for p in _QUAD_POS}), "horn": (2, {"l": ["lhorn"], "r": ["rhorn"]}),
            "ear": (2, {"l": ["lear"], "r": ["rear"]})},
    "sheep": {"leg": (4, {p: [f"{p}uleg", f"{p}lleg"] for p in _QUAD_POS}), "ear": (2, {"l": ["lear"], "r": ["rear"]})},
    "horse": {"leg": (4, {p: [f"{p}uleg", f"{p}lleg", f"{p}ho"] for p in _QUAD_POS}), "ear": (2, {"l": ["lear"], "r": ["rear"]})},
    "bird": {"leg": (2, {"l": ["lleg", "lfoot"], "r": ["rleg", "rfoot"]}), "wing": (2, {"l": ["lwing"], "r": ["rwing"]})},
    "person": {"leg": (2, {"l": ["luleg", "llleg", "lfoot"], "r": ["ruleg", "rlleg", "rfoot"]}),
               "arm": (2, {"l": ["luarm", "llarm", "lhand"], "r": ["ruarm", "rlarm", "rhand"]})},
    "bicycle": {"wheel": (2, {"f": ["fwheel"], "b": ["bwheel"]})},
    "motorbike": {"wheel": (2, {"f": ["fwheel"], "b": ["bwheel"]})},
    "car": {"wheel": (4, None)},  # wheel_1..wheel_n: one unit per numbered wheel
}
ANIMALS = ("cat", "dog", "cow", "sheep", "horse", "bird")  # leg counting for the training pools


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def fetch(refresh=False):
    """Annotations (~80 MB) and the VOC 2010 trainval images that have part annotations (~1.3 GB tar)."""
    from .coco import _download
    DIR.mkdir(parents=True, exist_ok=True)
    if refresh or not ANN.exists() or not any(ANN.iterdir()):
        with tarfile.open(fileobj=_download(ANN_URL, "Pascal-Part annotations"), mode="r:gz") as t:
            t.extractall(DIR, members=[m for m in t.getmembers() if m.name.startswith("Annotations_Part/")])
    wanted = {p.stem for p in ANN.glob("*.mat")}
    IMAGES.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in IMAGES.glob("*.jpg")}
    if wanted - have:
        import requests
        for url in VOC_URLS:  # stream the tar and keep only the annotated JPEGs
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with tarfile.open(fileobj=r.raw, mode="r|") as t:
                        for m in tqdm(t, desc="VOC2010 images", unit="file"):
                            name = m.name.rsplit("/", 1)[-1]
                            if "/JPEGImages/" in m.name and name[:-4] in wanted - have:
                                (IMAGES / name).write_bytes(t.extractfile(m).read())
                break
            except Exception as e:  # try the mirror
                print(f"VOC download from {url} failed: {e}")
    missing = wanted - {p.stem for p in IMAGES.glob("*.jpg")}
    print(f"Pascal-Part: {len(wanted)} annotated images, {len(missing)} without a local JPEG")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x.ravel())
    return [x]


def load(image_id):
    """[{class, mask, bbox, units: {unit: {position: mask}}}] for one image (masks are bool arrays)."""
    import scipy.io as sio
    m = sio.loadmat(ANN / f"{image_id}.mat", squeeze_me=True, struct_as_record=False)
    objs = []
    for o in _as_list(m["anno"].objects):
        cls = str(o.__dict__["class"])
        mask = np.asarray(o.mask, bool)
        parts = {str(p.part_name): np.asarray(p.mask, bool) for p in _as_list(o.parts)}
        units = {}
        for unit, (_, positions) in UNITS.get(cls, {}).items():
            if positions is None:  # numbered parts, e.g. wheel_1..wheel_n
                found = {k: v for k, v in parts.items() if k.startswith(unit + "_") and v.any()}
            else:
                found = {}
                for pos, names in positions.items():
                    segs = [parts[n] for n in names if n in parts and parts[n].any()]
                    if segs:
                        found[pos] = np.logical_or.reduce(segs)
            units[unit] = found
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, 0, 0]
        objs.append({"class": cls, "mask": mask, "bbox": bbox, "units": units, "has_parts": bool(parts)})
    return objs


def image_ids():
    return sorted(p.stem for p in ANN.glob("*.mat"))


def split_of(image_id, test_frac=0.2):
    return "test" if int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % 100 < test_frac * 100 else "train"


# ---------------------------------------------------------------------------
# Distribution stats: visible part counts vs the canonical count
# ---------------------------------------------------------------------------

def stats(write=True):
    """Visible-count distribution per (class, unit) over objects that have any part annotation, plus the
    share of objects showing exactly the canonical count. Pascal-Part only labels visible parts, so
    counts below canonical are occlusion/pose; counts above canonical would be annotation errors."""
    dist = defaultdict(Counter)
    n_obj = Counter()
    for iid in tqdm(image_ids(), desc="Pascal-Part stats", unit="img"):
        for o in load(iid):
            if o["class"] not in UNITS or not o["has_parts"]:
                continue
            n_obj[o["class"]] += 1
            for unit, found in o["units"].items():
                dist[(o["class"], unit)][len(found)] += 1
    rows = []
    for (cls, unit), c in sorted(dist.items()):
        canon = UNITS[cls][unit][0]
        n = sum(c.values())
        rows.append({"class": cls, "unit": unit, "canonical": canon, "objects": n,
                     "eq_canonical": c[canon] / n, "below": sum(v for k, v in c.items() if k < canon) / n,
                     "above": sum(v for k, v in c.items() if k > canon) / n,
                     "mean_visible": sum(k * v for k, v in c.items()) / n,
                     "dist": {int(k): v for k, v in sorted(c.items())}})
    if write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "pascal_part_stats.json").write_text(json.dumps(rows, indent=1))
        (REPORTS / "pascal_part_stats.md").write_text(stats_markdown(rows))
    return rows


def stats_markdown(rows):
    lines = ["# Pascal-Part: visible part counts vs canonical (objects with part annotations)", "",
             "| class | part | canonical | objects | = canonical | below | above | mean visible | distribution (count: objects) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = ", ".join(f"{k}: {v}" for k, v in r["dist"].items())
        lines.append(f"| {r['class']} | {r['unit']} | {r['canonical']} | {r['objects']} | {100 * r['eq_canonical']:.1f}% | "
                     f"{100 * r['below']:.1f}% | {100 * r['above']:.1f}% | {r['mean_visible']:.2f} | {d} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(stats_markdown(stats()))
