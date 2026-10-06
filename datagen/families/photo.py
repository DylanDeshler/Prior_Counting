"""L2 color generator (see photo_color.py for the pipeline)."""

import hashlib

import cv2
import numpy as np

from ..colorsets import load as load_color_objects
from ..common import IMG
from ..photo_color import FEATHER, K, VERIFY_MIN, _key, color_region, crop_mask, name_stats
from .base import COLOR_SUFFIX, Family


class PhotoColor(Family):
    name, unit = "photo_color", "object"
    fixed_frame = True
    templates = {"how_many": "What color is the {object}?", "count_the": "Name the color of the {object}."}

    def label_for(self, p):
        return None

    def question(self, p, template):
        return self.templates[template].format(object=p["object"]) + COLOR_SUFFIX

    def sample(self, rng, count=None, cand=None, mode="conflict", variant=0):
        return {"object": cand["object"], "image_id": cand["image_id"], "ann_id": cand["ann_id"], "crop": cand["crop"],
                "seg": cand["seg"], "mode": mode, "variant": variant, "ab": None, "answer": cand["current"],
                "targets": cand["targets"], "colors": {"main": [128, 128, 128]}}

    def counterfactual(self, p, delta, rng):
        o = load_color_objects()[p["object"]]
        terms = sorted(p["targets"])
        if p["mode"] == "conflict":  # variants of one photo get different colors
            start = int(hashlib.sha256(str(p["ann_id"]).encode()).hexdigest(), 16) % len(terms)
            target = terms[(start + p["variant"]) % len(terms)]
        else:  # neutral: sample from the object's own CoDa distribution over reachable colors
            probs = np.array([o["coda"][t] for t in terms])
            target = terms[int(rng.choice(len(terms), p=probs / probs.sum()))]
        ab = p["targets"][target]
        return ({**p, "ab": ab, "answer": target},
                {"op": "recolor", "params": {"from": p["answer"], "to": target, "ab": ab, "k": K, "feather": FEATHER}})

    def null_edit(self, p):
        return {"op": "identity_recolor", "params": {"k": K, "feather": FEATHER}, "null_edit": True}

    def scene(self, p):
        mask = crop_mask(p["seg"], p["crop"])
        ys, xs = np.nonzero(mask)
        depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        yi, xi = np.unravel_index(int(np.argmax(depth)), depth.shape)  # deepest point: always inside
        unit = {"point": (float(xi), float(yi)), "mask": [("image", _key(p, mask_only=True), (IMG / 2, IMG / 2), IMG)],
                "size": float(min(np.ptp(xs) + 1, np.ptp(ys) + 1))}
        return [("image", _key(p), (IMG / 2, IMG / 2), IMG)], [unit]

    def object_bbox(self, p):
        """Edit region: the mask's box grown by the feathering reach (4 sigma)."""
        ys, xs = np.nonzero(crop_mask(p["seg"], p["crop"]))
        pad = int(4 * FEATHER) + 1
        return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(IMG, int(xs.max()) + 1 + pad),
                min(IMG, int(ys.max()) + 1 + pad)]

    def colors(self, p):
        return [(128, 128, 128)]

    def verify_color(self, img, p, aff):
        rgb = np.asarray(img)
        name, share, _ = name_stats(rgb, color_region(rgb, crop_mask(p["seg"], p["crop"])))
        return name if share >= VERIFY_MIN else f"mixed ({name} covers {share:.0%})"

    def post_check(self, img, rec):
        import os
        from ..exclusions import EXCLUSIONS, PHASH_MAX_DIST, nearest_eval_distance
        if os.environ.get("COUNTERPOINT_PREVIEW") == "1" and not (EXCLUSIONS / "eval_phash.npz").exists():
            rec["checks"]["phash_min_dist"] = None  # previews may run before the eval-image hashes exist
            return
        d = nearest_eval_distance(img)
        rec["checks"]["phash_min_dist"] = d
        if d <= PHASH_MAX_DIST:
            from ..items import Unusable
            raise Unusable(f"near-duplicate of an eval image (pHash distance {d})")
