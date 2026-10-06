"""Job builders (for items.run_job) for Exp 2 L0 color pairs, T0 color, and neutral emoji crowds."""

from collections import Counter

from . import colornames, colorsets
from .common import item_seed, rng_for
from .families.emoji import CROWD_CATEGORIES, color_objects, fetch_crowd_assets

TEMPLATES = ("how_many", "count_the")


def fetch_all():
    colornames.fetch()
    colorsets.fetch()
    fetch_crowd_assets()


def _balanced_cycle(names, n, salt):
    """n names: each round is a fresh seeded permutation, so every prefix uses objects evenly."""
    out, r = [], 0
    while len(out) < n:
        perm = rng_for(item_seed(salt, r)).permutation(len(names))
        out += [names[i] for i in perm]
        r += 1
    return out[:n]


def _color_jobs(objects, n_pairs, dataset, out, data_type, mode, roles):
    order = _balanced_cycle(sorted(objects), n_pairs, dataset)
    return [{"kind": "pair", "generator": "emoji_color", "prior": "color", "family": obj.replace(" ", "_"),
             "dataset": dataset, "id_prefix": dataset.replace("_", "-").lower(), "idx": i, "pool_index": i,
             "template": TEMPLATES[i % 2], "data_type": data_type, "out": out, "roles": roles, "delta": 0,
             "sample_kw": {"object": obj, "mode": mode}}
            for i, obj in enumerate(order)]


def e2_color_jobs(n_pairs, data_type):
    """Exp 2 L0 color pairs. conflict: CoDa Single train objects (canonical/counterfactual);
    neutral: CoDa Any objects (original/edited)."""
    objs = color_objects()
    if data_type == "conflict":
        names = [n for n, o in objs.items() if o["group"] == "single" and o["split"] == "train"]
        return _color_jobs(names, n_pairs, "e2_L0_color_conflict", "e2/L0/color", "conflict", "conflict",
                           ("canonical", "counterfactual"))
    names = [n for n, o in objs.items() if o["group"] == "any"]
    return _color_jobs(names, n_pairs, "e2_L0_color_neutral", "e2/L0/color", "neutral", "neutral",
                       ("original", "edited"))


def t0_color_jobs(n_pairs=500):
    objs = color_objects()
    names = [n for n, o in objs.items() if o["group"] == "single" and o["split"] == "heldout"]
    return _color_jobs(names, n_pairs, "t0_color", "tests/T0/color", "conflict", "conflict",
                       ("canonical", "counterfactual"))


# ---------------------------------------------------------------------------
# Neutral crowds matched to the pooled conflict count histogram (§5.2)
# ---------------------------------------------------------------------------

def match_counts(counts, seed, tries=2000):
    """Pair up a multiset of counts so every pair differs by 1-3. Returns [(a, b)] or None."""
    for t in range(tries):
        rng = rng_for(item_seed("crowd-match", seed, t))
        left = Counter(counts)
        pairs = []
        ok = True
        while sum(left.values()):
            # Most constrained value first: fewest available partners.
            vals = [v for v in left if left[v]]
            partners = {v: sum(left[w] - (w == v) for w in vals if 1 <= abs(w - v) <= 3) for v in vals}
            v = min(vals, key=lambda x: (partners[x], rng.random()))
            left[v] -= 1
            opts = [w for w in left if left[w] and 1 <= abs(w - v) <= 3]
            if not opts:
                ok = False
                break
            weights = [left[w] for w in opts]
            w = opts[int(rng.choice(len(opts), p=[x / sum(weights) for x in weights]))]
            left[w] -= 1
            pairs.append((v, w))
        if ok:
            return pairs
    return None


def e2_crowd_jobs(conflict_records, n_pairs=None, block=64):
    """Neutral crowd pairs whose original+edited counts reproduce the pooled count multiset of the
    conflict count pairs. Matching is done per block of `block` conflict pairs (by pool_index), so
    every prefix that is a multiple of `block` pairs is matched exactly."""
    by_pair = {}
    for r in conflict_records:
        by_pair.setdefault(r["pool_index"], []).append(r["answer"])
    keys = sorted(by_pair)
    if n_pairs is not None:
        keys = keys[:n_pairs]
    pairs = []
    start = 0
    while start < len(keys):
        size = block
        while True:
            chunk = keys[start:start + size]
            matched = match_counts([c for k in chunk for c in by_pair[k]], seed=(start, size))
            if matched is not None or start + size >= len(keys):
                break
            size *= 2  # fall back to a bigger block if this one has no valid matching
        if matched is None:
            raise ValueError(f"cannot match crowd counts for pairs {start}..{start + size}")
        pairs += matched
        start += size
    cats = _balanced_cycle(sorted(CROWD_CATEGORIES), len(pairs), "crowd-categories")
    jobs = []
    for i, ((a, b), cat) in enumerate(zip(pairs, cats)):
        if rng_for(item_seed("crowd-sign", i)).random() < 0.5:
            a, b = b, a
        jobs.append({"kind": "pair", "generator": "emoji_crowd", "family": "crowd",
                     "dataset": "e2_L0_count_neutral", "id_prefix": "e2-l0-count-neutral", "idx": i,
                     "pool_index": i, "template": TEMPLATES[i % 2], "data_type": "neutral",
                     "out": "e2/L0/count", "roles": ("original", "edited"), "count": a, "delta": b - a,
                     "sample_kw": {"category": cat}})
    return jobs
