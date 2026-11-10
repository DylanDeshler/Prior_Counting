"""GRPO reward functions for ms-swift (loaded with --external_plugins tools/rl_rewards.py).

Rows come from exports/<arm>/<model>/<format>/train_rl.jsonl: answer (true count), points (ground-truth
unit points, model coordinates) and unit_size (same coordinates). Rewards, all in [0, 1]:
  count_acc          1 if the stated total ({N}, or a final "Total: N" line) equals the true count
  point_f1           F1 of predicted points vs ground-truth points, one-to-one matched within a radius of
                     max(25, 0.6 x unit size) on the 0-1000 grid: rewards pointing at the real units, so a
                     right total from a wrong list earns less than a right list
  points_consistent  1 if there is a point list and its length equals the stated total (format/agreement)
The familiar answer is never used: the reward is correctness only, so the policy cannot be paid for simply
moving away from the prior (the C-blind shortcut).
"""

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from answer_parsing import extract_answer  # noqa: E402
from swift.rewards import ORM, orms  # noqa: E402

POINT_RE = re.compile(r'"?point_2d"?\s*:\s*\[\s*"?(-?\d+(?:\.\d+)?)"?\s*,\s*"?(-?\d+(?:\.\d+)?)"?\s*\]')


def _num(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def _points(text):
    return [(float(x), float(y)) for x, y in POINT_RE.findall(text.split("</think>")[-1])]


def match_f1(pred, gt, radius):
    """Greedy one-to-one matching by distance (close to Hungarian for these sparse sets)."""
    if not pred or not gt:
        return 0.0
    pairs = sorted((math.dist(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt))
    used_p, used_g, hits = set(), set(), 0
    for d, i, j in pairs:
        if d > radius:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); hits += 1
    return 2 * hits / (len(pred) + len(gt))


def _col(kwargs, name, n):
    v = kwargs.get(name)
    return list(v) if v is not None else [None] * n


class CountAcc(ORM):
    def __call__(self, completions, **kwargs):
        answers = _col(kwargs, "answer", len(completions))
        return [float(_num(extract_answer(c)) is not None and _num(extract_answer(c)) == _num(a))
                for c, a in zip(completions, answers)]


class PointF1(ORM):
    def __call__(self, completions, **kwargs):
        gts = _col(kwargs, "points", len(completions))
        sizes = _col(kwargs, "unit_size", len(completions))
        out = []
        for c, gt, size in zip(completions, gts, sizes):
            gt = [tuple(p) for p in (gt or [])]
            out.append(match_f1(_points(c), gt, max(25.0, 0.6 * float(size or 0))))
        return out


class PointsConsistent(ORM):
    def __call__(self, completions, **kwargs):
        res = []
        for c in completions:
            n, total = len(_points(c)), _num(extract_answer(c))
            res.append(float(n > 0 and total == n))
        return res


orms["count_acc"] = CountAcc
orms["point_f1"] = PointF1
orms["points_consistent"] = PointsConsistent
