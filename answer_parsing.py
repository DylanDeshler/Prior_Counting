"""Answer parsing shared by the eval harness (run_vlms_are_biased.py) and the data checks.
Dependency-free so both environments can import it."""

import re


def normalize(s):
    return str(s).strip().strip("{}").strip().rstrip(".").lower()


def extract_answer(text):
    """Return the content of the last {...} in the response (after any thinking block)."""
    text = text.split("</think>")[-1]
    matches = re.findall(r"\{([^{}]*)\}", text)
    return matches[-1].strip() if matches else ""


def matches(pred, target):
    p, t = normalize(pred), normalize(target)
    if p == t:
        return True
    # Numeric answers: compare as numbers so e.g. "9 rows" or "09" count.
    pn, tn = re.findall(r"\d+", p), re.findall(r"\d+", t)
    return len(pn) == 1 and len(tn) == 1 and int(pn[0]) == int(tn[0])
