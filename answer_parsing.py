"""Answer parsing shared by the eval harness (run_vlms_are_biased.py) and the data checks.
Dependency-free so both environments can import it."""

import re


def normalize(s):
    return str(s).strip().strip("{}").strip().rstrip(".").lower()


BARE = re.compile(r"^\s*([A-Za-z]+|\d+)\s*\.?\s*$")


def extract_answer(text):
    """Content of the last {...} answer in the response (after any thinking block). If there are none,
    accept a reply that is nothing but the answer ("6", "red"); never pick numbers out of prose."""
    text = text.split("</think>")[-1]
    # Ignore JSON objects (e.g. {"point_2d": [x, y], "label": "pip"} in point lists): they are not answers.
    matches = [m for m in re.findall(r"\{([^{}]*)\}", text) if '"' not in m and ":" not in m]
    if matches:
        return matches[-1].strip()
    bare = BARE.match(text)
    return bare.group(1) if bare else ""


def matches(pred, target):
    p, t = normalize(pred), normalize(target)
    if p == t:
        return True
    # Numeric answers: compare as numbers so e.g. "9 rows" or "09" count.
    pn, tn = re.findall(r"\d+", p), re.findall(r"\d+", t)
    return len(pn) == 1 and len(tn) == 1 and int(pn[0]) == int(tn[0])


# ---------------------------------------------------------------------------
# Prompt variants for counting questions (shared by eval and training exports so wording matches).
# Questions not asking for a number (yes/no illusions, identification) are left unchanged.
# ---------------------------------------------------------------------------

PROMPT_SUFFIXES = {
    # Gate 1 baseline (original wording; the model usually lists points but omits the total)
    "point": (' First locate every one of them: list them as JSON, [{"point_2d": [x, y], "label": "item"}, ...], '
              'with x and y on a 0-1000 scale. Then give the total number in curly brackets, e.g., {9}.'),
    # pointing that must end with a stated total, so list length and stated total can be compared
    "point-total": (' First locate every one of them: list them as JSON, [{"point_2d": [x, y], "label": "item"}, ...], '
                    'with x and y on a 0-1000 scale. After the list, always finish with a final line '
                    '"Total: {N}" giving the number of items in curly brackets, e.g., Total: {9}.'),
    # answer directly, no listing or reasoning (tests whether grounding was distilled into the direct answer)
    "direct": " Answer immediately with only the number in curly brackets, e.g., {9}. Do not list, point or explain.",
}
STYLE_TAG = {"default": "", "point": "+point", "point-total": "+pointT", "direct": "+direct"}


def rewrite_prompt(prompt, style):
    if style == "default" or "number in curly brackets" not in prompt:
        return prompt
    return prompt.split(" Answer with")[0] + PROMPT_SUFFIXES[style]
