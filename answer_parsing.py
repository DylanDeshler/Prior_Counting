"""Answer parsing shared by the eval harness (run_vlms_are_biased.py) and the data checks.
Dependency-free so both environments can import it."""

import re


def normalize(s):
    return str(s).strip().strip("{}").strip().rstrip(".").lower()


BARE = re.compile(r"^\s*([A-Za-z]+|\d+)\s*\.?\s*$")


def extract_answer(text):
    """Content of the last {...} answer in the response (after any thinking block). If there are none,
    accept a final 'Total: N' line (point-total prompt), or a reply that is nothing but the answer ("6",
    "red"); never pick numbers out of prose."""
    text = text.split("</think>")[-1]
    # Ignore JSON objects (e.g. {"point_2d": [x, y], "label": "pip"} in point lists): they are not answers.
    matches = [m for m in re.findall(r"\{([^{}]*)\}", text) if '"' not in m and ":" not in m]
    if matches:
        return matches[-1].strip()
    # The point-total prompt asks for a last line 'Total: {N}'; the model often drops the braces.
    total = re.findall(r"(?im)^\W*total\W*:\s*\**\s*(\d+)\b", text)
    if total:
        return total[-1]
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
    # Prompt study (does instruction alone counter the prior?). {part} is the counted noun from the question.
    # plain chain of thought
    "cot": " Think step by step, then give the total number in curly brackets, e.g., {9}.",
    # explicit de-biasing instruction, no reasoning requested
    "debias": (" Count each and every one of the {part} individually, exactly as they appear in this image. The image may "
               "have been edited, so the usual number for this kind of object can be wrong: ignore what you expect and "
               "report only what you see. Give the total number in curly brackets, e.g., {9}."),
    # step-by-step enumeration of each part + edit check + ignore the prior
    "cot-debias": (" Think step by step. I need to count each and every one of the {part} and intentionally ignore my bias "
                   "about how many this kind of object usually has. Go over the image and number each of the {part} one at "
                   "a time, saying where it is. Check whether anything looks unusual or edited. Then give the total number "
                   "in curly brackets, e.g., {9}."),
    # pointing (point-total) with the de-biasing instruction in front
    "point-debias": (" The image may have been edited, so ignore the usual number for this kind of object. Locate each and "
                     'every one of the {part}: list them as JSON, [{{"point_2d": [x, y], "label": "item"}}, ...], with x '
                     'and y on a 0-1000 scale. After the list, always finish with a final line "Total: {{N}}" giving the '
                     "number of items in curly brackets, e.g., Total: {{9}}."),
}
STYLE_TAG = {"default": "", "point": "+point", "point-total": "+pointT", "direct": "+direct", "cot": "+cot",
             "debias": "+debias", "cot-debias": "+cotDebias", "point-debias": "+pointDebias"}
_PART = re.compile(r"(?i)\b(?:how many|count the(?: number of)?|number of)\s+([a-z][a-z\- ]*?)\s+"
                   r"(?:are|is|does|do|can|in|on|of|there|visible|appear|have)\b")


def counted_part(prompt):
    """The counted noun of a counting question ('legs', 'stars', 'chess pieces'); 'items' if not found."""
    m = _PART.search(prompt)
    return m.group(1).strip() if m else "items"


def rewrite_prompt(prompt, style):
    if style == "default" or "number in curly brackets" not in prompt:
        return prompt
    suffix = PROMPT_SUFFIXES[style]
    if "{part}" in suffix:  # templated styles: fill the noun; literal braces are doubled in the template
        suffix = suffix.replace("{{", "\x00").replace("}}", "\x01").replace("{9}", "\x029").replace("{part}", counted_part(prompt))
        suffix = suffix.replace("\x00", "{").replace("\x01", "}").replace("\x029", "{9}")
    return prompt.split(" Answer with")[0] + suffix
