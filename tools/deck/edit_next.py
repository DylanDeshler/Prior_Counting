"""Edit the status deck in place (the original builder scripts lived in /tmp and were lost)."""
import os
from pptx import Presentation
path = os.path.expanduser("~/Documents/research/counterpoint_status_2026-11-10.pptx")
prs = Presentation(path)
repl = {
    "RL from the pointing prompt (running)": "Per-item diagnosis",
    "Rewards: correct total + points on real legs; never the familiar answer":
        "Image vs subject-word attention, knockouts, logit attribution, patching trained into untrained",
    "Steering, round 2": "RL, then steering round 2",
    "0.1× to 10× state size (+ reversed) at every text position; base and leg-trained":
        "GRPO from the pointing prompt; steering at 0.1×–10× state size (+ reversed)",
}
n = 0
for slide in prs.slides:
    for sh in slide.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.text in repl:
                        r.text = repl[r.text]; n += 1
prs.save(path); print("replaced", n)
