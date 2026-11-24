"""Replace exact text runs in the status deck: python tools/deck/edit_text.py '{"old": "new", ...}'."""
import json, os, sys
from pptx import Presentation
path = os.path.expanduser("~/Documents/research/counterpoint_status_2026-11-10.pptx")
repl = json.loads(sys.argv[1])
prs = Presentation(path); n = 0
for slide in prs.slides:
    for sh in slide.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.text in repl:
                        r.text = repl[r.text]; n += 1
prs.save(path); print("replaced", n)
