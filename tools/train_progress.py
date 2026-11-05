"""Compact live view of an ms-swift training run: one status line, overwritten in place.

    swift sft ... 2>&1 | python tools/train_progress.py runs/<arm>/train_raw.log

  train  57/512  11%  loss 0.412  acc 0.871  grad 0.83  lr 9.8e-05  ep 0.22  mem 21.3G  3.6s/it  ETA 27:27

The full raw output (every log line, warning and progress bar) goes to the given log file. Evaluation
losses, checkpoint saves and anything unexpected (errors, tracebacks) are printed as normal lines above
the status line; known-harmless notices are only kept in the raw log.
"""

import ast
import re
import sys
import time

TQDM = re.compile(r"(Train|Val)[^|]*?(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.?]+)\s*(s/it|it/s)")
LOGDICT = re.compile(r"\{'(?:loss|eval_loss|train_runtime)'.*\}")
HARMLESS = ("None of the inputs have requires_grad", "check_backward_validity", "TRITON_DEFAULT_FP_FUSION",
            "warn_once(FP_FUSION", "falling back to its reference PyTorch implementation", "UserWarning: ",
            "use_logits_to_keep")
CLEAR = "\r\x1b[2K"


class View:
    def __init__(self, raw_path):
        self.raw = open(raw_path, "a", buffering=1)
        self.s = {}
        self.last_draw = 0.0

    def status(self):
        s = self.s
        if "step" not in s:
            return ""
        parts = [f"{s.get('phase', 'train'):<5} {s['step']}/{s['total']}  {100 * s['step'] / max(1, s['total']):3.0f}%"]
        for key, label, fmt in (("loss", "loss", "{:.4f}"), ("token_acc", "acc", "{:.3f}"), ("grad_norm", "grad", "{:.2f}"),
                                ("learning_rate", "lr", "{:.1e}"), ("epoch", "ep", "{:.2f}"), ("memory(GiB)", "mem", "{:.1f}G")):
            if key in s:
                try:
                    parts.append(f"{label} {fmt.format(float(s[key]))}")
                except (TypeError, ValueError):
                    pass
        if "rate" in s:
            parts.append(s["rate"])
        if "eta" in s:
            parts.append(f"ETA {s['eta']}")
        return "  ".join(parts)

    def draw(self, force=False):
        now = time.time()
        if force or now - self.last_draw > 0.5:
            sys.stdout.write(CLEAR + self.status())
            sys.stdout.flush()
            self.last_draw = now

    def say(self, line):
        """A permanent line above the status line."""
        sys.stdout.write(CLEAR + line.rstrip() + "\n")
        self.draw(force=True)

    def feed(self, raw):
        seg = raw.strip()
        if not seg:
            return
        if "Saving model checkpoint" in seg:  # often printed right after a progress bar, on the same line
            self.say("saved  " + seg.split("checkpoint to")[-1].strip())
        m = TQDM.search(seg)
        if m:
            phase, _, step, total, _, eta, rate, unit = m.groups()
            if phase == "Train":
                self.s.update(phase="train", step=int(step), total=int(total), eta=eta, rate=f"{rate}{unit}")
            else:
                self.s["phase"] = "eval"
            self.draw()
            return
        d = LOGDICT.search(seg)
        if d:
            try:
                logs = ast.literal_eval(d.group(0))
            except (ValueError, SyntaxError):
                logs = {}
            if "eval_loss" in logs:
                self.s["phase"] = "train"
                acc = logs.get("eval_token_acc")
                self.say(f"eval   step {self.s.get('step', '?')}: eval_loss {logs['eval_loss']:.4f}"
                         + (f"  eval_acc {acc:.3f}" if acc is not None else ""))
            elif "train_runtime" in logs:
                self.say(f"done   train_runtime {logs['train_runtime']:.0f}s  final loss {logs.get('train_loss', float('nan')):.4f}")
            else:
                self.s.update(logs)
                self.s["phase"] = "train"
                self.draw(force=True)
            return
        if any(h in seg for h in HARMLESS):
            return
        if "Saving model checkpoint" in seg:
            return
        self.say(raw)  # keep indentation (tracebacks)

    def run(self, stream):
        buf = ""
        while True:
            ch = stream.read(1)
            if not ch:
                break
            self.raw.write(ch)
            if ch in "\r\n":
                self.feed(buf)
                buf = ""
            else:
                buf += ch
        self.feed(buf)
        sys.stdout.write(CLEAR + self.status() + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    View(sys.argv[1] if len(sys.argv) > 1 else "train_raw.log").run(sys.stdin)
