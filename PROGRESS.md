# counterpoint-vlm: progress report and one-week experiment plan

*Status as of 2026-11-02. Companion to `data_generation_spec.md`, `vlm_prior_counting_design.md` and the proposal.*

## 1. Summary

**Question.** When a VLM sees a counterfactual image (a five-legged dog, a seven-pip die), it often answers
from its prior ("4", "6"). Which synthetic fine-tuning data fixes this?
- **Study 1:** conflict data vs neutral "skill" data, crossed with count-only vs point-then-count targets.
- **Study 2:** how realistic the training images must be (rendered L0 → edited real photos L2/L3).

**Done so far:**
- **Evaluation harness:** VLMBias plus every generated test set; LoRA adapters or merged checkpoints.
- **Base-model baseline:** Qwen3.5-4B on VLMBias, 10.8% accuracy / 72.7% bias on the counting items.
- **Data pipeline:**
  - All of Study 1's training data and the T0 count test set.
  - Study 2 at L0 (count) and L2 (color, real photos), with T2 color.
  - Automated QA, a viewer, and a static report.
- **Last full build:** 45,690 records and 246,608 export rows; every CI check passes.

**Not done:** no LoRA has been trained yet. The training and eval scripts are written but untested (§6.3).
Two GPU-based data checks haven't run because the shared GPU was busy.

**This week's goal:** train and evaluate the Study 1 core 2×2 (one seed), its three controls, and a first
Study 2 color pair. Show the results with the baseline and the data report, whether the effects are
positive or null.

---

## 2. What has been built

### 2.1 Evaluation harness: `run_vlms_are_biased.py`

- **Inputs:** VLMBias (`--split main|original|...`) or any generated pool (`--records <dir|pool|records.jsonl>`).
- **Models:** the base model, PEFT LoRA adapters (`--adapters`), or a merged checkpoint (`--model path`).
- **Backends:** vLLM by default on GPU (FlashAttention, MTP speculative decoding when no adapters are loaded),
  or HF transformers + PEFT. Greedy decoding, thinking off unless `--thinking`.
- **Metrics:**
  - Accuracy and bias rate (the share of answers equal to the familiar answer, counterfactuals only).
  - Per topic, per family and role, pooled by role (canonical / counterfactual / neutral).
  - **Counting-only** for VLMBias main (the 1,992-item primary outcome).
- **Parsing:** the shared `answer_parsing.py`, also used by the data checks. It takes the last `{…}`, or a
  bare answer reply.
- **Robustness:**
  - Resumable; results are written incrementally and re-scored from stored responses.
  - Per-question progress bar.
  - Merged checkpoints get unique result names.

### 2.2 Data generation: `datagen/` (spec §3–§6, §9, §10)

| Pool | What | Size (4N where applicable) |
|---|---|---|
| `shared_range` | Neutral large-count shapes, 12–51 (U = 51 from VLMBias's largest count), in every arm | 1,024 |
| `e1/conflict` | 9 rendered families: die, playing card, domino, clock (hour marks), stop sign, star, calendar, piano, Rubik's face; canonical/counterfactual pairs | 3,584 pairs |
| `e1/neutral` | Geometric twins, count histogram matched family by family | 7,168 |
| `e1/dice` | Dice pairs (B-count control) | 3,584 pairs |
| `e1/val_*` | Synthetic validation, disjoint seeds | 1,024 per data type |
| `tests/T0/count` | Held-out families: traffic lights, snowflakes | 1,000 pairs |
| `e2/L0/count` | Stop signs (Δ ±1,2) + numeral clocks (Δ −1..−3) | 4,096 pairs |
| `e2/L2/color` | Real COCO/LVIS photos, CIELAB recolor; conflict (CoDa "single") + neutral (CoDa "any") | conflict ≈ 2.2k pairs (supply-limited), neutral 4,096 |
| `tests/T2/color` | Held-out color objects on reserved photos | ≈ 400 of 500 targeted |

**Exports** (`data/exports/<arm>/<model>/<format>/{train,val,train_rl}.jsonl`). The format is ms-swift /
LLaMA-Factory `messages + images`. The RL files add the ground-truth answer and points. Exports are
byte-identical on re-export.

| Export arm | Design-doc name | Images | Target format |
|---|---|---|---|
| `conflict-count` | C-count | conflict pairs + shared | `number` → `{7}` |
| `conflict-point` | C-point | same | `points` → point JSON, then `Total: {7}` |
| `neutral-count` | N-count | neutral twins + shared | `number` |
| `neutral-point` | N-point | same | `points` |
| `neutral-list` | N-enum | same | `list` → `pip 1, …, pip 7. Total: {7}` |
| `conflict-blind` | C-blind | conflict pairs as gray frames + real shared block | `number` |
| `dice-only` | B-count | dice pairs + shared | `number` |
| `neutral-randpoint` | N-randpoint (conditional) | neutral twins + shared | `randpoints` |
| `C-L2`, `N-L2`, `C-L2x4`, `N-L2x4` | Study 2 L2 (color only for now) | L2 color (+ L2 count later) + shared | `number` / `points` |
| `C-L0`, `C-L0x4` | Study 2 L0 | stop signs, numeral clocks + shared | `number` / `points` |

Every Exp 1 arm has 8,192 training images. Qwen3.5 points use a 0–1000 grid, pending the probe in §6.1;
Qwen2.5-VL points use pixels.

**Quality controls** (`python -m datagen check` runs all of them; the last run passed everything):
- **Reproducible:** every record regenerates byte-for-byte from its metadata.
- **Counts verified:** each count by an independent pixel recount of a clean render; each color by
  naming the recolored pixels. Failures are resampled and logged.
- **Points inside masks:** every ground-truth point lies inside its unit mask.
- **Pairs:** each edit is confined to the object, and both images of a pair share compression.
- **Distributions:** Δ is exactly uniform per family, twin and crowd histograms match, family quotas are
  equal, templates are split 50/50.
- **Leakage:** no held-out families or objects in training; no POPE/ORIC image IDs; no perceptual-hash
  near-duplicates of eval images; no photo in both training and T2.
- **Length rule:** passes with both tokenizers (1,365 / 1,431 of 1,536 tokens).
- **Exports parse** with the eval parser.

**Tooling:**
- `preview`: a small version of everything in about 3 minutes.
- `all`: the full pipeline, resumable.
- `view`: a browser viewer over an SSH tunnel, with overlays, pair diffs, flagging, and the blind
  click-to-count audit for Gate 4.
- `report`: a static tarball with plots, galleries, contact sheets and CI results.
- `stats`.
- `--data PATH` on every command; logs for the GPU tools.

### 2.3 Decisions and deviations from the spec

| Date | Decision | Why |
|---|---|---|
| 10-05 | Placement minimum object size 190 → 130 px; dense families (calendar, card, Rubik's) ≥ 220 px with units ≥ 20 px; card pips 9.5% → 12% of card height | Centered objects violated "anywhere in frame"; the base model couldn't count small, dense units even on canonicals |
| 10-05 | Box-filter supersampling; numeral masks = text boxes | Lanczos leaked edits outside the object box; centroids of "11" fell between digits |
| 10-06 | **Emoji dropped as a source** (no L0 color, no L0 neutral crowds, no T0 color) | Stylized, multi-colored, ambiguous labels; no color bias at L0 (95% correct). Color moved to L2 real photos |
| 10-06 | Visual CounterFact exclusion uses its **color split only** | Its size split doesn't ask about color |
| 10-06 | **≥ 70% single-color rule** for color objects (deep shadows and highlights discounted), checked per object and per image | "What color is the watch?" had several right answers |
| 10-06 | L2 color masks come from **LVIS annotations** (no SAM 3) | Spec §5.5 allows LVIS masks for color; SAM 3 is for mined count families / L3 |
| 10-06 | Display/print surfaces excluded (billboard, poster, sign, screen) | Recoloring tints the picture they show |
| 10-07 | Δ fixed per item, cycling through allowed values; star recount merges close vertices | Retries had skewed the Δ histograms (3-point stars under-represented) |

---

## 3. Results so far

### 3.1 Base model on VLMBias (Qwen3.5-4B, greedy, thinking off, Oct 4)

| | n | acc % | bias % |
|---|---|---|---|
| **Counting items (primary)** | 1,992 | **10.8** | **72.7** |
| All items incl. optical illusions | 2,784 | 23.3 | 64.6 |
| Animals | 546 | 0.0 | 99.3 |
| Logos | 414 | 2.2 | 94.7 |
| Flags | 240 | 18.3 | 67.1 |
| Chess pieces | 288 | 12.5 | 49.0 |
| Game boards | 168 | 28.6 | 39.9 |
| Patterned grids | 336 | 23.5 | 43.5 |
| Optical illusions (yes/no, chance 50%) | 792 | 54.7 | 43.9 |

The counting-only row is derived from the per-topic numbers. That run capped generation at 1,024 tokens
(about 3% truncated), so rerun it with the standard suite (§6.2) for the final baseline. Gate 0's first
half passes: counterfactual accuracy is ≤ 30%. Canonical accuracy, from the `original` split, is the
other half.

### 3.2 Base model on a small preview of the generated data (Oct 6, n ≈ 8 per cell, before later fixes)

- **Fixed-prior families reproduce the VLMBias pattern.**

  | Family | canonical | counterfactual | bias |
  |---|---|---|---|
  | Clock (hour marks) | 100% | 0% | 100% |
  | Stop sign | 100% | 0% | 100% |
  | Numeral clock (Exp 2) | 100% | 0% | 100% |
  | Star, piano | 100% | 38–50% | ~38% |
  | Snowflake (held out) | 100% | 50% | 50% |

  The bias column is the share of counterfactuals answered with the familiar count.
- **Variable-prior families show little bias:** die 88% on counterfactuals, domino 75%, Rubik's 88%.
- **Problems found:** playing cards scored 0% even on canonicals (the model names the card instead of
  counting), and calendars 38% on canonicals (rows/columns swapped). Both prompted the size fixes in §2.3.
  **They must be re-checked this week.**
- **Color at L0 (emoji):** 95% correct, no bias. That's why emoji were dropped.

### 3.3 Data yield

- **L2 color:** about 25% of candidate photo instances pass the rules. Conflict objects pass at about 16%,
  limited by the spec's check that the object reads as its CoDa typical color.
- **Object sets:** 36 training conflict objects, 9 held-out, 63 neutral.
- **Shortfall:** the projected L2 conflict pool covers N = 1,024 but not the 4N volume arm. The build
  report (`data/reports/build_e2_L2_color.json`) has the actual counts.

---

## 4. Known issues and risks

1. **Training stack is untested.** `tools/train_arm.sh` follows ms-swift 4.6's documented flags for
   Qwen3.5, but has never run. Budget day 2 for it, as the design doc does.
2. **Blind check and point-format probe not run yet** (GPU busy). Until the probe runs, Qwen3.5 point
   targets use the 0–1000 grid.
3. **Cards and calendars** may still measure counting ability rather than prior override. If their
   canonical accuracy stays low after the size fix, report them as skill families and leave them out of
   conflict claims.
4. **L2 color label noise** that only a person catches: LVIS labels a bicycle pictogram on a sign
   "bicycle"; cream butter is named "yellow". Gate 4 needs the 100-item human audit (viewer Audit tab).
5. **The shared block** (12–51 shapes) is hard for the base model (3–14%). It's in every arm, so it can't
   explain differences between arms, but it dominates target length in the point arms.
6. **Exp 2 L0 neutral arms are empty**, pending real-photo crowds.
7. **Dynamic LoRA in vLLM** gives NaNs past about 400 tokens on Qwen3.5, so always evaluate merged
   checkpoints. `train_arm.sh` does this.

---

## 5. Experiment plan for this week

Ordered so that every day ends with something showable, and later days build on earlier gates. One shared
40–48 GB GPU is assumed. Times are rough: a 4B arm trains in about 50 min (count) to 80 min (points) per
8,192 images × 2 epochs; one eval suite takes about 35 min.

| Day | Experiment | Commands (§6) | GPU time | Output to show |
|---|---|---|---|---|
| 1 | **E0a. Finish data QA**: point-format probe, blind check | `datagen all` (resumes; only GPU tools run) | 0.5 h | `reports/blind_check.json`, `point_format_probe.json`, data report tarball |
| 1 | **E0b. Base-model baseline / Gate 0** on the core suite: VLMBias main + `original`, T0 count, T2 color, synthetic val | `tools/eval_suite.sh /data/vlm` | 0.6 h | Baseline row of `results/RESULTS.md`; per-family conflict dose on T0/T2 (Gate 5) |
| 1 (overnight) | E0c. *Optional:* thinking-mode baseline on VLMBias main | `run_vlms_are_biased.py --thinking --max-new-tokens 16384` | 2–3 h | Does reasoning fix prior override for free? |
| 2 | **E1-pilot / Gate 2**: train C-count seed 0, check in-distribution val ≥ 95% | `tools/train_arm.sh /data/vlm conflict-count number 0` | 1.5 h | Training works (or not); first trained row |
| 3 | **E1 core 2×2, seed 0**: N-count, N-point, C-point (C-count done on day 2) | `train_arm.sh … neutral-count number 0`, `… neutral-point points 0`, `… conflict-point points 0` | 5 h | Contrasts 1–3 of the design doc |
| 4 | **E1 controls, seed 0**: N-enum, C-blind, B-count | `… neutral-list list 0`, `… conflict-blind number 0`, `… dice-only number 0` | 3.5 h | Contrast 4 (coordinates vs enumeration), Gate 3 (shortcut), narrow vs broad |
| 5 | **E2 color pilot, seed 0**: C-L2 vs N-L2 (real-photo color) | `… C-L2 number 0`, `… N-L2 number 0` | 2.5 h | Does conflict color data move T2 color bias? |
| 5–6 (overnight) | **Seeds 1 (and 2)** for the core 2×2 | same as day 3 with seed 1, 2 | 6–12 h | Error bars |
| 6 | E1 conditional: N-randpoint, only if N-point > N-enum | `… neutral-randpoint randpoints 0` | 1.5 h | Do correct point locations matter? |
| 7 | **Write-up** | `uv run tools/summarize_results.py results`; data report | none | Tables + figures below |

**If time runs short:** cut, in order, E0c → day 6 → the seed 2 runs. The minimum showable set is E0b,
the four core arms at seed 0, and C-blind.

### 5.1 How to read the results (pre-registered in the design doc)

Primary Study 1 outcome: counterfactual accuracy on the 1,992 VLMBias counting items. An arm's gain counts
only if its canonical accuracy on the `original` split stays within 5 points of the base model's. Also
report balanced accuracy (the mean of the two).

| Contrast | Question |
|---|---|
| 1. N-point − N-count | Does grounded output help without a prior to fight? (H3) |
| 2. C-count − N-count | Does conflict data help one-shot answers? (H1 vs H2) |
| 3. C-point − N-point | Does conflict data add anything once the model points? |
| 4. N-point − N-enum | Coordinates, or just enumeration? |
| Gate 3: C-blind vs C-count | If C-blind recovers ≥ half of C-count's gain, the conflict effect is a text shortcut |

| Pattern | Interpretation |
|---|---|
| N-point ≈ C-point ≫ N-count ≈ C-count | Format is the lever; skill data is enough once the model points (hoped-for headline) |
| C-count > N-count, N-point ≈ C-point | Pointing substitutes for conflict data |
| C-arms ≫ N-arms, C-blind does not match | Conflict data is needed (H2) |
| C-arms ≫ N-arms, C-blind matches | The conflict effect is a text shortcut (cautionary result) |
| Nothing transfers to held-out VLMBias | Replicates the non-transfer result with stronger controls; T0 and in-distribution gains still show what the data teaches |

With one seed, call differences under about 3 points on the pooled 1,992 items "no clear difference".
Per-category differences under about 10 points are noise (game boards has only 168 items).

### 5.2 What to present

1. **The problem:** base-model VLMBias table (§3.1), with example images from the viewer.
2. **The data:** report tarball (galleries, contact sheets with ground-truth points, distributions), CI
   all-pass, and real-photo L2 color pairs.
3. **Gates:** Gate 0 (headroom), Gate 2 (training works), Gate 3 (shortcut check), Gate 5 (conflict dose
   per test set).
4. **The 2×2:** the `RESULTS.md` table and the four contrasts, plus controls.
5. **Study 2 first look:** C-L2 vs N-L2 on T2 color.
6. **Honest negatives and deviations:** §2.3 and §4.

---

## 6. How to run

Everything runs on the GPU box over SSH. Run long jobs in tmux. `/data/vlm` is the data directory used
so far; pass the same `--data` everywhere.

### 6.1 Setup and data

```bash
git pull && uv sync
uv run python -m datagen --data /data/vlm all      # resumable: skips finished sources/steps; runs GPU tools if missing
```
- If a GPU tool reports "GPU busy", wait for `nvidia-smi` to show about 14 GB free and rerun `all`.
  Logs are in `/data/vlm/reports/logs/`.
- Look at the data:
  - **Viewer:** `uv run python -m datagen --data /data/vlm view` on the box, then
    `ssh -N -L 7860:localhost:7860 <the host you ssh into>` on the laptop, then open
    http://localhost:7860.
  - **Static report:** `scp <host>:/data/vlm/reports.tar.gz .`
- **Gate 4 audit (human):** viewer → Audit tab → pool `e2/L2/color` → 100 items. Target ≥ 97% label
  accuracy. Answers are logged to `/data/vlm/reports/audit_log.jsonl`.

### 6.2 Evaluate a model

```bash
tools/eval_suite.sh /data/vlm                       # base model
tools/eval_suite.sh /data/vlm runs/<arm>/v*/checkpoint-*-merged   # a trained arm (train_arm.sh calls this)
uv run tools/summarize_results.py results           # -> results/RESULTS.md
```
- **The suite:** VLMBias main, VLMBias `original` (guardrail), T0 count, T2 color, and synthetic
  validation. All runs use greedy decoding, thinking off, and the same 4,096-token cap.
- **Knobs:** `EVAL_GPU_UTIL=0.5` when the GPU is shared; `EVAL_MAX_TOKENS` changes the cap.
- **Single runs:** `uv run run_vlms_are_biased.py --model <m> --split original` or `--records <pool>`.
  `--thinking` turns thinking on.

### 6.3 Train an arm

```bash
tools/train_arm.sh /data/vlm <arm> <format> <seed>  # e.g. conflict-count number 0; neutral-point points 1
```
- **Training:** ms-swift 4.6 in its own uv environment, with the design-doc settings (§9.2).
  - LoRA rank 16, α 32, dropout 0.05, on all language-model linear layers; vision tower and projector
    frozen.
  - Learning rate 1e-4, cosine schedule, 3% warmup, 2 epochs, effective batch size 32, bf16, gradient
    checkpointing.
  - Max length 1,536; the non-thinking prefix is added for Qwen3.5; Hugging Face Hub weights.
- **Then it** merges the adapter (always; see §4.7) and runs the eval suite.
- **Output:** `runs/<arm>-<format>-qwen3.5-s<seed>/`. Rerunning skips training if a merged checkpoint
  exists.
- **Knobs:** `BATCH=2` if it runs out of memory (gradient accumulation adjusts to keep the effective
  batch at 32); `SKIP_EVAL=1`.
- **Untested:** this is the first run of this stack. If ms-swift rejects a flag, check
  `swift sft --help` for 4.6 and adjust in `tools/train_arm.sh`.
  - **Gate 2 pass:** synthetic val accuracy ≥ 95% (the "Val acc" column in `RESULTS.md`).
  - **Gate 2 fail:** fix training before looking at any VLMBias number.

### 6.4 Pipeline reference

| Command | What |
|---|---|
| `python -m datagen [--data P] preview` | Small version of all pools + report in `P/preview` |
| `… all` | Everything in order, resumable |
| `… sources [refresh]` | Downloads, exclusions, U |
| `… build [shared e1 t0 e2 l2]` | Render pools (always rebuilds the named steps) |
| `… export` / `check` / `report` / `stats` / `audit` | Exports / CI / tarball / plots / contact sheets |
| `… view [port]` | Viewer (localhost only) |
| `uv run tools/check_length.py` · `point_format_probe.py` · `blind_check.py` `--data P` | Length rule · point convention · text-only leak |
| `uv run --group dev pytest tests` | Generator tests |

---

## 7. Not built yet (from the spec / design doc)

**Data:**
- L2 count: stop-sign re-warp, clock numeral removal (LaMa + OCR), copy-paste crowds; SAM 3 mining of
  extra families. This also fills the Exp 2 L0/L2 neutral count arms.
- L1 compositing, L3 generative edits (FLUX.2 klein / Qwen-Image-Edit), T1/T3.
- Probe sets: edited-canonical, masked-object, gray-frame VLMBias, rendered canonicals.
- The SigLIP edit-artifact probe for Gate 4.

**Eval:**
- Gray-frame image sensitivity.
- Pointing-prompt baseline (Gate 1).
- Contrastive decoding baseline.
- Point F1, point–answer agreement, point-list editing (§8.3).
- Visual CounterFact, ORIC, CountBenchQA, PixMo-Count and MMStar runs.

**Training:**
- Qwen2.5-VL-3B anchor runs; GRPO stage.

Suggested order after this week:
1. Gray-frame sensitivity and point–answer agreement (cheap, strengthen Study 1).
2. Visual CounterFact color eval (Study 2 primary set).
3. L2 count.
4. L3.
