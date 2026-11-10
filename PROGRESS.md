# counterpoint-vlm: progress report and one-week experiment plan

*Status as of 2026-11-05 (baseline, pointing baseline, C-count pilot and C-blind done). Companion to `data_generation_spec.md`, `vlm_prior_counting_design.md` and the proposal.*

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

**Baseline (Nov 4):** VLMBias counting accuracy 12.7% / bias 73.4% against 96.9% on unedited images,
so Gate 0 passes. T0 count conflict dose is 42%. Real-photo color shows no bias (1%), so color can't
show a bias fix for this model (§3.1).

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

### 3.1 Baseline: Qwen3.5-4B on the core suite (2026-11-04, greedy, thinking off)

| Set | n | Canonical acc | Counterfactual acc | Bias rate |
|---|---|---|---|---|
| **VLMBias counting (primary)** | 1,992 cf / 163 canonical | **96.9** (`original` split, counting questions) | **12.7** | **73.4** |
| T0 count: snowflake | 500 + 500 | 100.0 | 20.4 | 79.4 |
| T0 count: traffic light | 500 + 500 | 99.6 | 74.2 | 5.4 |
| T0 count: pooled | 1,000 + 1,000 | 99.8 | 47.3 | 42.4 |
| T2 color (real photos) | 291 + 291 | 95.5 | 95.2 | 1.0 |
| Synthetic val, conflict pairs | 512 + 512 | 84.6 | 38.3 | 37.7 |
| Synthetic val, shared block (12–51 shapes) | 1,024 neutral | 8.9 (all items) | | |

Balanced accuracy on VLMBias (mean of counterfactual and canonical accuracy) is 54.8.

*Corrected 11-05:* the canonical number was first reported as 85.9% because the `original` split also
contains identification questions ("which flag is this?"). Counting questions only, it is 96.9%.

| VLMBias topic | acc % | bias % |
|---|---|---|
| Animals (generative edit, "L3-like") | 0.0 | 99.3 |
| Logos (generative edit, "L3-like") | 2.4 | 94.0 |
| Flags | 18.3 | 68.3 |
| Chess pieces | 20.8 | 53.5 |
| Game boards | 32.7 | 39.9 |
| Patterned grids | 24.7 | 43.8 |

| Synthetic val family | canonical | counterfactual | bias |
|---|---|---|---|
| clock | 100.0 | 0.0 | 100.0 |
| stop sign | 100.0 | 5.3 | 87.7 |
| star | 100.0 | 43.9 | 26.3 |
| piano | 78.9 | 50.9 | 35.1 |
| calendar | 59.6 | 15.8 | 29.8 |
| playing card | 42.1 | 22.8 | 21.1 |
| domino | 87.7 | 56.1 | 10.5 |
| rubik | 92.9 | 76.8 | 19.6 |
| die | 100.0 | 73.7 | 8.8 |

**What it means.**
- **Gate 0 passes clearly.** Counterfactual accuracy is 12.7% (≤ 30%) and canonical accuracy 96.9%: an
  84-point gap. The failure is prior override, not perception, and there is plenty of headroom.
- **The bias is strongest on generative-edit images.** Animals and logos sit at 0–2% accuracy; the
  programmatic categories at 18–33%.
- **Gate 5 (conflict dose):**
  - *T0 count:* 42% pooled, so it counts as a conflict test. Within it, snowflakes are strong conflict
    (79%) but traffic lights are not (5%, below the 30% bar), so report traffic lights as a skill test.
  - *T2 color:* 1% bias. **Qwen3.5-4B shows essentially no color prior override on real-photo recolors**,
    so T2 color measures color naming, not prior override, and color arms cannot show a bias reduction for
    this model. This is a result in itself, consistent with the L0 emoji finding.
- **The rendered conflict families split three ways.**
  - *Strong-prior families* behave like VLMBias: clock and stop sign at 88–100% bias, star and piano
    moderate.
  - *Variable-prior families* show little bias (die, domino, rubik).
  - *Cards and calendars* improved after the size fix (canonical 0% → 42%, 38% → 60%) but are still
    partly skill-limited, so report them as such.
- **Gate 2 needs reading per pool.** The base model is at 61% on conflict validation and 9% on the shared
  block. The block's accuracy collapses above about 20 shapes (23% for 12–21, 3% for 32–51), so ≥ 95% on
  the shared block is unrealistic after 2 epochs. Judge Gate 2 on the conflict pool and report the shared
  block separately.
- **Small caveat:** the VLMBias main numbers re-scored the Oct 5 predictions (no token cap) instead of
  regenerating at 4,096. For exact comparability with the arms, rerun once:
  `uv run run_vlms_are_biased.py --max-new-tokens 4096 --rerun` (about 6 min).

The October numbers (1,024-token cap, before the parser fix) were 10.8% / 72.7% on counting items.

### 3.2 Stage 2 (2026-11-05): pointing-prompt baseline and the C-count pilot

| Model | VLMBias cnt acc | bias | Canonical (orig) | Balanced | T0 can | T0 cf | T0 bias | Val conflict | Val shared |
|---|---|---|---|---|---|---|---|---|---|
| Base | 12.7 | 73.4 | 96.9 | 54.8 | 99.8 | 47.3 | 42.4 | 61.4 | 8.9 |
| Base + pointing prompt, **point-list length** | 37.2 | 27.1 | 81.0 | **59.1** | 77.8 | 63.3 | 10.8 | 49.3* | 21.7 |
| **C-count pilot** (seed 0) | **20.6** | **56.6** | 92.0 | 56.3 | 99.1 | **75.7** | **12.6** | 93.1 | 31.2 |

\*Val conflict for pointing is the mean of canonical 59.8 and counterfactual 44.7.

**Pointing prompt (Gate 1).**
- **Readout:** the model points 99% of the time but usually never states a total (72–96% of replies), so
  the point-list length is the answer. The stated-total column in `RESULTS.md` (6.9%) only counts the
  replies where it did.
- **Large effect on prior override:** list-length accuracy is 37.2% vs 12.7% plain (+24.5, above Gate 1's
  +20 bar), and the bias rate falls from 73.4% to 27.1%.
- **By topic:** animals go from 0.0% to 73.4%; flags from 18.3% to 58.8%. Dense layouts don't benefit:
  chess 13.9%, game boards 15.5%, grids 14.3%, logos 20.8%.
- **Cost on unedited images:** 96.9% → 81.0% canonical (T0 canonical 99.8% → 77.8%).
- **Net:** balanced accuracy 59.1, **higher than the trained pilot (56.3)**. T0 point F1 is 47.4%.
- **Consequence (design doc Gate 1):** the trained point arms must be compared against the
  prompted-pointing base, not the plain base. It also makes the distillation question (§5.0) concrete:
  can a plain answer match the pointed one?

**C-count pilot.**
- **Training:** loss 2.00 → 0.04. Eval loss was 0.20 after epoch 1 and 0.16 after epoch 2, still falling.
- **VLMBias:** +7.9 points accuracy (12.7 → 20.6) and −16.8 bias (73.4 → 56.6). Paired over the 1,992
  items it fixed 290 and broke 132, net +158 (sign test, p ≪ 0.001).
  - *Gains:* animals 0.0 → 12.6, logos 2.4 → 23.4, flags 18.3 → 34.2, game boards 32.7 → 45.2.
  - *Losses:* chess 20.8 → 6.9, grids 24.7 → 19.6.
- **Guardrail:** canonical 96.9 → 92.0 (−4.9, just inside the 5-point validity rule), so balanced
  accuracy rises only +1.5 (54.8 → 56.3). Part of the gain could be a learned push away from the familiar
  answer. **C-blind (Gate 3) decides this and is the next priority.**
- **Held-out rendered families (T0) transfer well:**
  - Snowflake counterfactual 20.4 → 53.8 (bias 79.4 → 25.0).
  - Traffic light counterfactual 74.2 → 97.6.
  - Canonicals unchanged (99).
- **Gate 2: 93.1%, so a formal FAIL (−1.9).** It isn't a training failure. Every family is ≥ 84% except
  clock counterfactuals (24.6%), and those errors are now ±1 off (15→14, 13→12, 11→12), not the familiar
  12. The clock's evenly spaced hour marks (9–15) have become a counting-precision problem. A third
  epoch would probably pass, but then the protocol differs between arms. Recommendation: keep 2 epochs
  and report clock as the outlier.

**C-blind (Gate 3): is the pilot's gain a text shortcut?** C-blind trains on the same conflict
questions and answers as C-count, but every conflict image is replaced by a gray frame (the shared block
keeps its real images). Same 512 steps.

| Model | VLMBias acc | bias | Canonical | Balanced | T0 cf acc (bias) | Val conflict | Val shared |
|---|---|---|---|---|---|---|---|
| Base | 12.7 | 73.4 | 96.9 | 54.8 | 47.3 (42.4) | 61.4 | 8.9 |
| C-count | **20.6** | 56.6 | 92.0 | **56.3** | **75.7 (12.6)** | 93.1 | 31.2 |
| C-blind | 9.8 | 62.2 | 86.5 | 48.1 | 45.6 (45.4) | 56.5 | 30.4 |

- **Accuracy gains are not a shortcut.** C-blind recovers none of C-count's accuracy gain (−2.9 vs +7.9).
  Item by item, C-blind fixed 129 and broke 186 (net −57), against C-count's net +158. On held-out T0,
  C-blind is unchanged (45.6), while C-count goes from 47.3 to 75.7. The pilot's gains come from the
  images.
- **The bias-rate drop *is* partly a shortcut.** C-blind cuts bias by 11.2 points, two thirds of
  C-count's 16.8, which is past Gate 3's "half or more" bar for that metric. It does this by answering
  the familiar count **+2** (294 items) or +1 (125). That's a learned "edited images want a slightly
  different answer" rule, and it costs canonical accuracy (−10.4 points: unedited 4-legged animals get "5",
  two-star flags get "4"). C-count shows a weaker version of the same drift: 135 of its wrong answers are
  familiar +2, against 42 for the base.
- **Consequence:** report conflict arms on **accuracy and balanced accuracy**, not bias rate, as the
  design doc's Gate 3 rule says. On balanced accuracy, C-count beats C-blind by 8.2 points (56.3 vs 48.1).
- **By topic:** C-blind gains only on logos (2.4 → 11.8; C-count gets 23.4) and loses on chess
  (20.8 → 0.0) and flags. Its shared-block gain (30.4) matches C-count's (31.2), as expected, since both
  saw the same real shared-block images.

**Measurement fixes made while reading these results** (applied to all models by `tools/rescore_results.py`):
- JSON point objects are no longer taken as answers.
- "Counting" means numeric-answer questions only.

### 3.2c Queue run 1 (2026-11-09): point-then-total prompt, forced direct answers, real-photo counting

Base model, counting items, rescored with the parser fix for bare `Total: N` lines:

| prompt | VLMBias acc / bias | Orig (canonical) | Balanced | T0 can / cf / bias | Val conflict / shared | Real count (list acc, point F1) |
|---|---|---|---|---|---|---|
| default | 12.7 / 73.4 | 96.9 | 54.8 | 99.8 / 47.3 / 42.4 | 61.4 / 8.9 | (real-sweep pending) |
| direct (number only) | 9.0 / 73.1 | 95.1 | 52.1 | 99.4 / 52.4 / 35.9 | 64.6 / 5.7 | 37.2 |
| point-then-total | **39.0 / 26.1** | 79.1 | 59.1 | 71.5 / 61.1 / 5.3 | 46.1 / 19.6 | 47.1 (47.1, F1 77.1) |

- **Pointing with a stated total roughly triples VLMBias accuracy** (660 items gained, 150 lost vs default)
  and cuts bias 73 to 26, at a canonical cost (96.9 to 79.1, mostly logos/flags, typically off by 1–2).
- **Mechanism: the list and the total mostly agree (85%).** When they disagree (12% of VLMBias counting
  items), the stated total is closer to the familiar answer than the list is in 177 cases vs 58 the other
  way: a late pull toward the prior. It nets out in accuracy (18 list-right/total-wrong vs 18 the reverse)
  and is concentrated in dense layouts (chess/xiangqi pieces, tallies, dice, board grids). On T0 and
  real photos the two almost never disagree (<1%). So most of the remaining bias is already in the list
  (list bias 23%): the prior acts mainly while enumerating, not as an override of a correct list.
- **Forced direct answers are slightly worse than the default prompt** (9.0 vs 12.7, same bias): the
  short reasoning the model writes by default helps a little; prior override is not about verbosity.
- **Real-photo counting (1,754 held-out crops, 3–15 instances):** 47% exact with pointing (point F1 0.77)
  vs 37% with direct answers; pointing wins in every count bin (net +174 items). But direct accuracy is
  only 41% on crops with 3–4 instances, and 89% of direct errors are undercounts: many LVIS "instances"
  are probably too small or occluded to count. Treat tests/R/count as unaudited until Gate 4; 78% of
  items have 3–4 instances (count distribution is skewed low).

### 3.2d C-point s0 (2026-11-09): point-then-count training on conflict pairs (H3b)

Plain prompts (the arm answers with a point list + `Total: {N}` by default):

| model | VLMBias acc / bias | Orig | Balanced | T0 can / cf / bias | T0 point F1 | T2 cf | Val conflict / shared | Real count (F1) |
|---|---|---|---|---|---|---|---|---|
| base, default prompt | 12.7 / 73.4 | 96.9 | 54.8 | 99.8 / 47.3 / 42.4 | – | 95.2 | 61.4 / 8.9 | – |
| base, point-then-total prompt | **39.0** / 26.1 | 79.1 | **59.1** | 71.5 / 61.1 / 5.3 | 47.1 | 95.2 | 46.1 / 19.6 | **47.1** (77.1) |
| C-count s0 | 20.6 / 56.6 | **92.0** | 56.3 | 99.1 / 75.7 / 12.6 | – | 95.2 | 93.1 / 31.2 | – |
| C-point s0 | 28.4 / 26.1 | 72.4 | 50.4 | 95.2 / **88.7 / 1.2** | **90.0** | 82.1 | 93.4 / **73.2** | 44.4 (77.4) |

- **Synthetic: large gains.** Held-out families (T0) counterfactual accuracy 88.7 with 1% bias, point F1
  47 to 90; the hard 12–51 shared block goes from 9–31 to 73.
- **VLMBias: better than C-count, worse than just prompting the base model to point** (28.4 vs 39.0), and
  canonical accuracy falls to 72.4 (fails the 5-point guardrail). Most of the loss is Animal Add Legs
  (546 items): 78.8 with the base pointing prompt, 40.5 for C-point (40% of animal answers are the familiar
  count). Some point lists are regular lattices (e.g. 3x2 grids for a horse's legs): the arm learned the
  synthetic layouts' geometry along with pointing. It gains on board grids (sudoku 25 to 79, chess grid
  2 to 58) and chess/xiangqi pieces.
- **Format collapse:** the arm points on every question, including color questions ("Total: {1}" for
  "What color is the egg?", T2 95 to 82) and when told to answer with a number only (the +direct run
  still points 100% of the time). So C-point cannot test H3c; the mixed-format arms (half plain-number
  targets) are the distillation test.
- **Real photos:** unchanged (44.4 vs 47.1 base pointing; point F1 equal), so real-photo grounding
  neither improved nor broke.
- Reading for H3b: point-then-count training fixes counting within the synthetic distribution almost
  completely but does not transfer to VLMBias beyond what the pointing prompt already gives. Next checks:
  N-point (does conflict data matter), mixed arms (format collapse, H3c), real-point (does real-photo
  pointing transfer), and an earlier checkpoint (epoch 1) to test overfitting to the synthetic layouts.
- **Decision (2026-11-09): real-photo pointing next, and overfitting made measurable.** Real arms
  (real-point, real-mixed, real-count) now train on real photos only (no synthetic shared block). Every
  E1 arm's validation set is 300 held-out real photos (real/val, excluded from real/count training,
  disjoint from tests/R/count); its loss is logged every 50 steps. Checkpoints are all kept, and
  `tools/ckpt_sweep.sh` scores real/val and T0 accuracy at 25/50/75/100% of training (the queue runs it
  after each new arm). C-point's intermediate checkpoints were already deleted (old limit of 2), so its
  overfitting curve needs a retrain (seed 1 is queued).

### 3.2e real-point s0 (2026-11-09): point-then-count on real photos only

| model | VLMBias acc / bias | Orig | Balanced | T0 can / cf | Color cf | Real count (point F1) |
|---|---|---|---|---|---|---|
| base, point-then-total prompt | **39.0** / 26.1 | **79.1** | **59.1** | 71.5 / 61.1 | 95.2 | 47.1 (0.77) |
| C-point s0 (synthetic) | 28.4 / 26.1 | 72.4 | 50.4 | 95.2 / 88.7 | 82.1 | 44.4 (0.77) |
| real-point s0 (real photos, 442 steps) | 30.4 / 28.9 | 71.8 | 51.1 | 88.7 / 62.2 | 91.8 | **68.5 (0.85)** |

- **In-domain gain:** held-out real-photo counting 47 → 68.5, point F1 0.77 → 0.85; color mostly intact.
- **No VLMBias transfer:** 30.4, like C-point and below the prompted base. Animals 51 (prompted base 79);
  39% of animal answers are still the familiar 4. Canonical 71.8 fails the guardrail.
- **Sweep (partial):** at step 100 of 442, real/val accuracy is already 69% (final test 68.5): real-photo
  gains come early. Remaining checkpoints running.
- **Interpretation:** both training sources teach counting of separate objects (rendered shapes' parts are
  synthetic; LVIS crops are whole objects). VLMBias asks for parts of one familiar object (legs, stars,
  stripes) on realistic images. Candidate next data: part counting on real objects.

### 3.2f Real-photo part counting (2026-11-10): Pascal-Part legs — built, not yet trained

**Why:** synthetic training counts parts of rendered shapes; the LVIS pool counts separate whole objects.
VLMBias counts parts of one familiar object (legs) in realistic images. Neither source covered that.

**Visible vs canonical counts in natural photos** (Pascal-Part, objects with part annotations;
`reports/pascal_part_stats.md`):

| class | part | canonical | objects | = canonical | below | mean visible |
|---|---|---|---|---|---|---|
| horse | leg | 4 | 621 | 50.7% | 49.3% | 2.79 |
| bird | leg | 2 | 934 | 43.8% | 56.2% | 1.03 |
| cow | leg | 4 | 415 | 33.0% | 67.0% | 2.27 |
| dog | leg | 4 | 1,403 | 24.7% | 75.3% | 2.27 |
| cat | leg | 4 | 1,128 | 15.3% | 84.7% | 1.88 |
| sheep | leg | 4 | 661 | 10.0% | 90.0% | 0.71 |
| bicycle / motorbike | wheel | 2 | 613 / 595 | 79.6% / 69.7% | | 1.77 / 1.67 |
| car | wheel | 4 | 1,745 | 2.8% | 97.2% | 1.36 |
| person | leg / arm | 2 | 7,839 | 46.0% / 56.8% | | 1.06 / 1.44 |

Above-canonical counts essentially never occur (1 car of 1,745). So for VISIBLE legs, "4" is usually wrong
in natural photos: the natural version of the prior conflict.

**Pools** (`datagen/parts.py`, `python -m datagen build parts`; 20% of images held out by id hash):
- `parts/count`, `tests/P/count`: unedited crops of one animal, "How many legs of the dog can you see?",
  a point + mask per visible leg (legs >= 150 px; animal >= 12% of the crop; no second animal of the same
  class in the crop). About 1,450 animals in total (estimated from 19% of the images).
- `parts/conflict`, `tests/P/conflict`: fully visible animals, two pairs each. The counterfactual has a leg
  added (copy of the animal's own leg, feathered, attached to the body) or removed (OpenCV Telea inpainting
  of the dilated leg mask). The canonical partner gets the same kind of edit on a leg-shaped background
  region (null edit), so "edited" never predicts "count changed". About 950 pairs.
- Counts are exact by construction. Local checks pass: records schema, points inside masks, re-render from
  metadata (deterministic), no train/test image overlap.
- Known weakness: Telea inpainting can leave a visible smudge where a leg was removed (the null edits leave
  the same kind of smudge, so it does not predict the count). LaMa would look better.

**Deviation from the spec:** the leakage rules keep "leg"/"animal"/"dog"/"cat"/"bird"/"horse" out of
training questions. The part pools are exempt (`checks.CATEGORY_EXEMPT`), and RESULTS.md now reports
VLMBias animals and VLMBias excluding animals separately, so in-category gains are not read as transfer.

**Arms** (real photos only, no synthetic block): `part-conflict-point`, `part-neutral-point`,
`part-conflict-count`. Queued right after the real-point sweep; `real-sweep` now also evaluates the base
(plain and point-then-total) and every arm on `tests/P/count` and `tests/P/conflict`.

### 3.2g First part-counting arm and the real-point sweep (2026-11-10)

**part-conflict-point s0** (real leg edit pairs, ~1,800 images, 112 steps; plain prompts):

| model | VLMBias | excl. animals | animals | canonical |
|---|---|---|---|---|
| base, default | 12.7 | 17.4 | 0.0 | 96.9 |
| base, point-then-total prompt | **39.0** | 24.0 | **78.8** | 79.1 |
| C-point s0 | 28.4 | 23.9 | 40.5 | 72.4 |
| real-point s0 | 30.4 | 22.5 | 51.1 | 71.8 |
| part-conflict-point s0 | 38.8 | **29.5** | 63.2 | 78.5 |

- Best trained arm; level with prompting overall, with no pointing instruction at eval time.
- **Transfer outside animals** (the arm trains on animal legs, so this is the clean number): vs the prompted
  base on 1,446 non-animal items it fixes 233 and breaks 153 (sign test p = 5e-5). Gains on logos, grids,
  flags.
- Animals: 63 vs 79 for the prompted base (fixes 70, breaks 155). Unexplained so far.
- Canonical 78.5: best of the point-trained arms; the guardrail still fails (base 96.9).
- One seed. Held-out leg tests (tests/P) still running at download time.

**Queued next (2026-11-10):**
- *Prompt study* (VLMBias main + original, base model; `prompt` jobs): `cot` (think step by step),
  `debias` ("count each and every one of the {part}… the image may have been edited… ignore what you
  expect"), `cot-debias` (step-by-step enumeration of each {part} + edit check + "intentionally ignore my
  bias"), `point-debias` (point-then-total + de-bias), and thinking mode. {part} is parsed from the question
  (`answer_parsing.counted_part`: legs, stars, stripes, chess pieces, …). Also part-conflict-point + debias.
- *RL* (`tools/train_rl.sh`, GRPO in ms-swift, LoRA on top of merged part-conflict-point): prompts = the part
  pools' train_rl rows; rewards (`tools/rl_rewards.py`) = correct total (1.0) + point F1 against the
  ground-truth leg points within max(25, 0.6 x unit size) on the 0-1000 grid (1.0) + list length equals
  the stated total (0.2). Never rewards distance from the familiar answer. 200 steps x 8 prompts x 8
  completions, lr 1e-5, KL beta 0.04, HF generation by default (`RL_VLLM=1` for colocated vLLM rollouts).
  Untested on the GPU: the first run is also the smoke test.

**real-point overfitting sweep:** real/val accuracy 69.0 / 68.0 / 71.0 / 70.3 at steps 100 / 200 / 350 / 442
(point F1 0.85 → 0.87); T0 counterfactual 59 / 65 / 63 / 62. Flat after ~25% of training: no overfitting
on held-out real photos, and no gain from training longer.

### 3.2b Base model on a small preview of the generated data (Oct 6, n ≈ 8 per cell, before later fixes)

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
8. **Real-photo counting pool (`real/count`, test `tests/R/count`)** comes from LVIS instance
   annotations: 3–15 instances of one category per square crop, with per-instance points and masks, so it
   supports point targets and point F1. Filters (2026-11-07, after two visual checks): manufactured objects
   and whole fruit only (no dishes, vegetables or slices), no instance touching the photo border, no
   fragmented, tiny, thin or heavily overlapping instances, a small deny list, at most 120 train / 25 test
   crops per category. LVIS val alone gives ~1,560 crops over 274 categories; train adds about 5x. In the
   last check about 1 in 6 items was still debatable (stacked buckets/plates, teddy-bear "casts"). Labels
   are LVIS counts, not SAM 3 recounts: run the Gate 4 audit on `tests/R/count` before reporting it.
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
| 1 | ~~E0b. Base-model baseline / Gate 0~~ **done 11-04** (§3.1) | `tools/eval_suite.sh /data/vlm` | 0.6 h | Baseline row of `results/RESULTS.md`; Gate 0 pass; conflict dose T0 42%, T2 color 1% |
| 2 (before the pilot) | **E0d. Pointing-prompt baseline / Gate 1**: same suite, prompt asks the model to point at every item, then give the total | `PROMPT_STYLE=point tools/eval_suite.sh /data/vlm` | 0.7 h | Row `Qwen3.5-4B+point`; does prompting alone fix it? Stated total vs point-list length, point F1 |
| 1 (overnight) | E0c. *Optional:* thinking-mode baseline on VLMBias main | `run_vlms_are_biased.py --thinking --max-new-tokens 16384` | 2–3 h | Does reasoning fix prior override for free? |
| 2 | **E1-pilot / Gate 2**: train C-count seed 0, check in-distribution val ≥ 95% | `tools/train_arm.sh /data/vlm conflict-count number 0` | 1.5 h | Training works (or not); first trained row |
| 3 | **E1 core 2×2, seed 0**: N-count, N-point, C-point (C-count done on day 2) | `train_arm.sh … neutral-count number 0`, `… neutral-point points 0`, `… conflict-point points 0` | 5 h | Contrasts 1–3 of the design doc |
| 4 | **E1 controls, seed 0**: N-enum, C-blind, B-count | `… neutral-list list 0`, `… conflict-blind number 0`, `… dice-only number 0` | 3.5 h | Contrast 4 (coordinates vs enumeration), Gate 3 (shortcut), narrow vs broad |
| 5 | **E2 count pilot, seed 0**: C-L0 (stop signs + numeral clocks, the two highest-dose families) | `… C-L0 number 0` | 1.5 h | Does narrow anchor-family conflict data transfer to VLMBias / T0? (The color pilot was dropped: base color bias is 1%, so there is nothing to fix; see §3.1.) |
| 5–6 (overnight) | **Seeds 1 (and 2)** for the core 2×2 | same as day 3 with seed 1, 2 | 6–12 h | Error bars |
| 6 | E1 conditional: N-randpoint, only if N-point > N-enum | `… neutral-randpoint randpoints 0` | 1.5 h | Do correct point locations matter? |
| 7 | **Write-up** | `uv run tools/summarize_results.py results`; data report | none | Tables + figures below |

**If time runs short:** cut, in order, E0c → day 6 → the seed 2 runs. The minimum showable set is E0b,
the four core arms at seed 0, and C-blind.

### 5.0 Pointing is central (revised 2026-11-07 after the Gate 1 result)

The pointing-prompt baseline (§3.2) shows the base model can count *against* the prior when it lists
the items: 37.2% list-length accuracy vs 12.7% direct, bias 27% vs 73%, and animals 0% → 73%. That
ability just isn't the default, and it costs accuracy on unedited images (97% → 81%) and on dense layouts
(chess, boards, grids). The design doc's H3 ("point-then-count training transfers more") becomes:

| Hypothesis | Claim | Test | Status |
|---|---|---|---|
| **H3a: grounding is latent** | The model can count against the prior when it enumerates; the failure is the direct readout | base direct vs base + pointing prompt | **supported** (§3.2) |
| **Mechanism: where the prior re-enters** | Either the list is right and the final number reverts to the prior (late readout), or the list itself collapses to the familiar length | base `+pointT` (pointing prompt that must end with `Total: {N}`): list length vs stated total, and their agreement split by canonical vs counterfactual | queued |
| **H3b: training makes grounding default and accurate** | Point-then-count training beats prompted pointing on counterfactuals *and* recovers canonical accuracy and dense layouts | C-point / N-point vs base `+point`, on accuracy, canonical and balanced accuracy | queued |
| **H3c: grounding distills into the direct answer** | Training on the same images with both targets (point list + total, and direct `{N}`) improves the *direct* answer beyond C-count's, with no pointing at inference | `conflict-mixed` / `neutral-mixed` arms evaluated with the `direct` prompt, vs C-count `+direct` and base `+direct`; also C-point / N-point `+direct` (does pointing training alone carry over?) | queued |
| H1/H2 (unchanged) | Conflict vs neutral data | C-* vs N-* within each format | partly (C-count, C-blind done) |

**New pieces for these tests:**
- **Prompt variants** (shared by eval and training exports, in `answer_parsing.py`): `point` (the Gate 1
  wording), `point-total` (must end with `Total: {N}`) and `direct` (number only, no listing). Results are
  saved as `<model>+point`, `+pointT` and `+direct`.
- **Mixed-format arms:** `conflict-mixed` and `neutral-mixed`. Same images as C-point / N-point; half the
  examples use the `point-total` prompt with a point-list + total target, half the plain question with
  `{N}`.

**Why mixed rather than self-distillation from the model's own points:** on synthetic data the ground
truth is known, so distilling the model's own correct point counts into `{N}` would reproduce C-count's
targets. What the mixed arms test is whether supervising the grounded format on the same images transfers
to the direct answer. Self-distillation becomes the interesting version for unlabeled real photos (L2/L3).

**Queue** (`tools/queue_experiments.sh`), in this order:
1. Base `+pointT`, base `+direct`.
2. C-point (+ `direct`), N-point (+ `direct`).
3. C-mixed (+ `direct`, `point-total`), N-mixed (+ `direct`), C-count `+direct`.
4. N-count, N-enum, B-count, C-L0.
5. Seeds 1 and 2 for C-point, C-mixed, C-count, N-point, N-mixed, N-count.

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
5. **Study 2 first look:** the real-photo color data and the finding that Qwen3.5-4B has no color prior
   override (T2 bias 1%), plus the C-L0 anchor-family arm.
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
uv run python tools/results_report.py --data /data/vlm   # plots + result-driven samples -> results/report.tar.gz
```
`results_report.py` writes, per model:
- **Plots:** VLMBias accuracy vs bias by topic; canonical vs counterfactual accuracy per family; where
  counterfactual answers land (correct / familiar / other); answer offsets; shared-block accuracy vs
  count; color confusion; and a model comparison.
- **Sample cards,** grouped by outcome: prior override, correct counterfactual, other error, canonical
  failure, unparsed, shared-block miss, VLMBias bias-aligned. Each card shows canonical | counterfactual |
  pixel diff of the edit, with ground-truth points, the model's answers and the end of its reply.

It warns if a results file is older than the data it's matched to, since the sample images could then
differ from what the model saw. Copy it with `scp <host>:<repo>/results/report.tar.gz .`.
- **The suite:** VLMBias main, VLMBias `original` (guardrail), T0 count, T2 color, and synthetic
  validation. All runs use greedy decoding, thinking off, and the same 4,096-token cap.
- **Knobs:** `EVAL_GPU_UTIL=0.5` when the GPU is shared; `EVAL_MAX_TOKENS` changes the cap;
  `PROMPT_STYLE=point` uses the pointing prompt and saves results as `<model>+point`.
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
- **Micro-batch:** chosen automatically as the largest that fits the free GPU memory (count answers
  up to 32 per pass, point answers about 8), with gradient accumulation keeping the effective batch at 32.
  On out-of-memory it halves and resumes from the last checkpoint. `BATCH=N` fixes it; `SKIP_EVAL=1`
  skips the eval.
- **Untested:** this is the first run of this stack. If ms-swift rejects a flag, check
  `swift sft --help` for 4.6 and adjust in `tools/train_arm.sh`.
  - **Gate 2 pass:** synthetic val accuracy ≥ 95% on conflict pairs (the "Val conflict" column in
    `RESULTS.md`); the shared block is reported separately (base 9%).
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
