# Prior_Counting

## Evaluation

`run_vlms_are_biased.py` evaluates Qwen3.5-4B (and LoRA adapters) on VLMBias, or on generated data
with `--records <data dir | pool dirs | records.jsonl>` (e.g. `--records data/preview`,
`--records data/tests/T0/count --limit 400`). See its docstring.

## Data generation

Implements `data_generation_spec.md` for:
- **L0 (rendered):** record schema, seeding, compression, the 9 Exp 1 conflict families + neutral twins,
  shared range block, dice pool, synthetic validation, T0 count (traffic lights, snowflakes), Exp 2 L0
  count (stop signs, numeral clocks), exporters, and the §10.1/§10.2 checks.
- **L2 color (real photos):** COCO 2017 / LVIS v1 photos chosen by the §5.5 rules, LVIS masks, CIELAB
  recolor with L fixed and identity-recolor null edits, T2 color from held-out objects on reserved
  photos, and the §9 leakage controls (POPE/ORIC image ids, pHash near-duplicates of eval images).
  An object must read as one color over >= 70% of its surface (shadows and highlights discounted),
  so "what color is the X?" has one answer.

Emoji are not used (rejected as unrealistic, 2026-10-06), so there is no L0 color and no L0 neutral
crowds. Not built yet: L1, L2 count, L3, T1/T3, probe sets.

`sources` downloads (once): LVIS annotations (~0.4 GB), COCO annotations (0.25 GB), the Open Images box
file (streamed, 2.3 GB), eval images for the pHash list, and later only the COCO photos actually used
(roughly 5 GB in data/sources/coco/images).

Rendering runs on CPU (all cores but two, so it can share the box with eval/training). The two
model-based steps use vLLM with ~30% of GPU memory.

Quick look first (small version of every pool + report in `data/preview/`, ~3 min on CPU, no big downloads):

```bash
uv sync
uv run python -m datagen preview
uv run python -m datagen --data data/preview view     # or scp data/preview/reports.tar.gz
```

Full build:

```bash
uv sync
uv run python -m datagen sources      # VLMBias stats (sets U), Open Images backgrounds, COCO/LVIS annotations,
                                      # CoDa x LVIS color objects, eval-set exclusions + pHashes, color-name table
uv run tools/check_length.py          # §3.5 length rule with both tokenizers (CPU); lowers U if needed
uv run python -m datagen build        # L0 pools + L2 color -> data/{shared_range,e1,tests/T0,e2/L0,e2/L2,tests/T2}
uv run tools/point_format_probe.py    # §12: which point_2d convention Qwen3.5-4B emits (GPU)
uv run tools/blind_check.py           # §10.2 blind solvability / §10.3 text-only leak (GPU)
uv run python -m datagen export       # data/exports/<arm>/<model>/<format>/{train,val,train_rl}.jsonl
uv run python -m datagen check        # §10.1 CI checks -> data/reports/ci_report.json (non-zero exit on failure)
uv run python -m datagen report       # stats + plots, galleries, contact sheets, CI results -> data/reports.tar.gz
uv run --group dev pytest tests       # fast generator tests
```

### Looking at the data on the hosted GPU box

Two ways, both over SSH:

- **Interactive viewer** (browse with overlays, pair diffs, flag problems, blind click-to-count
  audit for Gate 4, stats). It listens on localhost only; tunnel it:
  ```bash
  # on the GPU box (inside tmux/screen so it survives disconnects)
  uv run python -m datagen view            # port 7860
  # on your laptop
  ssh -N -L 7860:localhost:7860 <user>@<gpu-host>     # then open http://localhost:7860
  ```
  VS Code Remote-SSH forwards the port automatically. Flags go to `data/reports/review_flags.jsonl`,
  audit answers to `data/reports/audit_log.jsonl`.
- **Static report**, no server: stats and plots, CI results, render-audit contact sheets and
  per-generator galleries (item + pair partner with points, masks, answers and checks).
  ```bash
  uv run python -m datagen report          # on the GPU box -> data/reports.tar.gz (~50 MB)
  scp <user>@<gpu-host>:<repo>/data/reports.tar.gz . && tar xzf reports.tar.gz && open reports/index.html
  ```

Long steps (`sources`, `build`) should run inside tmux/screen so an SSH drop doesn't kill them.

`build` takes step names to rebuild part of the data: `shared`, `e1`, `t0`, `e2`, `l2`.
Measured parameters (§12) live in `data/params.json`: shared-block upper bound U, the Qwen3.5
point convention, and the Exp 2 matched N (1,024 until the L3 yield is known; the N-pair arms are
prefixes of the 4N pools, so changing N only needs a re-export). Every command and tool takes `--data PATH` to choose where data is generated and read (default
`./data`; `$COUNTERPOINT_DATA` also works), e.g. `uv run python -m datagen --data /mnt/big/counterpoint build`
and `uv run tools/blind_check.py --data /mnt/big/counterpoint`. Exports store absolute image paths, so
re-export if you move the directory.

Every record holds the full render parameters; `check` regenerates random records and compares
image hashes. Each count is verified by an independent pixel recount of a clean render, and each
color by naming the recolored pixels; failing items are resampled and the rejections are logged in
`data/reports/build_*.json`.
