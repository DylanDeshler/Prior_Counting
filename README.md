# Prior_Counting

## Evaluation

`run_vlms_are_biased.py` evaluates Qwen3.5-4B (and LoRA adapters) on VLMBias. See its docstring.

## Data generation (L0: Exp 1 + Exp 2 rendered level)

Implements `data_generation_spec.md` for everything rendered: record schema, seeding, compression,
the 9 Exp 1 conflict families + neutral twins, shared range block, dice pool, synthetic validation,
T0 (traffic lights, snowflakes, held-out color objects), Exp 2 L0 count (stop signs, numeral clocks,
emoji crowds) and color (emoji recolor), exporters, and the §10.1/§10.2 checks. L1–L3, T1–T3 and the
probe sets are not built yet.

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
uv run python -m datagen sources      # VLMBias stats (sets U), Open Images backgrounds (streams a 2.3 GB csv),
                                      # CoDa, Noto Emoji, LVIS/COCO names, Visual CounterFact names, color-name table
uv run tools/check_length.py          # §3.5 length rule with both tokenizers (CPU); lowers U if needed
uv run python -m datagen build        # all L0 pools -> data/{shared_range,e1,tests/T0,e2/L0}
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

`build` takes step names to rebuild part of the data: `shared`, `e1`, `t0`, `e2`.
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
