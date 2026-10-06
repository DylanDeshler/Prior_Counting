# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4",
#     "torchvision",
#     "transformers>=5.2.0",
#     "accelerate",
#     "peft>=0.17",
#     "datasets",
#     "pillow",
#     "safetensors",
#     "tqdm",
#     # GPU speedups (Linux/CUDA): vLLM engine, and prebuilt Hub kernels (FlashAttention 2/3,
#     # flash-linear-attention, causal-conv1d) for the HF backend -- nothing to compile.
#     "vllm>=0.30; sys_platform == 'linux'",
#     "kernels>=0.17,<0.18; sys_platform == 'linux'",
# ]
# ///
"""Evaluate Qwen3.5-4B (and LoRA adapters on top of it) on "Vision Language Models are Biased".

Dataset: https://huggingface.co/datasets/anvo25/vlms-are-biased
Paper:   https://arxiv.org/abs/2505.23941

Each item asks a counting / yes-no question about a counterfactual image (e.g. a
chess board with 9 rows) and instructs the model to answer in curly brackets.
We report:
  - accuracy:   extracted answer == ground_truth
  - bias_ratio: extracted answer == expected_bias (the "prior knowledge" answer)

Backends (GPU speedups are on by default for both):
  vllm  Fastest; the default on CUDA. Continuous batching, FlashAttention, Triton kernels for
        the Gated DeltaNet (linear-attention) layers, CUDA graphs, chunked prefill. Base model
        and all LoRAs share one engine and are batched together.
  hf    transformers + PEFT with padded batches. On CUDA: FlashAttention 3 (Hopper) / 2
        (Ampere/Ada) plus flash-linear-attention and causal-conv1d kernels, all prebuilt from
        the HF Hub via `kernels`. Also runs on CPU/MPS, and handles any adapter PEFT can load
        (e.g. `modules_to_save`, DoRA) that vLLM can't.

Usage:
    # base model only
    uv run run_vlms_are_biased.py

    # base model + every PEFT adapter found under checkpoints/ (each dir with adapter_config.json)
    uv run run_vlms_are_biased.py --adapters checkpoints/

    # specific adapters, skip re-running the base model
    uv run run_vlms_are_biased.py --adapters ckpt/lora_a ckpt/lora_b --skip-base

    uv run run_vlms_are_biased.py --backend hf --batch-size 32

    uv run run_vlms_are_biased.py --limit 50           # quick smoke test
    uv run run_vlms_are_biased.py --thinking           # thinking mode (much longer outputs)

A fully fine-tuned / merged checkpoint is just `--model path/to/checkpoint`.

Outputs go to <output-dir>/<variant>_<split>_<mode>.jsonl (+ _summary.json) per variant and
a comparison.json across variants. Predictions are written incrementally, so an interrupted
run resumes where it left off when re-launched with the same arguments; pass --rerun to
discard existing predictions and regenerate everything.

Generation length is only bounded by the model's context window (262k tokens for Qwen3.5)
unless --max-new-tokens is given.

Generated data (datagen): pass --records with a data directory (every pool under it), pool
directories, or records.jsonl files. Ground truth is each record's answer and the "bias" answer
is its familiar answer; bias ratio is computed over counterfactuals (where the familiar answer is wrong).
Results are grouped by pool and by pool · family · role, so canonical vs counterfactual
accuracy shows up per family. With --records, --limit takes the first N items of each pool.

    uv run run_vlms_are_biased.py --records data/preview              # every preview pool
    uv run run_vlms_are_biased.py --records data/e1/conflict data/tests/T0/count --limit 400
"""

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from answer_parsing import extract_answer, matches

DATASET_ID = "anvo25/vlms-are-biased"
SPLITS = ["main", "identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]
VLLM_LORA_RANKS = [1, 8, 16, 32, 64, 128, 256, 320, 512]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Base model (HF id or local path).")
    p.add_argument("--adapters", nargs="*", type=Path, default=[],
                   help="PEFT adapter dirs, or parent dirs searched recursively for adapter_config.json.")
    p.add_argument("--skip-base", action="store_true", help="Only evaluate the adapters.")
    p.add_argument("--split", default="main", choices=SPLITS)
    p.add_argument("--records", nargs="*", type=Path, default=None,
                   help="Evaluate generated data instead of VLMBias: data dir, pool dirs, or records.jsonl files.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    p.add_argument("--thinking", action="store_true", help="Enable Qwen3.5 thinking mode (off by default).")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Default: no limit beyond the model's context window.")
    p.add_argument("--rerun", action="store_true",
                   help="Ignore existing predictions and regenerate all questions (overwrites results).")
    p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="Prompts submitted per generate call (across all variants); results are saved after each.")
    g = p.add_argument_group("vllm")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    g.add_argument("--tensor-parallel-size", type=int, default=1)
    g.add_argument("--max-model-len", type=int, default=None,
                   help="Default: the model's full context, or the largest that fits in GPU memory.")
    g = p.add_argument_group("hf")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected by default).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[("overall", "")].append(r)
        groups[("topic", r["topic"])].append(r)
        groups[("sub_topic", r["sub_topic"])].append(r)

    def stats(rs):
        n = len(rs)
        with_bias = [r for r in rs if r["expected_bias"] not in (None, "")]
        return {
            "n": n,
            "accuracy": sum(r["correct"] for r in rs) / n,
            # Over items that have a prior answer (all of VLMBias; conflict pairs in generated data).
            "bias_ratio": sum(r["bias_aligned"] for r in with_bias) / len(with_bias) if with_bias else None,
            "unparsed": sum(r["pred"] == "" for r in rs) / n,
            "truncated": sum(r.get("truncated", False) for r in rs) / n,
        }

    return {
        "overall": stats(groups[("overall", "")]),
        "by_topic": {k: stats(v) for (lvl, k), v in sorted(groups.items()) if lvl == "topic"},
        "by_sub_topic": {k: stats(v) for (lvl, k), v in sorted(groups.items()) if lvl == "sub_topic"},
    }


def print_summary(name, summary):
    def row(label, s):
        bias = f"{100 * s['bias_ratio']:>8.2f}" if s["bias_ratio"] is not None else f"{'-':>8}"
        print(f"{label:<52} {s['n']:>6} {100 * s['accuracy']:>8.2f} {bias} "
              f"{100 * s['unparsed']:>8.2f} {100 * s['truncated']:>8.2f}")

    print(f"\n=== {name}")
    print(f"{'':<52} {'n':>6} {'acc%':>8} {'bias%':>8} {'unpars%':>8} {'trunc%':>8}")
    row("OVERALL", summary["overall"])
    print("-- by topic")
    for k, s in summary["by_topic"].items():
        row(k, s)
    print("-- by sub_topic")
    for k, s in summary["by_sub_topic"].items():
        row(k, s)


def print_comparison(summaries):
    topics = sorted({t for s in summaries.values() for t in s["by_topic"]})
    width = max(len(n) for n in summaries) + 2
    print(f"\n=== Comparison (accuracy % / bias ratio %)")
    print(f"{'variant':<{width}} {'overall':>13} " + " ".join(f"{t[:18]:>18}" for t in topics))
    def cell(st):
        if not st:
            return "-"
        bias = f"{100 * st['bias_ratio']:5.1f}" if st["bias_ratio"] is not None else "  -  "
        return f"{100 * st['accuracy']:5.1f} / {bias}"

    for name, s in summaries.items():
        cells = [cell(s["overall"])] + [cell(s["by_topic"].get(t)) for t in topics]
        print(f"{name:<{width}} {cells[0]:>13} " + " ".join(f"{c:>18}" for c in cells[1:]))


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def find_adapters(paths):
    """Expand each path into adapter dirs (dirs containing adapter_config.json)."""
    found = []
    for p in paths:
        if (p / "adapter_config.json").exists():
            found.append(p)
        else:
            sub = sorted(c.parent for c in p.rglob("adapter_config.json"))
            if not sub:
                raise SystemExit(f"No adapter_config.json found in {p}")
            found.extend(sub)
    # Name each adapter by its path relative to the common parent so checkpoints of
    # different runs (run_a/checkpoint-500, run_b/checkpoint-500) don't collide.
    if len(found) <= 1:
        # Keep the run name for bare checkpoint dirs (run_a/checkpoint-500 -> run_a__checkpoint-500).
        return {(f"{f.resolve().parent.name}__{f.name}" if f.name.startswith("checkpoint") else f.name): f
                for f in found}
    root = os.path.commonpath([f.resolve() for f in found])
    return {str(f.resolve().relative_to(root)).replace("/", "__"): f for f in found}


def adapter_info(path, base_model):
    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("peft_type", "LORA") != "LORA":
        raise SystemExit(f"{path}: peft_type={cfg.get('peft_type')} is not LoRA")
    base = cfg.get("base_model_name_or_path")
    if base and Path(base).name != Path(base_model).name:
        print(f"WARNING: {path} was trained on {base}, evaluating on {base_model}")
    ranks = [cfg.get("r", 8), *cfg.get("rank_pattern", {}).values()]
    keys = []
    weights = path / "adapter_model.safetensors"
    if weights.exists():
        from safetensors import safe_open
        with safe_open(weights, "pt") as f:
            keys = list(f.keys())
    return {
        "max_rank": max(ranks),
        "touches_vision": any(".visual." in k or k.startswith("visual.") for k in keys),
        "vllm_unsupported": [k for k in ("modules_to_save", "use_dora") if cfg.get(k)],
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class ChunkProgress:
    """Stands in for the tqdm bar vLLM creates per generate call: forwards each finished request to
    the run's overall bar, and shows vLLM's token throughput plus how many requests in the current
    call are still running (a long tail = a few long generations, not a hang)."""

    def __init__(self, bar):
        self.bar = bar

    def __call__(self, iterable=None, total=None, **_):
        # vLLM also wraps plain iterables (e.g. "Rendering conversations"); pass those through.
        if iterable is not None:
            self.bar.set_postfix_str("preparing prompts")
            return iterable
        self.total, self.n = total, 0
        self.bar.set_postfix_str(f"batch: {total} running")
        return self

    @property
    def format_dict(self):
        return self.bar.format_dict

    @property
    def postfix(self):
        return ""

    @postfix.setter
    def postfix(self, text):  # vLLM writes "est. speed input: ... toks/s, output: ... toks/s"
        out = text.split("output:")[-1].strip() if text else ""
        # vLLM sets this just before counting the request that finished, hence the -1.
        self.bar.set_postfix_str(f"batch: {max(0, self.total - self.n - 1)} running, out {out}", refresh=False)

    def update(self, k=1):
        self.n += k
        if self.n >= self.total:
            self.bar.set_postfix_str("batch done, saving", refresh=False)
        self.bar.update(k)

    def refresh(self):
        self.bar.refresh()

    def close(self):
        pass


def build_messages(ex):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": ex["image"].convert("RGB")},
            {"type": "text", "text": ex["prompt"]},
        ],
    }]


class VLLMBackend:
    def __init__(self, args, adapters, max_new_tokens):
        from vllm import LLM, SamplingParams

        infos = {name: adapter_info(path, args.model) for name, path in adapters.items()}
        for name, info in infos.items():
            if info["vllm_unsupported"]:
                raise SystemExit(f"Adapter {name} uses {info['vllm_unsupported']}, which vLLM can't serve; "
                                 "use --backend hf.")
        lora_kwargs = {}
        if adapters:
            max_rank = max(i["max_rank"] for i in infos.values())
            lora_kwargs = dict(
                enable_lora=True,
                max_loras=min(8, len(adapters)),
                max_cpu_loras=len(adapters),
                max_lora_rank=next(r for r in VLLM_LORA_RANKS if r >= max_rank),
                enable_tower_connector_lora=any(i["touches_vision"] for i in infos.values()),
            )
        self.llm = LLM(
            model=args.model,
            dtype="bfloat16",
            max_model_len=args.max_model_len or -1,  # -1: full context if it fits, else largest that does
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
            # Qwen3.5's built-in multi-token-prediction head as the draft model (exact under greedy
            # decoding). Skipped when serving LoRAs: the MTP head isn't adapted, so drafts would mostly
            # be rejected, and spec decode + LoRA isn't a well-trodden path in vLLM.
            speculative_config=None if adapters else {"method": "mtp", "num_speculative_tokens": 2},
            **lora_kwargs,
        )
        # Greedy decoding, matching the lmms-eval config for this benchmark.
        # max_tokens=None generates until EOS or the context limit.
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.chat_kwargs = {"enable_thinking": args.thinking}
        self.adapters = adapters
        self.lora_ids = {name: i + 1 for i, name in enumerate(adapters)}

    def generate(self, examples, variants, bar):
        from vllm.lora.request import LoRARequest

        loras = [None if v is None else LoRARequest(v, self.lora_ids[v], str(self.adapters[v].resolve()))
                 for v in variants]
        convs = [[{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": ex["image"].convert("RGB")},
                {"type": "text", "text": ex["prompt"]},
            ],
        }] for ex in examples]
        outputs = self.llm.chat(convs, self.sampling, lora_request=loras,
                                chat_template_kwargs=self.chat_kwargs, use_tqdm=ChunkProgress(bar))
        return [(o.outputs[0].text.strip(), o.outputs[0].finish_reason == "length") for o in outputs]


class HFBackend:
    def __init__(self, args, adapters, max_new_tokens):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if args.device:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        cuda = device.startswith("cuda")
        major = torch.cuda.get_device_capability(device)[0] if cuda else 0
        attn = "flash_attention_3" if major >= 9 else "flash_attention_2" if major >= 8 else "sdpa"
        # If the flash-attn package isn't installed, transformers loads the prebuilt FlashAttention
        # kernel from the Hub instead (needs `kernels`); use_kernels does the same for the Gated
        # DeltaNet (fla) and causal-conv1d ops.
        load_kwargs = dict(
            dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
            attn_implementation=attn,
            use_kernels=cuda,
        )

        self.processor = AutoProcessor.from_pretrained(args.model, padding_side="left")
        try:
            self.model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)
        except (ImportError, ValueError, OSError) as e:
            if attn == "sdpa" and not load_kwargs["use_kernels"]:
                raise
            print(f"WARNING: fast kernels unavailable ({e}); falling back to sdpa without Hub kernels.")
            load_kwargs.update(attn_implementation="sdpa", use_kernels=False)
            self.model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)
        self.model.eval()
        print(f"HF backend: device={device}, attention={self.model.config._attn_implementation}, "
              f"hub_kernels={load_kwargs['use_kernels']}")
        for name, path in adapters.items():
            adapter_info(path, args.model)  # validation / warnings only
            self.model.load_adapter(str(path), adapter_name=name)
        self.has_adapters = bool(adapters)
        self.max_new_tokens = max_new_tokens
        self.context_len = self.model.config.get_text_config().max_position_embeddings
        self.thinking = args.thinking
        self.batch_size = args.batch_size

    def generate(self, examples, variants, bar):
        results = [None] * len(examples)
        for v in dict.fromkeys(variants):
            if self.has_adapters:
                if v is None:
                    self.model.disable_adapters()
                else:
                    self.model.enable_adapters()
                    self.model.set_adapter(v)
            idx = [i for i, vi in enumerate(variants) if vi == v]
            for i, r in zip(idx, self._generate([examples[i] for i in idx], bar)):
                results[i] = r
        return results

    def _generate(self, examples, bar):
        results = [None] * len(examples)
        # Sort by image size so each batch has similar sequence lengths (less padding).
        order = sorted(range(len(examples)), key=lambda i: examples[i]["pixel"])
        eos = self.model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) | {self.processor.tokenizer.pad_token_id}
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            inputs = self.processor.apply_chat_template(
                [build_messages(examples[i]) for i in idx],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
                enable_thinking=self.thinking,
            ).to(self.model.device)
            limit = self.max_new_tokens or self.context_len - inputs["input_ids"].shape[1]
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=limit,
                                          do_sample=False, temperature=None, top_p=None, top_k=None)
            gen = out[:, inputs["input_ids"].shape[1]:]
            for i, row in zip(idx, gen):
                # Finished rows end in EOS/padding; a row that hit the token limit ends mid-text.
                truncated = len(row) >= limit and row[-1].item() not in stop_ids
                results[i] = (self.processor.decode(row, skip_special_tokens=True).strip(), truncated)
            bar.update(len(idx))
        return results


def pick_backend(name):
    if name != "auto":
        return name
    return "vllm" if torch.cuda.is_available() and importlib.util.find_spec("vllm") else "hf"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_generated(paths, limit):
    """Generated-data records -> a dataset with the same columns as VLMBias."""
    from datasets import Dataset, Image
    files = []
    for p in paths:
        if p.is_file():
            files.append(p)
        else:
            # A data dir holds many pools; skip nested previews/exports unless pointed at directly.
            files += sorted(f for f in p.rglob("records.jsonl")
                            if not {"preview", "exports", "sources"} & set(f.relative_to(p).parts[:-1]))
    if not files:
        raise SystemExit(f"No records.jsonl found under {[str(p) for p in paths]}")
    rows = []
    for f in files:
        root = f.parent
        while not (root / "params.json").exists() and root != root.parent:  # data root holds params.json
            root = root.parent
        recs = [json.loads(line) for line in f.open()]
        by_pool = defaultdict(list)
        for r in recs:
            by_pool[r["dataset"]].append(r)
        for pool, rs in by_pool.items():
            for r in rs[:limit] if limit else rs:
                fam = r["family"]
                rows.append({
                    "ID": r["id"], "image": str((root / r["image"]) if (root / r["image"]).exists() else f.parent / r["image"]),
                    "topic": pool, "sub_topic": f"{pool} · {fam} · {r['role']}",
                    "type_of_question": r["question_template"], "pixel": 448, "prompt": r["question"],
                    "ground_truth": str(r["answer"]),
                    # Only where the prior answer is wrong (counterfactuals), as in VLMBias.
                    "expected_bias": None if r["familiar_answer"] in (None, r["answer"]) else str(r["familiar_answer"]),
                })
    print(f"Generated data: {len(rows)} items from {len(files)} records files")
    return Dataset.from_list(rows).cast_column("image", Image())


def load_done(path):
    done = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                done[r["ID"]] = r
    return done


def main():
    args = parse_args()
    max_new_tokens = args.max_new_tokens  # None = up to the context limit
    adapters = find_adapters(args.adapters)
    variants = ([] if args.skip_base else [None]) + list(adapters)
    if not variants:
        raise SystemExit("Nothing to evaluate: --skip-base given without --adapters.")

    if args.records:
        ds = load_generated(args.records, args.limit)
        data_tag = "gen-" + "+".join(p.resolve().name for p in args.records)[:60]
    else:
        ds = load_dataset(DATASET_ID, split=args.split)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        data_tag = args.split
    ids = ds["ID"]
    meta_cols = ["ID", "topic", "sub_topic", "type_of_question", "pixel", "prompt", "ground_truth", "expected_bias"]

    mode = "think" if args.thinking else "nothink"
    base_name = args.model.rstrip("/").split("/")[-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Build names by string concat: Path.with_suffix would treat ".5-4B..." in "Qwen3.5-4B" as a suffix.
    stems = {v: f"{base_name if v is None else base_name + '+' + v}_{data_tag}_{mode}" for v in variants}
    pred_paths = {v: args.output_dir / f"{stems[v]}.jsonl" for v in variants}
    done = {v: {} if args.rerun else load_done(pred_paths[v]) for v in variants}

    if any(len(done[v]) < len(ids) for v in variants):
        backend_name = pick_backend(args.backend)
        print(f"Backend: {backend_name}, variants: {['base' if v is None else v for v in variants]}")
        # Only register adapters that still have work, so vLLM's max rank etc. reflect them.
        pending = {v: p for v, p in adapters.items() if len(done[v]) < len(ids)}
        backend = (VLLMBackend if backend_name == "vllm" else HFBackend)(args, pending, max_new_tokens)

        # One job per (item, variant), item-major so each chunk covers all variants and each
        # image is decoded once per chunk; vLLM batches the different LoRAs together.
        jobs = [(i, v) for i, id_ in enumerate(ids) for v in variants if id_ not in done[v]]
        files = {v: pred_paths[v].open("w" if args.rerun else "a") for v in variants}
        with tqdm(total=len(jobs), desc="generating", unit="q", dynamic_ncols=True, smoothing=0.05) as bar:
            for start in range(0, len(jobs), args.chunk_size):
                chunk = jobs[start:start + args.chunk_size]
                rows = sorted({i for i, _ in chunk})
                by_row = dict(zip(rows, ds.select(rows)))
                examples = [by_row[i] for i, _ in chunk]
                outputs = backend.generate(examples, [v for _, v in chunk], bar)
                for (_, v), ex, (response, truncated) in zip(chunk, examples, outputs):
                    pred = extract_answer(response)
                    record = {k: ex[k] for k in meta_cols}
                    record.update(
                        response=response,
                        pred=pred,
                        truncated=truncated,
                        correct=matches(pred, ex["ground_truth"]),
                        bias_aligned=ex["expected_bias"] not in (None, "") and matches(pred, ex["expected_bias"]),
                    )
                    files[v].write(json.dumps(record) + "\n")
                    done[v][ex["ID"]] = record
                for f in files.values():
                    f.flush()
        for f in files.values():
            f.close()

    summaries = {}
    for v in variants:
        name = "base" if v is None else v
        # Re-score from the stored responses so parser changes apply to earlier runs too.
        for r in done[v].values():
            r["pred"] = extract_answer(r["response"])
            r["correct"] = matches(r["pred"], r["ground_truth"])
            r["bias_aligned"] = r["expected_bias"] not in (None, "") and matches(r["pred"], r["expected_bias"])
        summary = summarize([done[v][id_] for id_ in ids])
        summary["config"] = {"model": args.model, "adapter": None if v is None else str(adapters[v]),
                             "data": data_tag, "thinking": args.thinking,
                             "max_new_tokens": max_new_tokens, "limit": args.limit}
        (args.output_dir / f"{stems[v]}_summary.json").write_text(json.dumps(summary, indent=2))
        print_summary(name, summary)
        summaries[name] = summary
    if len(summaries) > 1:
        print_comparison(summaries)
        (args.output_dir / f"comparison_{data_tag}_{mode}.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nResults in {args.output_dir}/")


if __name__ == "__main__":
    main()
