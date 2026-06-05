# GNPR-SID (V1) — single-GPU Slurm reproduction pipeline

Scripts to reproduce the **V1** (KDD 2025) *Generative Next POI Recommendation
with Semantic ID* result on one GPU, within a **6-hour** wall-clock budget, with
**resumable** jobs (re-submitting the same script continues / skips finished
work).

```
scripts/
├── env.sh            # shared config: modules, conda, run-dir layout, markers
├── requirements.txt  # python deps (torch installed separately, CUDA-matched)
├── prepare_env.slurm # one-time: build the conda env from cluster modules
├── train.slurm       # (optional) RQ-VAE ID-gen  +  LLM LoRA fine-tune
└── eval.slurm        # generative eval: Acc@k / MRR / NDCG@k
```

The fine-tune/eval Python drivers live in `V1/code/`:
`finetune_llm.py`, `eval_llm.py`.

## Quick start

```bash
# from the repo root
mkdir -p log                         # Slurm opens log/%x-%j.out before the script runs
sbatch scripts/prepare_env.slurm     # build env once (idempotent)
sbatch scripts/train.slurm           # fine-tune (resume by re-submitting)
sbatch scripts/eval.slurm            # evaluate (resume by re-submitting)
```

> All three scripts write `%x-%j.out`/`.err` into `log/`. Create it once up
> front (Slurm opens those files at launch, before the script body's own
> `mkdir -p log` runs). To chain with dependencies:
> `sbatch --dependency=afterok:<train_jobid> scripts/eval.slurm`.

Defaults: dataset `nyc`, base model `Qwen/Qwen2.5-1.5B-Instruct`, LoRA. Override
anything via the environment, e.g.:

```bash
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct NUM_EPOCHS=5 sbatch scripts/train.slurm
DATASET=tky sbatch scripts/eval.slurm
```

## What each stage does

1. **prepare_env** — `module load anaconda + cuda/12.8.0`, create a project-local
   conda env at `.conda/gnpr-sid`, `pip install torch` (cu128 wheel, matches the
   `cuda/12.8.0` module → torch ≥2.7) + the rest,
   and pre-cache the base model into `.cache/huggingface`. Re-running is a no-op
   once `.conda/gnpr-sid/.env_ready` exists (`FORCE_REBUILD=1` to rebuild).

2. **train**
   - *Phase 1 — Semantic-ID generation (optional).* Trains the RQ-VAE
     (`train_rqvae.py`) and emits the codebook (`codebook.py`). This needs
     `V1/datasets/<DATASET>/poi_info.csv`, which is **not** shipped with the
     bundled NYC sample — the sample's SIDs are already baked into
     `llm_{train,val,test}.json`. So by default Phase 1 is **skipped** and marked
     done. Set `RUN_IDGEN=1` and provide `poi_info.csv` to actually run it.
   - *Phase 2 — LLM fine-tuning.* LoRA SFT of the base model on
     `llm_train.json`, mirroring the authors' V2 recipe (Alpaca prompt,
     completion-only loss, bf16, gradient checkpointing). Checkpoints every
     `SAVE_STEPS` (200) with a sliding window of `SAVE_TOTAL_LIMIT` (5). The
     final adapter lands in `V1/runs/<run>/sft/final`.

3. **eval** — beam-search (top-`NUM_BEAMS`) Semantic-ID generation on
   `llm_test.json`, then **Acc@k**, **MRR**, **NDCG@k** (`EVAL_KS="1 5 10"`).
   Results: `V1/runs/<run>/eval/{predictions.jsonl,metrics.json}`.

## Resume / "skip when finished" semantics

Each phase writes a marker under `V1/runs/<run>/markers/` (`idgen.done`,
`sft.done`, `eval.done`).

- Re-submitting **train.slurm**: if `sft.done` exists it exits immediately;
  otherwise the fine-tune **resumes from the newest `checkpoint-*`** (so a job
  killed at the 6h limit loses at most ~200 steps).
- Re-submitting **eval.slurm**: predictions stream to `predictions.jsonl`;
  a re-run skips the examples already written and continues, then recomputes
  metrics from the full file.

This is exactly the "submit the same job script; it resumes, and is a no-op once
finished" workflow.

## Key knobs (see `env.sh` for all)

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `nyc` | dataset under `V1/datasets/` |
| `BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HF model id or local path |
| `TUNING` | `lora` | `lora` or `full` |
| `SAVE_STEPS` | `200` | checkpoint cadence (fine-tune) |
| `SAVE_TOTAL_LIMIT` | `5` | sliding-window checkpoint count |
| `NUM_EPOCHS` / `LR` | `3` / `2e-5` | fine-tune schedule |
| `PER_DEVICE_BS` / `GRAD_ACCUM` | `4` / `4` | effective batch 16 |
| `MAX_SEQ_LEN` | `2048` | train/eval truncation length |
| `NUM_BEAMS` / `EVAL_KS` | `10` / `1 5 10` | eval beams & cut-offs |
| `ADD_SID_TOKENS` | `0` | `1` = add atomic `<a_*>/<b_*>/...` tokens (resizes embeddings, trains `embed_tokens`+`lm_head`); `0` mirrors V2 (SID as sub-words) |

## Caveats / notes

- **GPU index / `cuda:7`.** The original `train_rqvae.py`/`codebook.py` default to
  `--device cuda:7`; the scripts override this with `--device cuda:0` for the
  single GPU Slurm allocates (`CUDA_VISIBLE_DEVICES` is set by Slurm).
- **RQ-VAE → llm_*.json mismatch.** `codebook.py` hard-codes `data_mode="TKY"`
  and writes columns `[Pid, Codebook, Vector]`, while
  `datasets/llm_dataprocess.ipynb` expects `[pid, sid]`. If you enable
  `RUN_IDGEN=1` for a *new* dataset, reconcile these (set the data mode, rename
  columns) before regenerating `llm_*.json`. For the bundled NYC sample this is
  moot — its SIDs are pre-baked.
- **RQ-VAE checkpointing.** `trainer.py` already keeps a sliding window of
  `--save_limit` (5) best/recent checkpoints but is **epoch-based** (no
  mid-run resume). RQ-VAE on a few-thousand POIs finishes in minutes, well
  inside one 6h job, so resume there is not needed. The 200-step cadence +
  resume requirement is satisfied for the long-running LLM fine-tune and eval.
- **Gated models.** `Qwen2.5-*` is open. For gated models (e.g. Llama-3) export
  `HF_TOKEN` before `prepare_env`.
```
