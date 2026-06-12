# GNPR-SID (V1) — single-GPU Slurm reproduction pipeline

Scripts to reproduce the **V1** (KDD 2025) *Generative Next POI Recommendation
with Semantic ID* result on one GPU, within a **6-hour** wall-clock budget, with
**resumable** jobs (re-submitting the same script continues / skips finished
work).

```
scripts/
├── env.sh            # shared config: modules, conda, run-dir layout, markers, data fetch
├── requirements.txt  # python deps (torch installed separately, CUDA-matched)
├── prepare_env.slurm # one-time: build the conda env + download raw data
├── data.slurm        # (TKY/CA only) build Semantic-ID llm_*.json from raw check-ins
├── train.slurm       # (optional) RQ-VAE ID-gen  +  LLM LoRA fine-tune
├── eval.slurm        # generative eval: Acc@k / MRR / NDCG@k
├── v2_env.sh         # fully-V2 pipeline: V2 recipe knobs + V2/runs/ layout
├── v2_data.slurm     # fully-V2: SID build + SID<->attribute alignment data
├── v2_train.slurm    # fully-V2: [align -> merge ->] SFT (V2 hyper-parameters)
└── v2_eval.slurm     # fully-V2: same eval protocol on the V2 run
```

The fine-tune/eval/data-build Python drivers live in `V1/code/`:
`finetune_llm.py`, `eval_llm.py`, `build_dataset.py`, plus the V2-pipeline
helpers `build_align_data.py` and `merge_adapter.py`.

## Quick start

```bash
# from the repo root
mkdir -p log                         # Slurm opens log/%x-%j.out before the script runs
sbatch scripts/prepare_env.slurm     # build env + fetch raw data (idempotent)
# DATASET=tky sbatch scripts/data.slurm   # TKY/CA ONLY: build llm_*.json (skip for nyc)
sbatch scripts/train.slurm           # fine-tune (resume by re-submitting)
sbatch scripts/eval.slurm            # evaluate (resume by re-submitting)
```

> **NYC needs no `data.slurm`** — it ships pre-baked `llm_*.json`. TKY/CA ship no
> JSON, so they go through `data.slurm` first (see *Reproducing TKY / CA* below).

> All three scripts write `%x-%j.out`/`.err` into `log/`. Create it once up
> front (Slurm opens those files at launch, before the script body's own
> `mkdir -p log` runs). To chain with dependencies:
> `sbatch --dependency=afterok:<train_jobid> scripts/eval.slurm`.

Defaults follow the paper (KDD'25 §5.2/§6.6): dataset `nyc`, base model
`meta-llama/Meta-Llama-3-8B`, LoRA (r=16, α=32, dropout=0.1), 8 epochs, LR 1e-5
with a constant schedule + 20 warm-up steps, effective batch 64. Override
anything via the environment, e.g.:

```bash
# quick open-model smoke test (no gated-repo access / less VRAM needed)
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct NUM_EPOCHS=1 GRAD_ACCUM=8 sbatch scripts/train.slurm
DATASET=tky sbatch scripts/eval.slurm
```

> **Gated base model.** `meta-llama/Meta-Llama-3-8B` requires accepting Meta's
> license on Hugging Face and an `HF_TOKEN` in the environment **before**
> `prepare_env` (it pre-fetches the weights). Without access, override
> `BASE_MODEL` with an open model.

## What each stage does

1. **prepare_env** — `module load anaconda + cuda/12.8.0`, create a project-local
   conda env at `.conda/gnpr-sid`, `pip install torch` (cu128 wheel, matches the
   `cuda/12.8.0` module → torch ≥2.7) + the rest,
   and pre-cache the base model into `.cache/huggingface`. It also **fetches the
   raw check-in data** for `DATASET` into `V1/datasets/<ds>/raw/` (no-op for nyc
   and for already-present data — see *Reproducing TKY / CA*). Re-running is a
   no-op once `.conda/gnpr-sid/.env_ready` exists (`FORCE_REBUILD=1` to rebuild);
   a re-run still tops up missing raw data for the current `DATASET`.

2. **train**
   - *Phase 1 — Semantic-ID generation (optional).* Trains the RQ-VAE
     (`train_rqvae.py`) and emits the codebook (`codebook.py`). This needs
     `V1/datasets/<DATASET>/poi_info.csv`, which is **not** shipped with the
     bundled NYC sample — the sample's SIDs are already baked into
     `llm_{train,val,test}.json`. So by default Phase 1 is **skipped** and marked
     done. Set `RUN_IDGEN=1` and provide `poi_info.csv` to actually run it.
   - *Phase 2 — LLM fine-tuning.* LoRA SFT of the base model on
     `llm_train.json` with the paper's recipe (Alpaca prompt, completion-only
     loss, bf16, gradient checkpointing, LR 1e-5 constant + 20 warm-up steps,
     effective batch 64). Checkpoints every `SAVE_STEPS` (50) with a sliding
     window of `SAVE_TOTAL_LIMIT` (5). The final adapter lands in
     `V1/runs/<run>/sft/final`.

3. **eval** — beam-search (top-`NUM_BEAMS`) Semantic-ID generation on
   `llm_test.json`, then **Acc@k**, **MRR**, **NDCG@k** (`EVAL_KS="1 5 10"`).
   Results: `V1/runs/<run>/eval/{predictions.jsonl,metrics.json}`.

## Reproducing TKY / CA

Only **NYC** ships pre-baked `llm_*.json`. **TKY** and **CA** ship nothing, so you
build their JSON from raw check-ins with `data.slurm`, which runs the **V2 CRQVAE
Semantic-ID module** (`V1/code/build_dataset.py`):

```
train_sample.csv ──▶ train seqs + poi_info ──▶ category emb (MiniLM+PCA)
   ──▶ POI feature vectors ──▶ CRQVAE train + SID emit ──▶ llm_train.json
test_qa_pairs_kqt.txt ─────────────────────────(map POIs→SID)────────────▶ llm_test.json
```

> ⚠️ V2 uses cosine-similarity quantisation + EMA, so the SIDs (and thus metrics)
> differ from the V1-paper RQ-VAE numbers. This is the runnable path; V1's RQ-VAE
> SID code is incomplete for fresh datasets (schema gaps in `poi_info.csv`).

**1 — Get the data (default: zero-config from Hugging Face).** The fetcher pulls
the public dataset [`w11wo/LLM4POI`](https://huggingface.co/datasets/w11wo/LLM4POI)
— the exact LLM4POI preprocessed check-ins — using the `HF_TOKEN` already wired in
(the repo is public, so the token is optional). `prepare_env` does this for you:

```bash
DATASET=tky sbatch scripts/prepare_env.slurm   # downloads tky/preprocessed/* into V1/datasets/tky/raw/
```

Per city it provides `train_sample.csv` (the **train** split; no `SplitTag`) and
`test_qa_pairs_kqt.txt` (the **test** set as text QA). The builder treats the CSV
as the train split and parses the `.txt` for the test set. **Note:** this layout
has **no validation split**, so `llm_val.json` is skipped (the paper trains a fixed
8 epochs and reports on test, so val isn't needed). To use a different source
instead, set `HF_DATA_REPO=<repo>`, or per dataset `RAW_URL_<DS>=<zip-url>` /
`RAW_GDRIVE_ID_<DS>=<id>`, or drop a combined `sample.csv` (with a `SplitTag`
column) into `V1/datasets/<ds>/raw/` by hand.

**2 — Build the JSON** (GPU job; CRQVAE on a few-thousand POIs takes minutes):

```bash
DATASET=tky sbatch scripts/data.slurm     # -> V1/datasets/tky/llm_train.json + llm_test.json
```

**3 — Train & eval as usual**, just carrying `DATASET`:

```bash
DATASET=tky sbatch scripts/train.slurm
DATASET=tky sbatch scripts/eval.slurm
```

`data.slurm` is idempotent: every stage skips when its output exists, and the job
is a no-op once `llm_*.json` is present.

## Fully-V2 pipeline (`v2_*.slurm`)

The default `train.slurm`/`eval.slurm` reproduce the **V1 paper recipe** (for
TKY/CA on top of V2-built SIDs). The `v2_*` scripts instead run the authors'
**V2 LLM recipe** (`V2/LLM/train/*` + `V2/dataprocess/get_align_data.ipynb`)
end-to-end, with its own run dir (`V2/runs/<ds>_<model>_v2`) and markers so V1
and V2 runs of the same dataset/model coexist:

```
v2_data.slurm   raw -> CRQVAE SIDs -> llm_*.json  AND  llm_align_{train,val}.json
                (alignment data: per POI, attributes<->SID instruction pairs)
v2_train.slurm  Phase A  align:  LoRA on embed_tokens ONLY, on the align data
                Phase B  merge:  alignment LoRA folded into the base model
                Phase C  SFT:    LoRA(q,k,v,gate,up), lr 2e-5, 5 epochs, len 3072
v2_eval.slurm   beam-search Acc@k / MRR / NDCG@k on llm_test.json
```

```bash
DATASET=tky sbatch scripts/v2_data.slurm    # SID build (no-op if done) + align data
DATASET=tky sbatch scripts/v2_train.slurm   # align -> merge -> SFT (resumable)
DATASET=tky sbatch scripts/v2_eval.slurm    # metrics -> V2/runs/<run>/eval/
```

Set `V2_ALIGN=0` to skip the alignment/merge stages (= the authors'
`sft_without_alignment.py`, which LoRA-tunes `q,k,v,o,gate,up` directly on the
raw base model).

Notes / caveats:

- **No published V2 reference numbers exist** (`V2/V2.md` reports none), so V2
  results can't be checked against the paper — the paper's table is V1.
- The stages are driven by our `finetune_llm.py` with the V2 hyper-parameters
  rather than `V2/LLM/train/*.py` as shipped: those scripts have blanked-out
  paths, hard-coded wandb logging, a Llama-3-only `<|eot_id|>` literal in the
  prompt, and no checkpoint resume (the V2 README itself recommends not using
  them directly). The recipe (LoRA targets, lr, epochs, batch, cutoff, warm-up)
  follows their `__main__` values; override via `V2_*` env knobs (see
  `v2_env.sh`).
- **Alignment needs `poi_info.csv` + `codebook.csv`** (left behind by the SID
  build). Prebaked NYC has neither — run NYC with `V2_ALIGN=0`, or force a
  from-raw V2 rebuild with `PREBAKED_DATASETS="" sbatch scripts/v2_data.slurm`.
- The authors' effective SFT batch is 8×2 = 16 (vs 64 in the V1 paper recipe);
  per-device batch 8 at length 3072 fits a 1.5B model on the L40S but may OOM
  with an 8B base — lower `V2_SFT_BS` (and raise `V2_SFT_ACCUM`) if so.

## Resume / "skip when finished" semantics

Each phase writes a marker under `V1/runs/<run>/markers/` (`idgen.done`,
`sft.done`, `eval.done`).

- Re-submitting **train.slurm**: if `sft.done` exists it exits immediately;
  otherwise the fine-tune **resumes from the newest `checkpoint-*`** (so a job
  killed at the 6h limit loses at most `SAVE_STEPS` (50) optimizer steps).
- Re-submitting **eval.slurm**: predictions stream to `predictions.jsonl`;
  a re-run skips the examples already written and continues, then recomputes
  metrics from the full file.

This is exactly the "submit the same job script; it resumes, and is a no-op once
finished" workflow.

## Key knobs (see `env.sh` for all)

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `nyc` | dataset under `V1/datasets/` |
| `BASE_MODEL` | `meta-llama/Meta-Llama-3-8B` | HF model id or local path (gated; see note above) |
| `TUNING` | `lora` | `lora` or `full` |
| `SAVE_STEPS` | `50` | checkpoint cadence (optimizer steps) |
| `SAVE_TOTAL_LIMIT` | `5` | sliding-window checkpoint count |
| `NUM_EPOCHS` / `LR` | `8` / `1e-5` | fine-tune schedule (paper §6.6/§5.2) |
| `LR_SCHEDULER` / `WARMUP_STEPS` | `constant_with_warmup` / `20` | paper §5.2 |
| `PER_DEVICE_BS` / `GRAD_ACCUM` | `2` / `32` | effective batch 64 (4×L40 paper → 1 GPU) |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | `16` / `32` / `0.1` | LoRA config (paper §5.2) |
| `MAX_SEQ_LEN` | `2048` | train/eval truncation length |
| `NUM_BEAMS` / `EVAL_KS` | `10` / `1 5 10` | eval beams & cut-offs |
| `ADD_SID_TOKENS` | `0` | `1` = add atomic `<a_*>/<b_*>/...` tokens (resizes embeddings, trains `embed_tokens`+`lm_head`); `0` mirrors the authors' code (SID as sub-words) |
| `PREBAKED_DATASETS` | `nyc` | datasets that ship `llm_*.json` (no download / no SID build) |
| `HF_DATA_REPO` | `w11wo/LLM4POI` | default raw-data source (HF dataset; uses `HF_TOKEN`) |
| `RAW_URL[_<DS>]` / `RAW_GDRIVE_ID[_<DS>]` | _(empty)_ | override raw-data source (direct link / Google-Drive id; per-dataset suffix wins, takes precedence over `HF_DATA_REPO`) |
| `SID_EPOCHS` | `3000` | CRQVAE training epochs (must be ≥210 to save a best checkpoint) |
| `SID_NUM_EMB` / `SID_E_DIM` | `64 64 64` / `64` | CRQVAE codebook sizes & embedding dim (3 books → `<a><b><c>`) |
| `CAT_MODEL` / `CAT_DIM` | `all-MiniLM-L6-v2` / `64` | category text encoder & PCA dim |
| `KEEP_LAST_K` | `5` | train: keep last K samples per user (V2 data recipe) |

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
