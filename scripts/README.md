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
├── v2_train.slurm    # fully-V2: [embed-align ->] SFT (V2 hyper-parameters)
├── v2_eval.slurm     # fully-V2: same eval protocol on the V2 run
├── uaware_env.sh     # user-aware (GenUP x SID): PROFILE_MODE + V2/runs/_uaware layout
├── uaware_data.slurm # user-aware: SID build + SID-aware profiles + fused/align data
├── uaware_train.slurm# user-aware: [embed-align (POI[+profile]) ->] SFT w/ profile
└── uaware_eval.slurm # user-aware: eval on the fused test set + stratified Acc@1
```

The fine-tune/eval/data-build Python drivers live in `V1/code/`:
`finetune_llm.py`, `eval_llm.py`, `build_dataset.py`, plus the V2-pipeline
helper `build_align_data.py` (`merge_adapter.py` is also present for ad-hoc
LoRA merges, but the V2 align path no longer needs it). The user-aware pipeline
adds `rewrite_profile_sid.py` (offline SID-aware profiles), `build_uaware_data.py`
(inject a profile into each record) and `strat_eval.py` (cold-start / rare-POI
breakdown).

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
v2_train.slurm  Phase A  align:  add SID tokens, train ONLY the embeddings on the
                                 align data -> a full SID-aware base model
                Phase B  SFT:    LoRA(q,k,v,gate,up) on that base, lr 2e-5, 5 ep, len 3072
v2_eval.slurm   beam-search Acc@k / MRR / NDCG@k on llm_test.json
```

```bash
DATASET=tky sbatch scripts/v2_data.slurm    # SID build (no-op if done) + align data
DATASET=tky sbatch scripts/v2_train.slurm   # align -> SFT (resumable)
DATASET=tky sbatch scripts/v2_eval.slurm    # metrics -> V2/runs/<run>/eval/
```

The V2 recipe treats the SID atoms as **new vocabulary tokens** (the paper, §4.2,
integrates SIDs into the LLM vocabulary; they are OOV and need the alignment
stage to learn their embeddings). So the V2 pipeline forces `ADD_SID_TOKENS=1`
when aligning. Set `V2_ALIGN=0` to skip alignment (= the authors'
`sft_without_alignment.py`, LoRA-tuning `q,k,v,o,gate,up` directly on the raw
base model); pair it with `ADD_SID_TOKENS=1` to still get atomic SID tokens.

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

## User-aware pipeline — GenUP × GNPR-SID (`uaware_*.slurm`)

Fuses the **user side** of GenUP (SIGSPATIAL'25) with the **POI side** of
GNPR-SID, following **GenUP's method**: the user's *full* history is summarised
**offline** into the profile, and the **prompt carries only the recent
trajectory** (`HISTORY_KEEP_LAST` visits) — not the full history. Both train on
the same `w11wo/LLM4POI` data, so user/POI ids line up. GenUP's per-user profile
(traits, demographics, preferences, routines, a narrative) is made **SID-aware**
— its raw "POI id N" mentions are rewritten to SID tokens via the codebook, and
the user's top SIDs are appended — then injected into the prompt in a
truncation-protected `### User Profile:` block. The alignment stage additionally
grounds **profile → SID** priors. This setup is built to study **whether
Semantic IDs help the cold-start regime** under GenUP's method: with the full
history removed from the prompt, accuracy on thin-history users depends on what
the (SID-aware) profile can carry. Compare against GenUP's own repo (raw POI
ids) for the no-SID baseline; sweep `HISTORY_KEEP_LAST` (e.g. `-1, 15, 8, 3, 0`)
to trade raw trajectory for profile.

```
uaware_data.slurm   SID build (+ poi_info/codebook) -> SID-aware profiles
                    (rewrite_profile_sid.py) -> fused llm_{train,test}_uaware[_raw].json
                    -> POI[+profile]<->SID alignment data
uaware_train.slurm  Phase A embed-align (POI[+profile] pairs) -> Phase B SFT w/ profile
uaware_eval.slurm   beam-search Acc@k/MRR/NDCG + strat_eval.py (cold users / rare POIs)
```

Three ablation arms, selected by `PROFILE_MODE`. Each arm + history window gets
its own files (`llm_*_uaware[_raw|_none]_h<K>.json`) and run dir
(`V2/runs/<ds>_<model>_uaware_<mode>_h<K>`), so a recency sweep never collides:

| arm | `PROFILE_MODE` | prompt | profile | alignment | tests |
|---|---|---|---|---|---|
| **B0** | `none` | recent trajectory | — | POI-only | recent-trajectory baseline (no profile) |
| **B1** | `raw` | recent traj. + profile | GenUP raw (POI-id) | POI-only | user side **without** SID-awareness |
| **B2** | `sid` | recent traj. + profile | SID-rewritten (+ affinity line) | POI **+ profile↔SID** | full fusion (ours) |

```bash
# default base model is meta-llama/Meta-Llama-3-8B (gated -> accept the license and
# export HF_TOKEN before prepare_env). Profiles come from a sibling GenUP checkout:
# ../GenUP/data/<ds>/user_profiles/
# NYC ships prebaked llm_*.json but no codebook/poi_info -> force a from-raw build first:
rm -f V1/datasets/nyc/llm_*.json
PREBAKED_DATASETS="" DATASET=nyc sbatch scripts/data.slurm

# HISTORY_KEEP_LAST=8 by default (GenUP recent-trajectory). Set it (same value on
# data + train + eval) to change the window; -1 keeps the full GNPR-SID history.
DATASET=nyc sbatch scripts/uaware_data.slurm
for m in none raw sid; do
  PROFILE_MODE=$m DATASET=nyc sbatch scripts/uaware_train.slurm
done
# then uaware_eval.slurm per PROFILE_MODE; compare B0/B1/B2 (overall + stratified)
# smaller-model smoke test: add  BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct  to each line.
```

Notes:
- **8B default.** The base model defaults to `meta-llama/Meta-Llama-3-8B`. The
  SFT/align micro-batches default to 8B-safe values on a 48G GPU (per-device 2 /
  4, effective batch preserved at 16 / 32 via `V2_SFT_ACCUM` / `V2_ALIGN_ACCUM`).
  If eval OOMs at length 3072 with 10 beams, drop `EVAL_BATCH_SIZE` to 2; raise
  the per-device batches for smaller models.
- **Disk: the alignment phase saves a FULL model, not a LoRA adapter.**
  `--tuning embed` trains the whole embedding matrix, so each checkpoint is the
  entire ~16 GB 8B model (+ optimizer); rotation transiently needs ~2×. This is
  the one heavy write the fusion adds over plain LoRA GenUP/GNPR-SID runs. The
  align phase therefore defaults (`ALIGN_SAVE_STEPS` huge) to **no intermediate
  checkpoints** — it writes the full model once at the end → `align/final`
  (~16 GB, the SFT base). Budget ~16 GB per arm for `align/final` (B0 and B1 use
  identical POI-only alignment, so you can reuse one across them); point
  `RUN_DIR` at scratch if your home quota is tight.
- **SFT keeps training the SID embeddings (don't freeze them).** Whenever SID
  tokens are used, `embed_tokens`/`lm_head` stay in `modules_to_save` so the
  *recommendation* objective keeps steering the output head — freezing them
  (relying on alignment alone) collapses Acc@1 to ~0. That makes each checkpoint
  carry a ~2 GB embedding matrix, so the uaware SFT defaults to `SAVE_ONLY_MODEL=1`
  (drops the ~8 GB optimizer state) and `SAVE_TOTAL_LIMIT=2` — a few GB total
  instead of the ~50 GB that `SAVE_TOTAL_LIMIT=5` + optimizer produced.
- **Offline / reproducible.** Profiles are rewritten deterministically from
  GenUP's committed JSON — no OpenAI key or network. Point `GENUP_DIR` /
  `GENUP_PROFILES_DIR` at GenUP's `data/<ds>/user_profiles`.
- `EVAL_MAX_NEW_TOKENS` defaults to **32** here (atomic 4-atom SIDs need ~5
  tokens; the old 12 truncated them). The profile block is protected so only the
  middle of a long history is dropped — the profile and the query both survive.
- Like V2, no published reference numbers exist; the result is the internal
  **B2 > B1 > B0** ablation (largest gap expected on cold-start users / rare POIs).

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
