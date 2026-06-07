#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/env.sh
#
# Shared environment for the GNPR-SID (V1) reproduction pipeline.
# Sourced by prepare_env.slurm / train.slurm / eval.slurm.
#
# It is responsible for:
#   * locating the repo root,
#   * loading cluster *modules* (anaconda + cuda) instead of building from
#     scratch,
#   * activating a project-local conda environment,
#   * defining all the paths / run-dir layout used by the pipeline,
#   * providing tiny "phase marker" helpers so a re-submitted job skips the
#     phases that already finished (resume-ability).
#
# Everything is overridable from the environment, e.g.:
#   DATASET=tky BASE_MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/train.slurm
# ---------------------------------------------------------------------------

# Resolve the directory this file lives in (scripts/) and the repo root.
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_ENV_SH_DIR}/.." && pwd)}"

# --------------------------- user-tunable knobs ----------------------------
export DATASET="${DATASET:-nyc}"                       # nyc | tky | ca (sample ships with nyc)
# Paper (KDD'25 §5.2) fine-tunes LLaMA3-8B. NOTE: meta-llama/Meta-Llama-3-8B is a
# GATED repo -- accept the license on HF and export HF_TOKEN before prepare_env,
# or override BASE_MODEL with an open model (e.g. Qwen/Qwen2.5-1.5B-Instruct) for
# a quick smoke test.
export BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
export TUNING="${TUNING:-lora}"                        # lora | full

# Checkpoint / resume policy (applies to fine-tune & eval).
# At the paper's effective batch of 64 the NYC run is only ~52 optimizer
# steps/epoch (~414 total over 8 epochs), so checkpoint every 50 steps (~1/epoch)
# to keep the 6h-resume guarantee meaningful.
export SAVE_STEPS="${SAVE_STEPS:-50}"                  # checkpoint cadence (optimizer steps)
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"       # sliding window of 5 ckpts

# Fine-tune hyper-parameters -- aligned to the paper (KDD'25 §5.2 + §6.6).
# The paper trains on 4xL40 with per-GPU bs=2, grad_accum=8 (effective batch
# 2*8*4 = 64). On our single GPU we keep bs=2 and raise grad_accum to 32 so the
# effective batch stays 64, preserving the batch/LR relationship they tuned.
export NUM_EPOCHS="${NUM_EPOCHS:-8}"                   # §6.6: "train ... for 8 epochs"
export PER_DEVICE_BS="${PER_DEVICE_BS:-2}"             # §5.2: batch size 2 per GPU
export GRAD_ACCUM="${GRAD_ACCUM:-32}"                  # 4 GPUs x accum 8 -> 32 on 1 GPU (eff batch 64)
export LR="${LR:-1e-5}"                                # §5.2: learning rate 1e-5
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"              # §5.2: sequence length 2048
export WARMUP_STEPS="${WARMUP_STEPS:-20}"              # §5.2: warm-up phase of 20 steps
export LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"  # §5.2: constant LR schedule
export LORA_R="${LORA_R:-16}"                          # §5.2: LoRA rank R=16
export LORA_ALPHA="${LORA_ALPHA:-32}"                  # alpha=2r (authors' reference code)
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"             # §5.2: LoRA dropout 0.1
export ADD_SID_TOKENS="${ADD_SID_TOKENS:-0}"           # 1 = add atomic <a_*>/<b_*>/... tokens

# Eval knobs.
export EVAL_KS="${EVAL_KS:-1 5 10}"                    # cut-offs for Acc@k / NDCG@k
export NUM_BEAMS="${NUM_BEAMS:-10}"                    # >= max(EVAL_KS)
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-12}"
# Generation holds EVAL_BATCH_SIZE x NUM_BEAMS sequences in the KV-cache at once.
# For an 8B model with 2048-token prompts, batch 4 x 10 beams peaks ~27GB (16GB
# weights + ~11GB KV), leaving headroom on the 48GB L40S. Lower to 2 if you hit OOM.
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

# --------------------- Semantic-ID build (TKY/CA) --------------------------
# NYC ships pre-baked llm_*.json (its SIDs are already embedded), so it never
# needs this. TKY/CA have no shipped JSON -- scripts/data.slurm builds it from
# raw LLM4POI check-ins via the V2 CRQVAE SID module (V1/code/build_dataset.py).
# Datasets that ship their JSON pre-baked (no SID build / no download needed):
export PREBAKED_DATASETS="${PREBAKED_DATASETS:-nyc}"
# Category text encoder (downloaded from HF on first use) + PCA target dim.
export CAT_MODEL="${CAT_MODEL:-all-MiniLM-L6-v2}"
export CAT_DIM="${CAT_DIM:-64}"
export KEEP_LAST_K="${KEEP_LAST_K:-5}"        # train: keep last K samples/user (V2 notebook)
export SID_EPOCHS="${SID_EPOCHS:-3000}"       # CRQVAE epochs (must be >=210 to save best ckpt)
export SID_NUM_EMB="${SID_NUM_EMB:-64 64 64}" # 3 codebooks of 64 -> <a_*><b_*><c_*>
export SID_E_DIM="${SID_E_DIM:-64}"
export SID_LAYERS="${SID_LAYERS:-512 256 128}"
export SID_DEVICE="${SID_DEVICE:-cuda:0}"

# Raw-data download. Default source is the public HF dataset w11wo/LLM4POI, which
# ships per city {nyc,tky,ca}/preprocessed/{train_sample.csv,test_qa_pairs_kqt.txt}
# -- the exact LLM4POI preprocessed data, fetched with the HF_TOKEN already wired
# in for the base model (the repo is public, so the token is optional here).
# Override per dataset (suffix in CAPS, e.g. RAW_URL_TKY) to use another source:
#   RAW_GDRIVE_ID_<DS> Google-Drive file/folder id (fetched with gdown)
#   RAW_URL_<DS>       direct http(s) link to a .zip/.tar.gz, unpacked into RAW_DIR
# A configured RAW_GDRIVE_ID / RAW_URL takes precedence over HF_DATA_REPO.
export HF_DATA_REPO="${HF_DATA_REPO:-w11wo/LLM4POI}"
export RAW_URL="${RAW_URL:-}"
export RAW_GDRIVE_ID="${RAW_GDRIVE_ID:-}"

# --------------------------- derived paths ---------------------------------
# RUN_NAME keeps runs of different dataset/model separated so markers &
# checkpoints never collide.
_MODEL_TAG="$(basename "${BASE_MODEL}")"
export RUN_NAME="${RUN_NAME:-${DATASET}_${_MODEL_TAG}_${TUNING}}"

export DATA_DIR="${PROJECT_ROOT}/V1/datasets/${DATASET}"
export RAW_DIR="${RAW_DIR:-${DATA_DIR}/raw}"   # raw LLM4POI check-in CSVs (download target)
export CODE_DIR="${PROJECT_ROOT}/V1/code"
export RUN_DIR="${PROJECT_ROOT}/V1/runs/${RUN_NAME}"
export SID_DIR="${RUN_DIR}/sid"          # RQ-VAE checkpoints (optional ID-gen)
export SFT_DIR="${RUN_DIR}/sft"          # HF Trainer output dir (checkpoint-*)
export EVAL_DIR="${RUN_DIR}/eval"        # predictions.jsonl + metrics.json
export MARKER_DIR="${RUN_DIR}/markers"   # phase completion sentinels

# Project-local conda env + HF cache so we never touch the home quota and the
# whole thing stays self-contained / reproducible.
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${PROJECT_ROOT}/.conda/gnpr-sid}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
# Pin to the single GPU Slurm allocated us.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --------------------------- cluster modules -------------------------------
# Load anaconda + cuda from the module system instead of building toolchains.
# Names come from `module avail` (see modules.txt).
load_modules() {
    # `module` returns non-zero on harmless warnings; don't let errexit trip.
    set +e
    if command -v module >/dev/null 2>&1; then
        module load shared            >/dev/null 2>&1
        module load anaconda/25.5.1   2>/dev/null || module load miniconda3/25.5.1 2>/dev/null
        module load cuda/12.8.0       2>/dev/null
        echo "[env] modules loaded:"; module list 2>&1 | sed 's/^/[env]   /'
    else
        echo "[env] 'module' command not found; assuming conda/cuda already on PATH."
    fi
    set -e
}

# --------------------------- conda activation ------------------------------
activate_conda() {
    set +e
    local base
    base="$(conda info --base 2>/dev/null)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${base}/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV_PREFIX}"
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "[env] ERROR: could not activate ${CONDA_ENV_PREFIX}." >&2
        echo "[env]        Run  sbatch scripts/prepare_env.slurm  first." >&2
        return 1
    fi
    echo "[env] python: $(command -v python)"
    python - <<'PY' 2>/dev/null || true
import torch
print(f"[env] torch {torch.__version__} | cuda avail: {torch.cuda.is_available()} | "
      f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY
}

# --------------------------- phase markers ---------------------------------
# A "phase" is done once  ${MARKER_DIR}/<name>.done  exists.
phase_done()  { [[ -f "${MARKER_DIR}/$1.done" ]]; }
mark_done()   { mkdir -p "${MARKER_DIR}"; date -u +"%Y-%m-%dT%H:%M:%SZ" > "${MARKER_DIR}/$1.done"; \
                echo "[env] phase '$1' marked complete -> ${MARKER_DIR}/$1.done"; }

make_run_dirs() { mkdir -p "${RUN_DIR}" "${SID_DIR}" "${SFT_DIR}" "${EVAL_DIR}" "${MARKER_DIR}" "${PROJECT_ROOT}/log"; }

# --------------------------- data presence / fetch -------------------------
# A dataset is "prebaked" when it ships its llm_*.json (NYC) -> no build needed.
is_prebaked() {
    local ds=" ${PREBAKED_DATASETS} "
    [[ "${ds}" == *" ${DATASET} "* ]]
}
# Final LLM data is ready when all three llm_*.json exist.
data_ready() {
    [[ -f "${DATA_DIR}/llm_train.json" && -f "${DATA_DIR}/llm_val.json" \
        && -f "${DATA_DIR}/llm_test.json" ]]
}
# Raw check-ins are present when at least sample.csv (or a per-split CSV) exists.
raw_ready() {
    [[ -f "${RAW_DIR}/sample.csv" || -f "${RAW_DIR}/train_sample.csv" ]]
}

# Resolve a per-dataset override (e.g. RAW_URL_TKY) else the generic fallback.
_ds_var() {  # _ds_var VARPREFIX -> echoes value of <PREFIX>_<DS-UPPER> or $<PREFIX>
    local prefix="$1" up; up="$(echo "${DATASET}" | tr '[:lower:]' '[:upper:]')"
    local specific="${prefix}_${up}"
    echo "${!specific:-${!prefix:-}}"
}

# Download + unpack the raw dataset into RAW_DIR. Idempotent:
#   * prebaked dataset (nyc)            -> nothing to do
#   * llm_*.json already present        -> nothing to do
#   * raw csv already present           -> nothing to do
#   * a source (RAW_URL*/RAW_GDRIVE_ID*) is configured -> fetch it
#   * otherwise                         -> print instructions and return 1
fetch_dataset() {
    if is_prebaked;  then echo "[data] ${DATASET} is prebaked; no download needed."; return 0; fi
    if data_ready;   then echo "[data] ${DATASET}: llm_*.json already present; skip download."; return 0; fi
    if raw_ready;    then echo "[data] ${DATASET}: raw CSVs already in ${RAW_DIR}; skip download."; return 0; fi

    mkdir -p "${RAW_DIR}"
    local url gid; url="$(_ds_var RAW_URL)"; gid="$(_ds_var RAW_GDRIVE_ID)"
    local archive="${RAW_DIR}/_download"

    if [[ -n "${gid}" ]]; then
        echo "[data] ${DATASET}: fetching from Google Drive id=${gid} via gdown"
        if echo "${gid}" | grep -qiE 'folder'; then
            python -m gdown --folder "https://drive.google.com/drive/folders/${gid}" -O "${RAW_DIR}"
        else
            python -m gdown "${gid}" -O "${archive}"
        fi
    elif [[ -n "${url}" ]]; then
        echo "[data] ${DATASET}: downloading ${url}"
        if command -v wget >/dev/null 2>&1; then wget -q -O "${archive}" "${url}"
        else curl -fsSL -o "${archive}" "${url}"; fi
    elif [[ -n "${HF_DATA_REPO}" ]]; then
        echo "[data] ${DATASET}: fetching ${HF_DATA_REPO} (${DATASET}/preprocessed/*) from HF"
        python - "${HF_DATA_REPO}" "${DATASET}" "${RAW_DIR}" <<'PY'
import os, sys, shutil
from huggingface_hub import hf_hub_download
repo, ds, raw = sys.argv[1], sys.argv[2], sys.argv[3]
token = os.environ.get("HF_TOKEN")
os.makedirs(raw, exist_ok=True)
wants = {f"{ds}/preprocessed/train_sample.csv": "train_sample.csv",
         f"{ds}/preprocessed/test_qa_pairs_kqt.txt": "test_qa_pairs_kqt.txt"}
for remote, local in wants.items():
    try:
        p = hf_hub_download(repo, remote, repo_type="dataset", token=token)
        shutil.copyfile(p, os.path.join(raw, local))
        print(f"[data]   {remote} -> {local}")
    except Exception as e:
        # train_sample.csv is required; the test QA file is best-effort.
        lvl = "ERROR" if local == "train_sample.csv" else "WARN"
        print(f"[data]   {lvl}: could not fetch {remote}: {e}", file=sys.stderr)
        if local == "train_sample.csv":
            sys.exit(1)
PY
    else
        echo "[data] ERROR: no raw data for '${DATASET}' and no source configured." >&2
        echo "[data]   Default source HF_DATA_REPO is empty. Either set it back to" >&2
        echo "[data]   w11wo/LLM4POI, drop train_sample.csv (+ test_qa_pairs_kqt.txt)" >&2
        echo "[data]   into ${RAW_DIR}/, or set RAW_URL_${DATASET^^} / RAW_GDRIVE_ID_${DATASET^^}." >&2
        return 1
    fi

    # unpack if we downloaded a single archive
    if [[ -f "${archive}" ]]; then
        echo "[data] unpacking archive into ${RAW_DIR}"
        case "$(file -b "${archive}" 2>/dev/null || echo)" in
            *Zip*)   ( cd "${RAW_DIR}" && unzip -o "${archive}" ) ;;
            *gzip*)  tar -xzf "${archive}" -C "${RAW_DIR}" ;;
            *tar*)   tar -xf  "${archive}" -C "${RAW_DIR}" ;;
            *)       ( cd "${RAW_DIR}" && (unzip -o "${archive}" || tar -xf "${archive}") ) || \
                     echo "[data] WARN: could not auto-unpack ${archive}; unpack it manually." ;;
        esac
        rm -f "${archive}"
    fi

    if raw_ready; then echo "[data] ${DATASET}: raw data ready in ${RAW_DIR}"; return 0; fi
    echo "[data] WARN: download finished but no sample.csv/train_sample.csv found in ${RAW_DIR}." >&2
    echo "[data]   Check the archive layout; the build expects them at ${RAW_DIR}/." >&2
    return 1
}

print_config() {
    cat <<CFG
[env] ===================== GNPR-SID config =====================
[env] PROJECT_ROOT : ${PROJECT_ROOT}
[env] DATASET      : ${DATASET}        DATA_DIR: ${DATA_DIR}
[env] BASE_MODEL   : ${BASE_MODEL}     TUNING: ${TUNING}
[env] RUN_DIR      : ${RUN_DIR}
[env] ckpt policy  : save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}
[env] sft hp       : epochs=${NUM_EPOCHS} bs=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM} (eff batch $((PER_DEVICE_BS*GRAD_ACCUM))) lr=${LR} max_seq_len=${MAX_SEQ_LEN}
[env] sft schedule : lr_scheduler=${LR_SCHEDULER} warmup_steps=${WARMUP_STEPS}
[env] lora         : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} add_sid_tokens=${ADD_SID_TOKENS}
[env] eval         : Ks=[${EVAL_KS}] num_beams=${NUM_BEAMS}
[env] data         : prebaked=$(is_prebaked && echo yes || echo no) llm_json=$(data_ready && echo present || echo missing) raw=$(raw_ready && echo present || echo missing)
[env] sid build    : epochs=${SID_EPOCHS} num_emb=[${SID_NUM_EMB}] cat_model=${CAT_MODEL} RAW_DIR=${RAW_DIR}
[env] conda env    : ${CONDA_ENV_PREFIX}
[env] HF_HOME      : ${HF_HOME}
[env] ============================================================
CFG
}
