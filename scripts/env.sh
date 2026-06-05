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
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export TUNING="${TUNING:-lora}"                        # lora | full

# Checkpoint / resume policy (applies to fine-tune & eval).
export SAVE_STEPS="${SAVE_STEPS:-200}"                 # checkpoint every ~200 steps
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"       # sliding window of 5 ckpts

# Fine-tune hyper-parameters (single-GPU friendly defaults).
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export LR="${LR:-2e-5}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
export ADD_SID_TOKENS="${ADD_SID_TOKENS:-0}"           # 1 = add atomic <a_*>/<b_*>/... tokens

# Eval knobs.
export EVAL_KS="${EVAL_KS:-1 5 10}"                    # cut-offs for Acc@k / NDCG@k
export NUM_BEAMS="${NUM_BEAMS:-10}"                    # >= max(EVAL_KS)
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-12}"

# --------------------------- derived paths ---------------------------------
# RUN_NAME keeps runs of different dataset/model separated so markers &
# checkpoints never collide.
_MODEL_TAG="$(basename "${BASE_MODEL}")"
export RUN_NAME="${RUN_NAME:-${DATASET}_${_MODEL_TAG}_${TUNING}}"

export DATA_DIR="${PROJECT_ROOT}/V1/datasets/${DATASET}"
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

print_config() {
    cat <<CFG
[env] ===================== GNPR-SID config =====================
[env] PROJECT_ROOT : ${PROJECT_ROOT}
[env] DATASET      : ${DATASET}        DATA_DIR: ${DATA_DIR}
[env] BASE_MODEL   : ${BASE_MODEL}     TUNING: ${TUNING}
[env] RUN_DIR      : ${RUN_DIR}
[env] ckpt policy  : save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}
[env] sft hp       : epochs=${NUM_EPOCHS} bs=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM} lr=${LR} max_seq_len=${MAX_SEQ_LEN}
[env] eval         : Ks=[${EVAL_KS}] num_beams=${NUM_BEAMS}
[env] conda env    : ${CONDA_ENV_PREFIX}
[env] HF_HOME      : ${HF_HOME}
[env] ============================================================
CFG
}
