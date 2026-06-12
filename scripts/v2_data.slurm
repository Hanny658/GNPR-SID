#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=48G
#
### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=8
#
### Specify number of node to compute ###
#SBATCH --nodes=1
#
### Optional: Specify node to execute the job ###
#SBATCH --nodelist=TC2N07
#
### Specify Time Limit ###
#SBATCH --time=06:00:00
#
#SBATCH --job-name=v2_data_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Fully-V2 pipeline, step 1: dataset build.
#   a) Semantic-ID llm_*.json  -- delegates to the idempotent scripts/data.slurm
#      (raw fetch -> V2 CRQVAE SID build); a no-op when already built.
#   b) SID<->attribute ALIGNMENT data (llm_align_{train,val}.json), ported from
#      V2/dataprocess/get_align_data.ipynb. Needs poi_info.csv + codebook.csv,
#      which step (a) leaves in DATA_DIR.
# NOTE for nyc: it is prebaked, so step (a) skips and produces no poi_info/
# codebook -> alignment data cannot be built. Either run the V2 LLM stage with
# V2_ALIGN=0, or force a full from-raw V2 rebuild with PREBAKED_DATASETS="".
# Idempotent: every step skips when its outputs exist.
# ---------------------------------------------------------------------------
if   [[ -f scripts/v2_env.sh ]]; then source scripts/v2_env.sh
elif [[ -f v2_env.sh         ]]; then source v2_env.sh
else echo "[v2-data] ERROR: cannot find scripts/v2_env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_v2_run_dirs
print_config
print_v2_config

# ---- (a) Semantic-ID dataset (llm_train/test.json + poi_info + codebook) ----
if [[ -f "${DATA_DIR}/llm_train.json" && -f "${DATA_DIR}/llm_test.json" ]]; then
    echo "[v2-data] llm_*.json already present for ${DATASET}; SID build skipped."
else
    echo "[v2-data] building Semantic-ID dataset via scripts/data.slurm"
    bash "${PROJECT_ROOT}/scripts/data.slurm"
fi

# ---- (b) alignment data ----------------------------------------------------
POI_INFO="${DATA_DIR}/poi_info.csv"
CODEBOOK="${DATA_DIR}/codebook.csv"
if [[ -f "${ALIGN_TRAIN_JSON}" && -f "${ALIGN_VAL_JSON}" ]]; then
    echo "[v2-data] alignment data already present; skipping."
elif [[ -f "${POI_INFO}" && -f "${CODEBOOK}" ]]; then
    echo "[v2-data] building SID<->attribute alignment data"
    python "${CODE_DIR}/build_align_data.py" \
        --poi_info "${POI_INFO}" \
        --codebook "${CODEBOOK}" \
        --out_train "${ALIGN_TRAIN_JSON}" \
        --out_val "${ALIGN_VAL_JSON}"
else
    echo "[v2-data] WARN: ${POI_INFO} / ${CODEBOOK} not found -- cannot build alignment data."
    echo "[v2-data]       (${DATASET} is likely prebaked, so the SID build was skipped.)"
    echo "[v2-data]       Options: V2_ALIGN=0 sbatch scripts/v2_train.slurm   (skip alignment)"
    echo "[v2-data]                PREBAKED_DATASETS=\"\" sbatch scripts/v2_data.slurm  (rebuild SIDs from raw)"
fi

echo "[v2-data] done. next: DATASET=${DATASET} sbatch scripts/v2_train.slurm"
