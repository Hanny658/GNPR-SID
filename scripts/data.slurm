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
#SBATCH --job-name=data_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Semantic-ID data build (for datasets WITHOUT pre-baked JSON, i.e. TKY / CA).
# Turns the raw LLM4POI check-ins into llm_{train,val,test}.json via the V2
# CRQVAE SID module (V1/code/build_dataset.py). Idempotent / resumable: each
# stage skips when its output already exists, and the job is a no-op once the
# final JSON is present. NYC is prebaked -> this exits immediately.
# ---------------------------------------------------------------------------
if   [[ -f scripts/env.sh ]]; then source scripts/env.sh
elif [[ -f env.sh         ]]; then source env.sh
else echo "[data] ERROR: cannot find scripts/env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_run_dirs
print_config

if phase_done data || data_ready; then
    echo "[data] llm_*.json already present for ${DATASET}; nothing to build."
    mark_done data
    exit 0
fi
if is_prebaked; then
    echo "[data] ERROR: ${DATASET} is marked prebaked but its llm_*.json is missing." >&2
    echo "[data]        Restore ${DATA_DIR}/llm_*.json or remove ${DATASET} from PREBAKED_DATASETS." >&2
    exit 1
fi

# Make sure the raw check-ins are here (downloads if a source is configured).
fetch_dataset || { echo "[data] ERROR: raw data unavailable; cannot build." >&2; exit 1; }

# Self-heal: the SID build needs sentence-transformers (stage 2). If the env was
# built before that dep was added to requirements.txt, top it up here instead of
# forcing a full env rebuild. Cheap when already satisfied (just an import check).
if ! python -c "import sentence_transformers" 2>/dev/null; then
    echo "[data] build deps missing; syncing requirements.txt into the env"
    python -m pip install -r "${PROJECT_ROOT}/scripts/requirements.txt"
fi

echo "[data] building Semantic-ID dataset for ${DATASET} (V2 CRQVAE)"
python "${CODE_DIR}/build_dataset.py" \
    --dataset "${DATASET}" \
    --raw_dir "${RAW_DIR}" \
    --out_dir "${DATA_DIR}" \
    --ckpt_dir "${SID_DIR}" \
    --device "${SID_DEVICE}" \
    --cat_model "${CAT_MODEL}" \
    --cat_dim "${CAT_DIM}" \
    --keep_last_k "${KEEP_LAST_K}" \
    --epochs "${SID_EPOCHS}" \
    --num_emb_list ${SID_NUM_EMB} \
    --e_dim "${SID_E_DIM}" \
    --layers ${SID_LAYERS}

# data_ready requires train+val+test; the HF layout has no val, so accept train+test too.
if ! data_ready && [[ ! ( -f "${DATA_DIR}/llm_train.json" && -f "${DATA_DIR}/llm_test.json" ) ]]; then
    echo "[data] ERROR: build finished but llm_train/test.json still missing." >&2; exit 1
fi
mark_done data
echo "[data] complete -> ${DATA_DIR}/llm_train.json + llm_test.json (val skipped if no source)"
echo "[data] next: DATASET=${DATASET} sbatch scripts/train.slurm"
