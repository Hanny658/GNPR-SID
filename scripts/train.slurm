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
#SBATCH --job-name=train_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Phase 1 (optional): Semantic-ID generation -- RQ-VAE training + codebook.
# Phase 2          : LLM LoRA fine-tuning on the SID check-in sequences.
#
# Resumable: each phase writes a done-marker; re-submitting the identical
# script skips finished phases and resumes an interrupted fine-tune from its
# newest 200-step checkpoint.  If everything is done the job exits immediately.
# ---------------------------------------------------------------------------
if   [[ -f scripts/env.sh ]]; then source scripts/env.sh
elif [[ -f env.sh         ]]; then source env.sh
else echo "[train] ERROR: cannot find scripts/env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_run_dirs
print_config

# Fast exit if the whole training pipeline already finished.
if phase_done sft; then
    echo "[train] fine-tune already complete (markers/sft.done present); nothing to do."
    exit 0
fi

# ============================ Phase 1: ID generation ========================
# RQ-VAE -> Semantic IDs.  This needs V1/datasets/$DATASET/poi_info.csv, which
# is NOT shipped with the bundled NYC sample (its SIDs are already baked into
# llm_{train,val,test}.json).  So by default we skip and mark the phase done.
# Set RUN_IDGEN=1 (and provide poi_info.csv) to actually run RQ-VAE.
POI_INFO="${DATA_DIR}/poi_info.csv"
if phase_done idgen; then
    echo "[train] ID-gen already handled; skipping."
elif [[ "${RUN_IDGEN:-0}" == "1" && -f "${POI_INFO}" ]]; then
    echo "[train] Phase 1: training RQ-VAE on ${POI_INFO}"
    # train_rqvae.py / POIdataset.py expect cwd=V1 and a /datasets/... data_path;
    # imports live in V1/code, so put it on PYTHONPATH.
    ( cd "${PROJECT_ROOT}/V1" && PYTHONPATH="${CODE_DIR}" python "${CODE_DIR}/train_rqvae.py" \
        --device cuda:0 \
        --data_path "/datasets/${DATASET}/poi_info.csv" \
        --ckpt_dir "${SID_DIR}" \
        --save_limit "${SAVE_TOTAL_LIMIT}" \
        --eval_step "${RQVAE_EVAL_STEP:-50}" \
        --epochs "${RQVAE_EPOCHS:-2000}" )
    echo "[train] Phase 1: generating codebook / Semantic IDs"
    echo "[train] NOTE: V1/code/codebook.py hard-codes data_mode='TKY' and writes"
    echo "[train]       columns [Pid,Codebook,Vector] while llm_dataprocess.ipynb"
    echo "[train]       expects [pid,sid] -- adjust those for your dataset before"
    echo "[train]       regenerating llm_*.json. See scripts/README.md."
    ( cd "${PROJECT_ROOT}/V1" && PYTHONPATH="${CODE_DIR}" python "${CODE_DIR}/codebook.py" \
        --device cuda:0 \
        --data_path "/datasets/${DATASET}/poi_info.csv" \
        --ckpt_dir "${SID_DIR}" \
        --save_limit "${SAVE_TOTAL_LIMIT}" ) || \
        echo "[train] WARN: codebook.py failed (see note above); continuing."
    mark_done idgen
else
    echo "[train] Phase 1 skipped: using pre-baked Semantic IDs in ${DATA_DIR}/llm_*.json"
    echo "[train]   (set RUN_IDGEN=1 and provide ${POI_INFO} to run RQ-VAE instead)"
    mark_done idgen
fi

# ============================ Phase 2: LLM fine-tune ========================
TRAIN_JSON="${DATA_DIR}/llm_train.json"
VAL_JSON="${DATA_DIR}/llm_val.json"
[[ -f "${TRAIN_JSON}" ]] || { echo "[train] ERROR: missing ${TRAIN_JSON}" >&2; exit 1; }

echo "[train] Phase 2: LoRA fine-tuning ${BASE_MODEL}"
FT_ARGS=(
    --base_model "${BASE_MODEL}"
    --train_file "${TRAIN_JSON}"
    --output_dir "${SFT_DIR}"
    --tuning "${TUNING}"
    --add_sid_tokens "${ADD_SID_TOKENS}"
    --num_epochs "${NUM_EPOCHS}"
    --per_device_bs "${PER_DEVICE_BS}"
    --grad_accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
)
if [[ -f "${VAL_JSON}" ]]; then
    FT_ARGS+=( --val_file "${VAL_JSON}" )
fi
python "${CODE_DIR}/finetune_llm.py" "${FT_ARGS[@]}"

mark_done sft
echo "[train] training pipeline complete -> ${SFT_DIR}/final"
echo "[train] next: sbatch scripts/eval.slurm"
