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
#SBATCH --job-name=uaware_train_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# User-aware pipeline, step 2: train with the V2 recipe + a user profile.
#
#   Phase A (V2_ALIGN=1, default): SID-LLM alignment -- add the SID tokens and
#           train ONLY the embeddings (finetune_llm.py --tuning embed) on the
#           alignment data. With ALIGN_USER_PAIRS=1 (the B2 arm) that data also
#           carries profile->SID pairs, so user-conditioned SID priors are
#           grounded too. Produces a FULL SID-aware model -- the SFT base.
#   Phase B: SFT on the (profile-injected) llm_train data. finetune_llm renders
#           the record's `profile` field in a protected "### User Profile:" block.
#
# Arm is chosen by PROFILE_MODE (none|raw|sid); each arm has its own RUN_DIR.
# Resumable: phase markers + auto-resume from the newest checkpoint.
# ---------------------------------------------------------------------------
if   [[ -f scripts/uaware_env.sh ]]; then source scripts/uaware_env.sh
elif [[ -f uaware_env.sh         ]]; then source uaware_env.sh
else echo "[uaware-train] ERROR: cannot find scripts/uaware_env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_uaware_run_dirs
print_config
print_uaware_config

if phase_done uaware_sft; then
    echo "[uaware-train] training already complete (markers/uaware_sft.done); nothing to do."
    exit 0
fi

TRAIN_JSON="${UAWARE_TRAIN_JSON}"
if [[ ! -f "${TRAIN_JSON}" ]]; then
    echo "[uaware-train] ERROR: missing ${TRAIN_JSON} (PROFILE_MODE=${PROFILE_MODE})." >&2
    echo "[uaware-train]        Build it first:  DATASET=${DATASET} sbatch scripts/uaware_data.slurm" >&2
    exit 1
fi

SFT_BASE_MODEL="${BASE_MODEL}"
SFT_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj"   # sft_without_alignment.py
SFT_ADD_SID="${ADD_SID_TOKENS}"

if [[ "${V2_ALIGN}" == "1" ]]; then
    SFT_ADD_SID=1
    # ============ Phase A: SID-LLM alignment (train token embeddings) =======
    if phase_done uaware_align; then
        echo "[uaware-train] Phase A (alignment) already done; skipping."
    else
        if [[ ! -f "${ALIGN_TRAIN_JSON}" ]]; then
            echo "[uaware-train] ERROR: alignment data missing (${ALIGN_TRAIN_JSON})." >&2
            echo "[uaware-train]        Build it:  DATASET=${DATASET} sbatch scripts/uaware_data.slurm" >&2
            echo "[uaware-train]        or skip alignment: V2_ALIGN=0 sbatch scripts/uaware_train.slurm" >&2
            exit 1
        fi
        echo "[uaware-train] Phase A: SID alignment (embed-only, user_pairs=${ALIGN_USER_PAIRS}) of ${BASE_MODEL}"
        python "${CODE_DIR}/finetune_llm.py" \
            --base_model "${BASE_MODEL}" \
            --train_file "${ALIGN_TRAIN_JSON}" \
            --val_file "${ALIGN_VAL_JSON}" \
            --output_dir "${ALIGN_DIR}" \
            --tuning embed \
            --add_sid_tokens 1 \
            --num_epochs "${V2_ALIGN_EPOCHS}" \
            --per_device_bs "${V2_ALIGN_BS}" \
            --grad_accum "${V2_ALIGN_ACCUM}" \
            --lr "${V2_LR}" \
            --lr_scheduler "${V2_LR_SCHEDULER}" \
            --warmup_steps "${V2_ALIGN_WARMUP}" \
            --max_seq_len "${V2_ALIGN_MAX_SEQ_LEN}" \
            --save_steps "${SAVE_STEPS}" \
            --save_total_limit 1
        mark_done uaware_align
    fi
    SFT_BASE_MODEL="${ALIGN_DIR}/final"
    SFT_TARGETS="q_proj,k_proj,v_proj,gate_proj,up_proj"      # sft_after_alignment.py
fi

# ===================== Phase B: SFT (profile-injected) ======================
echo "[uaware-train] Phase B: SFT of ${SFT_BASE_MODEL} on ${TRAIN_JSON} (targets: ${SFT_TARGETS}, add_sid=${SFT_ADD_SID})"
FT_ARGS=(
    --base_model "${SFT_BASE_MODEL}"
    --train_file "${TRAIN_JSON}"
    --output_dir "${SFT_DIR}"
    --tuning lora
    --add_sid_tokens "${SFT_ADD_SID}"
    --lora_targets "${SFT_TARGETS}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${V2_LORA_DROPOUT}"
    --num_epochs "${V2_SFT_EPOCHS}"
    --per_device_bs "${V2_SFT_BS}"
    --grad_accum "${V2_SFT_ACCUM}"
    --lr "${V2_LR}"
    --lr_scheduler "${V2_LR_SCHEDULER}"
    --warmup_steps "${V2_SFT_WARMUP}"
    --max_seq_len "${V2_SFT_MAX_SEQ_LEN}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
)
if [[ -f "${DATA_DIR}/llm_val.json" ]]; then
    FT_ARGS+=( --val_file "${DATA_DIR}/llm_val.json" )
fi
python "${CODE_DIR}/finetune_llm.py" "${FT_ARGS[@]}"

mark_done uaware_sft
echo "[uaware-train] training complete -> ${SFT_DIR}/final"
echo "[uaware-train] next: PROFILE_MODE=${PROFILE_MODE} sbatch scripts/uaware_eval.slurm"
