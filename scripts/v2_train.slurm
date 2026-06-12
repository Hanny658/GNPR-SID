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
#SBATCH --job-name=v2_train_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Fully-V2 pipeline, step 2: LLM training with the authors' V2 recipe
# (V2/LLM/train/*), driven by our resumable finetune_llm.py because the V2
# scripts as shipped have blanked-out paths, hardcoded wandb logging, Llama-3
# tokens baked into the prompt, and no checkpoint resume.
#
#   Phase A (V2_ALIGN=1, default): SID-LLM semantic alignment -- LoRA on
#           embed_tokens ONLY, trained on llm_align_*.json  (align_sid.py).
#   Phase B: merge the alignment LoRA into the base -> align_merged/
#           (merge_model.py equivalent).
#   Phase C: SFT on llm_train.json on top of the (merged) base
#           (sft_after_alignment.py: LoRA q,k,v,gate,up; without alignment:
#            sft_without_alignment.py: LoRA q,k,v,o,gate,up).
#
# Resumable: each phase writes a marker under V2/runs/<run>/markers/, and the
# fine-tunes auto-resume from their newest checkpoint on re-submission.
# ---------------------------------------------------------------------------
if   [[ -f scripts/v2_env.sh ]]; then source scripts/v2_env.sh
elif [[ -f v2_env.sh         ]]; then source v2_env.sh
else echo "[v2-train] ERROR: cannot find scripts/v2_env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_v2_run_dirs
print_config
print_v2_config

if phase_done v2_sft; then
    echo "[v2-train] V2 training already complete (markers/v2_sft.done); nothing to do."
    exit 0
fi

TRAIN_JSON="${DATA_DIR}/llm_train.json"
VAL_JSON="${DATA_DIR}/llm_val.json"
if [[ ! -f "${TRAIN_JSON}" ]]; then
    echo "[v2-train] ERROR: missing ${TRAIN_JSON}." >&2
    echo "[v2-train]        Build it first:  DATASET=${DATASET} sbatch scripts/v2_data.slurm" >&2
    exit 1
fi

SFT_BASE_MODEL="${BASE_MODEL}"
SFT_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj"   # sft_without_alignment.py

if [[ "${V2_ALIGN}" == "1" ]]; then
    # ================= Phase A: SID-LLM alignment (embed_tokens LoRA) =======
    if phase_done v2_align; then
        echo "[v2-train] Phase A (alignment) already done; skipping."
    else
        if [[ ! -f "${ALIGN_TRAIN_JSON}" ]]; then
            echo "[v2-train] ERROR: alignment data missing (${ALIGN_TRAIN_JSON})." >&2
            echo "[v2-train]        Either build it:  DATASET=${DATASET} sbatch scripts/v2_data.slurm" >&2
            echo "[v2-train]        or skip alignment: V2_ALIGN=0 sbatch scripts/v2_train.slurm" >&2
            exit 1
        fi
        echo "[v2-train] Phase A: SID alignment (LoRA on embed_tokens) of ${BASE_MODEL}"
        python "${CODE_DIR}/finetune_llm.py" \
            --base_model "${BASE_MODEL}" \
            --train_file "${ALIGN_TRAIN_JSON}" \
            --val_file "${ALIGN_VAL_JSON}" \
            --output_dir "${ALIGN_DIR}" \
            --tuning lora \
            --lora_targets "embed_tokens" \
            --lora_r "${LORA_R}" \
            --lora_alpha "${LORA_ALPHA}" \
            --lora_dropout "${V2_LORA_DROPOUT}" \
            --num_epochs "${V2_ALIGN_EPOCHS}" \
            --per_device_bs "${V2_ALIGN_BS}" \
            --grad_accum "${V2_ALIGN_ACCUM}" \
            --lr "${V2_LR}" \
            --lr_scheduler "${V2_LR_SCHEDULER}" \
            --warmup_steps "${V2_ALIGN_WARMUP}" \
            --max_seq_len "${V2_ALIGN_MAX_SEQ_LEN}" \
            --save_steps "${SAVE_STEPS}" \
            --save_total_limit "${SAVE_TOTAL_LIMIT}"
        mark_done v2_align
    fi

    # ================= Phase B: merge alignment LoRA into the base ==========
    if phase_done v2_merge; then
        echo "[v2-train] Phase B (merge) already done; skipping."
    else
        echo "[v2-train] Phase B: merging ${ALIGN_DIR}/final into ${BASE_MODEL}"
        python "${CODE_DIR}/merge_adapter.py" \
            --base_model "${BASE_MODEL}" \
            --adapter_dir "${ALIGN_DIR}/final" \
            --out_dir "${MERGED_DIR}"
        mark_done v2_merge
    fi

    SFT_BASE_MODEL="${MERGED_DIR}"
    SFT_TARGETS="q_proj,k_proj,v_proj,gate_proj,up_proj"      # sft_after_alignment.py
fi

# ===================== Phase C: SFT with the V2 recipe ======================
echo "[v2-train] Phase C: V2 SFT of ${SFT_BASE_MODEL} (targets: ${SFT_TARGETS})"
FT_ARGS=(
    --base_model "${SFT_BASE_MODEL}"
    --train_file "${TRAIN_JSON}"
    --output_dir "${SFT_DIR}"
    --tuning lora
    --add_sid_tokens "${ADD_SID_TOKENS}"
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
if [[ -f "${VAL_JSON}" ]]; then
    FT_ARGS+=( --val_file "${VAL_JSON}" )
fi
python "${CODE_DIR}/finetune_llm.py" "${FT_ARGS[@]}"

mark_done v2_sft
echo "[v2-train] V2 training complete -> ${SFT_DIR}/final"
echo "[v2-train] next: DATASET=${DATASET} sbatch scripts/v2_eval.slurm"
