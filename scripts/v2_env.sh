#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/v2_env.sh
#
# Shared environment for the *fully-V2* pipeline (v2_data / v2_train /
# v2_eval .slurm). Sources scripts/env.sh first (same conda env, same data
# fetch, same helpers), then:
#   * adds the V2 training knobs, defaulting to the authors' values in
#     V2/LLM/train/{align_sid,sft_after_alignment,sft_without_alignment}.py,
#   * moves the run layout to V2/runs/<dataset>_<model>_v2 so V2 runs and
#     their phase markers never collide with the V1-recipe runs.
#
# The V2 flow: [align (LoRA on embed_tokens) -> merge ->] SFT -> eval.
# Alignment is optional (V2/V2.md: "or directly fine-tuning without
# alignment"); toggle with V2_ALIGN.
# ---------------------------------------------------------------------------

_V2_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_V2_ENV_DIR}/env.sh"

# ------------------------- V2 recipe knobs ---------------------------------
export V2_ALIGN="${V2_ALIGN:-1}"                 # 1 = align_sid + merge before SFT
export V2_LR="${V2_LR:-2e-5}"                    # authors' learning_rate (both stages)
export V2_LR_SCHEDULER="${V2_LR_SCHEDULER:-linear}"  # authors use the HF default

# SFT stage (sft_after_alignment.py / sft_without_alignment.py __main__ values)
export V2_SFT_EPOCHS="${V2_SFT_EPOCHS:-5}"
export V2_SFT_BS="${V2_SFT_BS:-8}"
export V2_SFT_ACCUM="${V2_SFT_ACCUM:-2}"
export V2_SFT_MAX_SEQ_LEN="${V2_SFT_MAX_SEQ_LEN:-3072}"
export V2_SFT_WARMUP="${V2_SFT_WARMUP:-100}"

# Alignment stage (align_sid.py __main__ values)
export V2_ALIGN_EPOCHS="${V2_ALIGN_EPOCHS:-6}"
export V2_ALIGN_BS="${V2_ALIGN_BS:-16}"
export V2_ALIGN_ACCUM="${V2_ALIGN_ACCUM:-2}"
export V2_ALIGN_MAX_SEQ_LEN="${V2_ALIGN_MAX_SEQ_LEN:-1024}"
export V2_ALIGN_WARMUP="${V2_ALIGN_WARMUP:-180}"

export V2_LORA_DROPOUT="${V2_LORA_DROPOUT:-0.05}"   # V2 scripts use 0.05 (V1: 0.1)

# ------------------------- V2 run layout -----------------------------------
# Re-point the run dirs (and the phase-marker dir the env.sh helpers use) at
# V2/runs/. _MODEL_TAG comes from env.sh.
export V2_RUN_NAME="${V2_RUN_NAME:-${DATASET}_${_MODEL_TAG}_v2}"
export RUN_DIR="${PROJECT_ROOT}/V2/runs/${V2_RUN_NAME}"
export SID_DIR="${RUN_DIR}/sid"
export SFT_DIR="${RUN_DIR}/sft"
export EVAL_DIR="${RUN_DIR}/eval"
export MARKER_DIR="${RUN_DIR}/markers"
export ALIGN_DIR="${RUN_DIR}/align"              # alignment LoRA checkpoints
export MERGED_DIR="${RUN_DIR}/align_merged"      # base + alignment LoRA, merged

# Alignment data lives next to the other dataset artifacts.
export ALIGN_TRAIN_JSON="${DATA_DIR}/llm_align_train.json"
export ALIGN_VAL_JSON="${DATA_DIR}/llm_align_val.json"

make_v2_run_dirs() { make_run_dirs; mkdir -p "${ALIGN_DIR}"; }

print_v2_config() {
    cat <<CFG
[env] ===================== V2 pipeline config ===================
[env] V2 RUN_DIR    : ${RUN_DIR}
[env] align stage   : enabled=${V2_ALIGN} epochs=${V2_ALIGN_EPOCHS} bs=${V2_ALIGN_BS}x${V2_ALIGN_ACCUM} len=${V2_ALIGN_MAX_SEQ_LEN} warmup=${V2_ALIGN_WARMUP}
[env] sft stage     : epochs=${V2_SFT_EPOCHS} bs=${V2_SFT_BS}x${V2_SFT_ACCUM} len=${V2_SFT_MAX_SEQ_LEN} warmup=${V2_SFT_WARMUP}
[env] lr            : ${V2_LR} (${V2_LR_SCHEDULER})   lora_dropout=${V2_LORA_DROPOUT}
[env] align data    : $( [[ -f "${ALIGN_TRAIN_JSON}" ]] && echo present || echo missing ) (${ALIGN_TRAIN_JSON})
[env] ============================================================
CFG
}
