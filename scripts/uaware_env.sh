#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/uaware_env.sh
#
# Shared environment for the *user-aware* (GenUP x GNPR-SID) pipeline:
# uaware_data / uaware_train / uaware_eval .slurm. It fuses GenUP's user
# profiles with GNPR-SID's Semantic-ID POIs by injecting a (SID-aware) profile
# into the recommender prompt and grounding user->SID priors in the alignment
# stage.
#
# Sources scripts/env.sh first (same conda env / data fetch / helpers), reuses
# the V2 training recipe (align embed -> SFT, no merge), and moves the run
# layout to V2/runs/<dataset>_<model>_uaware_<mode> so the three ablation arms
# never collide:
#
#   PROFILE_MODE=none  -> B0  POI-side only           (no profile, POI-only align)
#   PROFILE_MODE=raw   -> B1  GenUP raw profile        (POI-only align)
#   PROFILE_MODE=sid   -> B2  SID-aware profile (ours) (POI + profile<->SID align)
# ---------------------------------------------------------------------------

# Snapshot any explicit overrides before env.sh applies its own defaults.
# 4-atom SIDs need ~5 tokens incl. EOS, so this pipeline defaults eval to 32.
_UAW_EVAL_MNT="${EVAL_MAX_NEW_TOKENS:-}"
_UAW_STL="${SAVE_TOTAL_LIMIT:-}"

_UAW_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_UAW_ENV_DIR}/env.sh"

export EVAL_MAX_NEW_TOKENS="${_UAW_EVAL_MNT:-32}"
# SFT trains embed_tokens/lm_head (modules_to_save) so each checkpoint carries a
# ~2GB embedding matrix; keep a short sliding window and drop optimizer state so
# the SFT run stays small (~a few GB) instead of tens of GB.
export SAVE_TOTAL_LIMIT="${_UAW_STL:-2}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"

# ------------------------- V2 recipe knobs (mirror v2_env.sh) ---------------
# Default base model is meta-llama/Meta-Llama-3-8B (env.sh). The per-device
# micro-batches below are split smaller than v2_env's (which targeted a 1.5B
# model) so the 8B base fits a single 48G GPU, while the *effective* batch is
# preserved (SFT 2x8=16, align 4x8=32). Raise the per-device batch (and lower
# accum) for smaller models / more VRAM.
export V2_ALIGN="${V2_ALIGN:-1}"
export V2_LR="${V2_LR:-2e-5}"
export V2_LR_SCHEDULER="${V2_LR_SCHEDULER:-linear}"
export V2_SFT_EPOCHS="${V2_SFT_EPOCHS:-5}"
export V2_SFT_BS="${V2_SFT_BS:-2}"                       # 8B-safe (v2: 8); eff batch 2x8=16
export V2_SFT_ACCUM="${V2_SFT_ACCUM:-8}"
export V2_SFT_MAX_SEQ_LEN="${V2_SFT_MAX_SEQ_LEN:-3072}"   # room for the profile block
export V2_SFT_WARMUP="${V2_SFT_WARMUP:-100}"
export V2_ALIGN_EPOCHS="${V2_ALIGN_EPOCHS:-6}"
export V2_ALIGN_BS="${V2_ALIGN_BS:-4}"                   # 8B-safe (v2: 16); eff batch 4x8=32
export V2_ALIGN_ACCUM="${V2_ALIGN_ACCUM:-8}"
export V2_ALIGN_MAX_SEQ_LEN="${V2_ALIGN_MAX_SEQ_LEN:-1024}"
export V2_ALIGN_WARMUP="${V2_ALIGN_WARMUP:-180}"
export V2_LORA_DROPOUT="${V2_LORA_DROPOUT:-0.05}"

# The embed-alignment phase trains the FULL model (not LoRA), so each step
# checkpoint writes the whole ~16GB 8B model + optimizer state, and rotation
# transiently doubles that (~48GB) -- unlike the small LoRA adapters the SFT
# phase / GenUP / plain GNPR-SID write. The phase is short (~1-2h, < one 6h
# job), so by default we SKIP intermediate checkpoints and save the full model
# just ONCE at the end (trainer.save_model -> align/final). Lower this only if
# you need mid-align resume on a slow / frequently-preempted node.
export ALIGN_SAVE_STEPS="${ALIGN_SAVE_STEPS:-100000000}"

# ------------------------- user-aware knobs --------------------------------
export PROFILE_MODE="${PROFILE_MODE:-sid}"          # sid | raw | none
export GENUP_DIR="${GENUP_DIR:-${PROJECT_ROOT}/../GenUP}"
export GENUP_PROFILES_DIR="${GENUP_PROFILES_DIR:-${GENUP_DIR}/data/${DATASET}/user_profiles}"
export PROFILES_SID_JSON="${PROFILES_SID_JSON:-${DATA_DIR}/user_profiles_sid.json}"
export PROFILE_TOP_N="${PROFILE_TOP_N:-10}"

# GenUP method: the user's FULL history feeds only the (offline) profile, while
# the PROMPT carries just the most recent HISTORY_KEEP_LAST visits (the "current
# trajectory"). This is what makes the cold-start study meaningful -- a rich
# history can't be smuggled into the prompt. Set -1 to keep the full GNPR-SID
# history; 0 for profile-only. Sweep it (e.g. -1, 15, 8, 3, 0) to study how much
# the SID-aware profile can substitute for raw trajectory.
export HISTORY_KEEP_LAST="${HISTORY_KEEP_LAST:-8}"

case "${PROFILE_MODE}" in
    sid)  _UAW_FIELD="text_sid"; _UAW_SUF="_uaware";      _UAW_AUP_DEFAULT=1 ;;
    raw)  _UAW_FIELD="text_raw"; _UAW_SUF="_uaware_raw";  _UAW_AUP_DEFAULT=0 ;;
    none) _UAW_FIELD="none";     _UAW_SUF="_uaware_none"; _UAW_AUP_DEFAULT=0 ;;
    *) echo "[uaware] ERROR: PROFILE_MODE must be sid|raw|none (got '${PROFILE_MODE}')" >&2; return 1 ;;
esac
# Tag fused data + run dir by the history window so different HISTORY_KEEP_LAST
# values (a recency sweep) never share files or checkpoints.
if [[ "${HISTORY_KEEP_LAST}" -lt 0 ]]; then _UAW_HTAG="hfull"; else _UAW_HTAG="h${HISTORY_KEEP_LAST}"; fi
_UAW_SUF="${_UAW_SUF}_${_UAW_HTAG}"
export UAWARE_FIELD="${_UAW_FIELD}"
export UAWARE_TRAIN_JSON="${DATA_DIR}/llm_train${_UAW_SUF}.json"
export UAWARE_TEST_JSON="${DATA_DIR}/llm_test${_UAW_SUF}.json"

# Whether the embedding-alignment stage also learns profile->SID priors. When it
# does, the align data file is named distinctly so the POI-only arms (B0/B1) and
# the user-augmented arm (B2) never share a file in the same DATA_DIR.
export ALIGN_USER_PAIRS="${ALIGN_USER_PAIRS:-${_UAW_AUP_DEFAULT}}"
if [[ "${ALIGN_USER_PAIRS}" == "1" ]]; then
    export ALIGN_TRAIN_JSON="${DATA_DIR}/llm_align_uaware_train.json"
    export ALIGN_VAL_JSON="${DATA_DIR}/llm_align_uaware_val.json"
else
    export ALIGN_TRAIN_JSON="${DATA_DIR}/llm_align_train.json"
    export ALIGN_VAL_JSON="${DATA_DIR}/llm_align_val.json"
fi

# ------------------------- run layout (one dir per arm) --------------------
export UAWARE_RUN_NAME="${UAWARE_RUN_NAME:-${DATASET}_${_MODEL_TAG}_uaware_${PROFILE_MODE}_${_UAW_HTAG}}"
export RUN_DIR="${PROJECT_ROOT}/V2/runs/${UAWARE_RUN_NAME}"
export SID_DIR="${RUN_DIR}/sid"
export SFT_DIR="${RUN_DIR}/sft"
export EVAL_DIR="${RUN_DIR}/eval"
export MARKER_DIR="${RUN_DIR}/markers"
export ALIGN_DIR="${RUN_DIR}/align"

make_uaware_run_dirs() { make_run_dirs; mkdir -p "${ALIGN_DIR}"; }

print_uaware_config() {
    cat <<CFG
[env] ================= user-aware (GenUP x SID) config ==========
[env] PROFILE_MODE  : ${PROFILE_MODE}  (field='${UAWARE_FIELD}')  history_keep_last=${HISTORY_KEEP_LAST} ($( [[ "${HISTORY_KEEP_LAST}" -lt 0 ]] && echo 'full history' || echo 'GenUP recent-trajectory' ))
[env] RUN_DIR       : ${RUN_DIR}
[env] GenUP profiles: $( [[ -d "${GENUP_PROFILES_DIR}" ]] && echo present || echo MISSING ) (${GENUP_PROFILES_DIR})
[env] profiles json : $( [[ -f "${PROFILES_SID_JSON}" ]] && echo present || echo missing ) (${PROFILES_SID_JSON})  top_n=${PROFILE_TOP_N}
[env] sft data      : $( [[ -f "${UAWARE_TRAIN_JSON}" ]] && echo present || echo missing ) (${UAWARE_TRAIN_JSON})
[env] align stage   : enabled=${V2_ALIGN} user_pairs=${ALIGN_USER_PAIRS} epochs=${V2_ALIGN_EPOCHS} bs=${V2_ALIGN_BS}x${V2_ALIGN_ACCUM} len=${V2_ALIGN_MAX_SEQ_LEN}
[env] align data    : $( [[ -f "${ALIGN_TRAIN_JSON}" ]] && echo present || echo missing ) (${ALIGN_TRAIN_JSON})
[env] sft stage     : epochs=${V2_SFT_EPOCHS} bs=${V2_SFT_BS}x${V2_SFT_ACCUM} len=${V2_SFT_MAX_SEQ_LEN} lr=${V2_LR} (${V2_LR_SCHEDULER})
[env] eval          : max_new_tokens=${EVAL_MAX_NEW_TOKENS} num_beams=${NUM_BEAMS} Ks=[${EVAL_KS}]
[env] ============================================================
CFG
}
