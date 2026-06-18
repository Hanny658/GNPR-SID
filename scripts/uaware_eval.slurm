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
#SBATCH --job-name=uaware_eval_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# User-aware pipeline, step 3: generative evaluation (same protocol as
# scripts/eval.slurm) on the profile-injected test set, plus a stratified
# Acc@1 breakdown (cold-start users / rare POIs) where the SID-aware profile is
# expected to win. When alignment ran, the SFT adapter sits on the aligned full
# model (which carries the SID-token embeddings), not the raw BASE_MODEL.
# Resumable: predictions stream to predictions.jsonl.
# ---------------------------------------------------------------------------
if   [[ -f scripts/uaware_env.sh ]]; then source scripts/uaware_env.sh
elif [[ -f uaware_env.sh         ]]; then source uaware_env.sh
else echo "[uaware-eval] ERROR: cannot find scripts/uaware_env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_uaware_run_dirs
print_config
print_uaware_config

MODEL_DIR="${SFT_DIR}/final"
TEST_JSON="${UAWARE_TEST_JSON}"
TRAIN_JSON="${DATA_DIR}/llm_train.json"
if [[ ! -f "${MODEL_DIR}/_FINETUNE_COMPLETE" ]]; then
    echo "[uaware-eval] ERROR: trained model not found at ${MODEL_DIR}." >&2
    echo "[uaware-eval]        Run  PROFILE_MODE=${PROFILE_MODE} sbatch scripts/uaware_train.slurm  first." >&2
    exit 1
fi
[[ -f "${TEST_JSON}" ]] || { echo "[uaware-eval] ERROR: missing ${TEST_JSON}" >&2; exit 1; }

# Base = aligned full model (carries the SID-token embeddings) when alignment ran.
EVAL_BASE="${BASE_MODEL}"
if phase_done uaware_align && [[ -f "${ALIGN_DIR}/final/_FINETUNE_COMPLETE" ]]; then
    EVAL_BASE="${ALIGN_DIR}/final"
fi

if phase_done uaware_eval; then
    echo "[uaware-eval] generation already complete (markers/uaware_eval.done). Metrics:"
    cat "${EVAL_DIR}/metrics.json" 2>/dev/null || true
else
    echo "[uaware-eval] evaluating ${MODEL_DIR} (base: ${EVAL_BASE}) on ${TEST_JSON}"
    python "${CODE_DIR}/eval_llm.py" \
        --model_dir "${MODEL_DIR}" \
        --base_model "${EVAL_BASE}" \
        --test_file "${TEST_JSON}" \
        --out_dir "${EVAL_DIR}" \
        --num_beams "${NUM_BEAMS}" \
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
        --max_prompt_len "${V2_SFT_MAX_SEQ_LEN}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --ks ${EVAL_KS}
    mark_done uaware_eval
fi

# ---- stratified Acc@1 (cold-start users / rare POIs) -----------------------
if [[ -f "${EVAL_DIR}/predictions.jsonl" && -f "${TRAIN_JSON}" ]]; then
    echo "[uaware-eval] stratified breakdown:"
    python "${CODE_DIR}/strat_eval.py" \
        --pred "${EVAL_DIR}/predictions.jsonl" \
        --test_json "${TEST_JSON}" \
        --train_json "${TRAIN_JSON}" \
        --out "${EVAL_DIR}/metrics_strat.json"
fi

echo "[uaware-eval] complete. Overall metrics:"
cat "${EVAL_DIR}/metrics.json"
