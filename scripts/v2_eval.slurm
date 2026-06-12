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
#SBATCH --job-name=v2_eval_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Fully-V2 pipeline, step 3: generative evaluation (same protocol as
# scripts/eval.slurm -- beam-search SID generation, Acc@k / MRR / NDCG@k) on
# the V2-trained adapter. When the alignment stage ran, the adapter sits on
# top of the MERGED base (align_merged/), not the raw BASE_MODEL.
# Resumable: predictions stream to predictions.jsonl.
# ---------------------------------------------------------------------------
if   [[ -f scripts/v2_env.sh ]]; then source scripts/v2_env.sh
elif [[ -f v2_env.sh         ]]; then source v2_env.sh
else echo "[v2-eval] ERROR: cannot find scripts/v2_env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_v2_run_dirs
print_config
print_v2_config

if phase_done v2_eval; then
    echo "[v2-eval] already complete (markers/v2_eval.done present). Metrics:"
    cat "${EVAL_DIR}/metrics.json" 2>/dev/null || true
    exit 0
fi

MODEL_DIR="${SFT_DIR}/final"
TEST_JSON="${DATA_DIR}/llm_test.json"
if [[ ! -f "${MODEL_DIR}/_FINETUNE_COMPLETE" ]]; then
    echo "[v2-eval] ERROR: V2 fine-tuned model not found at ${MODEL_DIR}." >&2
    echo "[v2-eval]        Run  DATASET=${DATASET} sbatch scripts/v2_train.slurm  first." >&2
    exit 1
fi
[[ -f "${TEST_JSON}" ]] || { echo "[v2-eval] ERROR: missing ${TEST_JSON}" >&2; exit 1; }

# The SFT adapter was trained on the alignment-merged base when phase B ran.
EVAL_BASE="${BASE_MODEL}"
if phase_done v2_merge && [[ -f "${MERGED_DIR}/_MERGE_COMPLETE" ]]; then
    EVAL_BASE="${MERGED_DIR}"
fi

echo "[v2-eval] evaluating ${MODEL_DIR} (base: ${EVAL_BASE}) on ${TEST_JSON}"
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

mark_done v2_eval
echo "[v2-eval] complete. Final metrics:"
cat "${EVAL_DIR}/metrics.json"
