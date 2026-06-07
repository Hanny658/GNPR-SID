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
#SBATCH --job-name=eval_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Generative evaluation: beam-search top-k Semantic-ID generation on the test
# set, then Acc@k / MRR / NDCG@k.  Resumable -- predictions stream to
# predictions.jsonl, so a re-submitted job continues from where it stopped.
# ---------------------------------------------------------------------------
if   [[ -f scripts/env.sh ]]; then source scripts/env.sh
elif [[ -f env.sh         ]]; then source env.sh
else echo "[eval] ERROR: cannot find scripts/env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_run_dirs
print_config

if phase_done eval; then
    echo "[eval] already complete (markers/eval.done present). Metrics:"
    cat "${EVAL_DIR}/metrics.json" 2>/dev/null || true
    exit 0
fi

MODEL_DIR="${SFT_DIR}/final"
TEST_JSON="${DATA_DIR}/llm_test.json"
if [[ ! -f "${MODEL_DIR}/_FINETUNE_COMPLETE" ]]; then
    echo "[eval] ERROR: fine-tuned model not found at ${MODEL_DIR}." >&2
    echo "[eval]        Run  sbatch scripts/train.slurm  first." >&2
    exit 1
fi
[[ -f "${TEST_JSON}" ]] || { echo "[eval] ERROR: missing ${TEST_JSON}" >&2; exit 1; }

echo "[eval] evaluating ${MODEL_DIR} on ${TEST_JSON}"
python "${CODE_DIR}/eval_llm.py" \
    --model_dir "${MODEL_DIR}" \
    --base_model "${BASE_MODEL}" \
    --test_file "${TEST_JSON}" \
    --out_dir "${EVAL_DIR}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
    --max_prompt_len "${MAX_SEQ_LEN}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --ks ${EVAL_KS}

mark_done eval
echo "[eval] complete. Final metrics:"
cat "${EVAL_DIR}/metrics.json"
