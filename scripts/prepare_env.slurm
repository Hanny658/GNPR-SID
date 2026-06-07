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
#SBATCH --job-name=env_prep_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Build a project-local conda env using the cluster *modules* (anaconda + cuda)
# instead of compiling toolchains from scratch.  Idempotent: re-submitting the
# same script is a no-op once the env is ready (override with FORCE_REBUILD=1).
# ---------------------------------------------------------------------------
# locate scripts/env.sh whether submitted from the repo root or scripts/
if   [[ -f scripts/env.sh ]]; then source scripts/env.sh
elif [[ -f env.sh         ]]; then source env.sh
else echo "[prep] ERROR: cannot find scripts/env.sh" >&2; exit 1; fi

print_config
load_modules

# torch wheel MUST match the cluster CUDA module (cuda/12.8.0 -> cu128).
# Pitfall: the cu124 index caps torch at 2.6, which lacks `torch.float8_e8m0fnu`
# (added in torch 2.7). The latest `transformers` references that dtype at import
# time (integrations/finegrained_fp8.py), so a cu124/torch-2.6 env crashes with
#   AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
# the moment `from transformers import Trainer` runs. cu128 ships torch >=2.7 and
# matches the loaded CUDA module, keeping the whole stack consistent.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.7,<2.9}"
READY_MARK="${CONDA_ENV_PREFIX}/.env_ready"

if [[ -f "${READY_MARK}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[prep] env already prepared at ${CONDA_ENV_PREFIX} (FORCE_REBUILD=1 to rebuild)."
    activate_conda
    # Env is built, but the *data* for this DATASET may still be missing -- this
    # lets `DATASET=tky sbatch scripts/prepare_env.slurm` fetch raw data without
    # rebuilding the whole env. Skips nyc / already-present data.
    echo "[prep] ensuring raw data for DATASET=${DATASET}"
    fetch_dataset || echo "[prep] WARN: raw data not fetched (see message above)."
    echo "[prep] nothing else to do."
    exit 0
fi

# ---- create env if needed ----
if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
    echo "[prep] creating conda env at ${CONDA_ENV_PREFIX} (python ${PY_VERSION:-3.11})"
    conda create -y -p "${CONDA_ENV_PREFIX}" "python=${PY_VERSION:-3.11}"
fi

activate_conda

# ---- python deps ----
python -m pip install --upgrade pip wheel setuptools

echo "[prep] installing torch from ${TORCH_INDEX_URL}"
python -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"

echo "[prep] installing pipeline requirements"
python -m pip install -r "${PROJECT_ROOT}/scripts/requirements.txt"

# ---- sanity: torch sees the GPU ----
python - <<'PY'
import torch
print(f"[prep] torch {torch.__version__} | cuda build {torch.version.cuda} | "
      f"cuda avail {torch.cuda.is_available()}")
PY

# ---- sanity: the torch<->transformers<->peft stack actually imports ----
# This is the exact path that crashed before (transformers' Trainer pulls in the
# fp8 integration that touches torch.float8_e8m0fnu). Importing the real training
# entry points here makes a version skew fail *now*, during env-prep, instead of
# six hours into a training job.
python - <<'PY'
import torch, transformers, peft, accelerate
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model  # noqa: F401
print(f"[prep] import OK: transformers {transformers.__version__} | "
      f"peft {peft.__version__} | accelerate {accelerate.__version__}")
PY

# ---- pre-download the base model into the project-local HF cache ----
# (best-effort; if the compute node is offline it will be fetched at train time)
echo "[prep] pre-fetching ${BASE_MODEL} into ${HF_HOME}"
python - <<PY || echo "[prep] WARN: base model pre-fetch skipped/failed (will retry at train time)"
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["BASE_MODEL"],
                  token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["*.json", "*.model", "*.safetensors", "tokenizer*", "*.txt"])
print("[prep] base model cached")
PY

# ---- fetch the dataset's raw check-ins (skips nyc / already-present data) ----
# Downloads into RAW_DIR so scripts/data.slurm can build llm_*.json offline.
# Non-fatal: a missing source only warns (it doesn't break the env build), and
# for non-prebaked datasets we also pre-cache the category encoder.
echo "[prep] ensuring raw data for DATASET=${DATASET}"
fetch_dataset || echo "[prep] WARN: raw data not fetched (see message above); env build continues."
if ! is_prebaked; then
    echo "[prep] pre-fetching category encoder ${CAT_MODEL}"
    python - <<PY || echo "[prep] WARN: category encoder pre-fetch skipped/failed (retried at build time)"
import os
from sentence_transformers import SentenceTransformer
SentenceTransformer(os.environ.get("CAT_MODEL", "all-MiniLM-L6-v2"))
print("[prep] category encoder cached")
PY
fi

mkdir -p "${CONDA_ENV_PREFIX}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${READY_MARK}"
echo "[prep] DONE. env ready -> ${READY_MARK}"
