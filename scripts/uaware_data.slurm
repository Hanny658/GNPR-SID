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
#SBATCH --job-name=uaware_data_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# User-aware (GenUP x GNPR-SID) pipeline, step 1: build everything the train /
# eval steps consume, on ONE consistent Semantic-ID space. Idempotent -- every
# step skips when its outputs already exist.
#
#   a) Semantic-ID dataset      llm_*.json + poi_info.csv + codebook.csv
#                               (delegates to scripts/data.slurm)
#   b) POI-only alignment data  llm_align_{train,val}.json        (B0/B1)
#   c) SID-aware profiles       user_profiles_sid.json            (offline rewrite)
#   d) fused datasets           llm_{train,test}_uaware[_raw].json (B2 / B1)
#   e) user-augmented alignment llm_align_uaware_{train,val}.json  (B2)
# Steps c-e are skipped (with a warning) when GenUP profiles are unavailable;
# the B0 arm only needs (a)+(b).
# ---------------------------------------------------------------------------
if   [[ -f scripts/uaware_env.sh ]]; then source scripts/uaware_env.sh
elif [[ -f uaware_env.sh         ]]; then source uaware_env.sh
else echo "[uaware-data] ERROR: cannot find scripts/uaware_env.sh" >&2; exit 1; fi

load_modules
activate_conda
make_uaware_run_dirs
print_config
print_uaware_config

# ---- (a) Semantic-ID dataset (llm_*.json + poi_info + codebook) ------------
if [[ -f "${DATA_DIR}/llm_train.json" && -f "${DATA_DIR}/llm_test.json" ]]; then
    echo "[uaware-data] llm_*.json already present for ${DATASET}; SID build skipped."
else
    echo "[uaware-data] building Semantic-ID dataset via scripts/data.slurm"
    bash "${PROJECT_ROOT}/scripts/data.slurm"
fi

POI_INFO="${DATA_DIR}/poi_info.csv"
CODEBOOK="${DATA_DIR}/codebook.csv"
SEQ_CSV="${DATA_DIR}/train_poi_sequence.csv"

# Alignment (all arms) AND the profile rewrite need poi_info + codebook. A
# prebaked dataset (NYC) ships llm_*.json but NOT these -> force a from-raw build.
if [[ ! -f "${POI_INFO}" || ! -f "${CODEBOOK}" ]]; then
    echo "[uaware-data] ERROR: ${POI_INFO} / ${CODEBOOK} missing." >&2
    echo "[uaware-data]   The user-aware pipeline needs the SID codebook + POI" >&2
    echo "[uaware-data]   attributes on one consistent SID space. ${DATASET} looks" >&2
    echo "[uaware-data]   prebaked (ships llm_*.json only). Rebuild it from raw:" >&2
    echo "[uaware-data]     rm -f ${DATA_DIR}/llm_*.json" >&2
    echo "[uaware-data]     PREBAKED_DATASETS=\"\" DATASET=${DATASET} sbatch scripts/data.slurm" >&2
    echo "[uaware-data]   then re-run this script." >&2
    exit 1
fi

# ---- (b) POI-only alignment data (needed by every arm) ---------------------
if [[ -f "${DATA_DIR}/llm_align_train.json" && -f "${DATA_DIR}/llm_align_val.json" ]]; then
    echo "[uaware-data] POI-only alignment data already present; skipping."
else
    echo "[uaware-data] building POI<->attribute alignment data"
    python "${CODE_DIR}/build_align_data.py" \
        --poi_info "${POI_INFO}" \
        --codebook "${CODEBOOK}" \
        --out_train "${DATA_DIR}/llm_align_train.json" \
        --out_val   "${DATA_DIR}/llm_align_val.json"
fi

# ---- GenUP method: the prompt carries only the recent trajectory -----------
# (HISTORY_KEEP_LAST visits); the user's full history reaches the model via the
# profile. Files are tagged by the window so a recency sweep never collides.
HTAG=$( [[ "${HISTORY_KEEP_LAST}" -lt 0 ]] && echo hfull || echo "h${HISTORY_KEEP_LAST}" )
echo "[uaware-data] fused datasets: history_keep_last=${HISTORY_KEEP_LAST} (tag ${HTAG})"

# (b2) 'none' arm (no profile) -- always buildable, even without GenUP profiles.
for split in train test; do
    src="${DATA_DIR}/llm_${split}.json"
    [[ -f "${src}" ]] || { echo "[uaware-data] WARN: ${src} missing; skipping ${split}."; continue; }
    out="${DATA_DIR}/llm_${split}_uaware_none_${HTAG}.json"
    if [[ -f "${out}" ]]; then echo "[uaware-data] ${out} present; skipping."
    else python "${CODE_DIR}/build_uaware_data.py" \
            --in_json "${src}" --field none \
            --history_keep_last "${HISTORY_KEEP_LAST}" --out_json "${out}"; fi
done

# ---- (c)-(e) profile-dependent artifacts (raw/sid arms) --------------------
if [[ ! -d "${GENUP_PROFILES_DIR}" ]]; then
    echo "[uaware-data] WARN: GenUP profiles not found at ${GENUP_PROFILES_DIR}." >&2
    echo "[uaware-data]       Only the none arm (PROFILE_MODE=none) can run." >&2
    echo "[uaware-data]       Set GENUP_DIR/GENUP_PROFILES_DIR to GenUP's data/<ds>/user_profiles." >&2
    echo "[uaware-data] done (none-only). next: PROFILE_MODE=none sbatch scripts/uaware_train.slurm"
    exit 0
fi

# (c) SID-aware profiles
if [[ -f "${PROFILES_SID_JSON}" ]]; then
    echo "[uaware-data] ${PROFILES_SID_JSON} present; profile rewrite skipped."
else
    echo "[uaware-data] rewriting GenUP profiles -> SID-aware (offline)"
    python "${CODE_DIR}/rewrite_profile_sid.py" \
        --codebook "${CODEBOOK}" \
        --seq_csv "${SEQ_CSV}" \
        --genup_profiles "${GENUP_PROFILES_DIR}" \
        --out "${PROFILES_SID_JSON}" \
        --top_n "${PROFILE_TOP_N}"
fi

# (d) raw + sid fused datasets (recent-trajectory prompt + profile)
for split in train test; do
    src="${DATA_DIR}/llm_${split}.json"
    [[ -f "${src}" ]] || continue
    for mode in sid raw; do
        if [[ "${mode}" == "sid" ]]; then field="text_sid"; out="${DATA_DIR}/llm_${split}_uaware_${HTAG}.json"
        else                              field="text_raw"; out="${DATA_DIR}/llm_${split}_uaware_raw_${HTAG}.json"; fi
        if [[ -f "${out}" ]]; then echo "[uaware-data] ${out} present; skipping."
        else python "${CODE_DIR}/build_uaware_data.py" \
                --in_json "${src}" --profiles "${PROFILES_SID_JSON}" \
                --field "${field}" --history_keep_last "${HISTORY_KEEP_LAST}" --out_json "${out}"; fi
    done
done

# (e) user-augmented alignment data (POI + profile<->SID pairs) -> B2
if [[ -f "${DATA_DIR}/llm_align_uaware_train.json" && -f "${DATA_DIR}/llm_align_uaware_val.json" ]]; then
    echo "[uaware-data] user-augmented alignment data already present; skipping."
else
    echo "[uaware-data] building POI + profile<->SID alignment data"
    python "${CODE_DIR}/build_align_data.py" \
        --poi_info "${POI_INFO}" \
        --codebook "${CODEBOOK}" \
        --user_profiles "${PROFILES_SID_JSON}" \
        --out_train "${DATA_DIR}/llm_align_uaware_train.json" \
        --out_val   "${DATA_DIR}/llm_align_uaware_val.json"
fi

echo "[uaware-data] done. next: PROFILE_MODE=${PROFILE_MODE} sbatch scripts/uaware_train.slurm"
