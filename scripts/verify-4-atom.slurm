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
### Specify Time Limit (short -- this is a read-only analysis) ###
#SBATCH --time=00:20:00
#
#SBATCH --job-name=verify_4atom_gnpr_sid
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p log

# ---------------------------------------------------------------------------
# Diagnostic: 4-atom Semantic-ID coverage / truncation check for a V2 run.
#
# Tests the hypothesis that EVAL_MAX_NEW_TOKENS is too small to emit 4-atom
# (collision-disambiguated <a><b><c><d>) SIDs, so those targets are auto-missed.
# It is READ-ONLY (no training, no generation) and prints everything to the
# job's .out log:
#   1. gold-SID atom-count distribution in llm_test.json (how many are 4-atom),
#   2. with the *actual eval tokenizer*, how many new tokens each gold SID needs
#      and what fraction cannot fit in EVAL_MAX_NEW_TOKENS (=> guaranteed miss),
#   3. if predictions.jsonl exists: Acc@1 split by gold atom-count, the
#      atom-count distribution of the model's top-1 predictions ("coverage" --
#      does it ever emit 4-atom codes?), and the prefix-truncation signature.
#
# Sources v2_env.sh so paths resolve to the V2 run (V2/runs/<ds>_<model>_v2).
# To inspect a V1-recipe run instead, swap the source line to env.sh.
# Usage: DATASET=tky sbatch scripts/verify-4-atom.slurm
# ---------------------------------------------------------------------------
if   [[ -f scripts/v2_env.sh ]]; then source scripts/v2_env.sh
elif [[ -f v2_env.sh         ]]; then source v2_env.sh
else echo "[verify] ERROR: cannot find scripts/v2_env.sh" >&2; exit 1; fi

load_modules
activate_conda
print_config
print_v2_config

python - <<'PY'
import json, os, re, statistics, collections

DATA_DIR   = os.environ["DATA_DIR"]
EVAL_DIR   = os.environ["EVAL_DIR"]
SFT_DIR    = os.environ["SFT_DIR"]
BASE_MODEL = os.environ.get("BASE_MODEL", "")
MAX_NEW    = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "12"))

test_json = os.path.join(DATA_DIR, "llm_test.json")
pred_path = os.path.join(EVAL_DIR, "predictions.jsonl")
model_dir = os.path.join(SFT_DIR, "final")
tok_src   = model_dir if os.path.isdir(model_dir) else BASE_MODEL

SID_RUN = re.compile(r"(?:<[abcd]_\d+>)+")
ATOM    = re.compile(r"<([abcd])_\d+>")

def first_sid(t):
    m = SID_RUN.search(t or "")
    return m.group(0) if m else ""

def natoms(s):
    return len(ATOM.findall(s or ""))

print("=" * 64)
print("[verify] 4-atom SID coverage / truncation check")
print(f"[verify] DATASET            : {os.environ.get('DATASET')}")
print(f"[verify] EVAL_MAX_NEW_TOKENS : {MAX_NEW}")
print(f"[verify] ADD_SID_TOKENS      : {os.environ.get('ADD_SID_TOKENS', '0')}")
print(f"[verify] test json          : {test_json}")
print(f"[verify] predictions        : {pred_path}")
print(f"[verify] tokenizer source    : {tok_src}")
print("=" * 64)

# --- (1) gold atom-count distribution ---------------------------------------
golds = []
if os.path.exists(test_json):
    items = json.load(open(test_json, encoding="utf-8"))
    golds = [g for g in (first_sid(it.get("output", "")) for it in items) if g]
    n = len(golds)
    dist = collections.Counter(natoms(g) for g in golds)
    print(f"\n[gold] {n} test targets with a parseable SID")
    for k in sorted(dist):
        print(f"[gold]   {k}-atom: {dist[k]:6d}  ({100*dist[k]/n:5.1f}%)")
else:
    print("\n[gold] WARN: llm_test.json not found; skipping gold distribution.")

# --- (2) tokenizer-based length check ---------------------------------------
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tlen = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    if golds:
        print(f"\n[tokens] new-token cost to emit the gold SID "
              f"(must be <= {MAX_NEW}; the model also needs +1 for EOS)")
        by = collections.defaultdict(list)
        for g in golds:
            by[natoms(g)].append(tlen(g))
        for k in sorted(by):
            L = by[k]
            over = sum(1 for x in L if x > MAX_NEW)
            print(f"[tokens]   {k}-atom: n={len(L):6d}  "
                  f"min/median/max = {min(L)}/{int(statistics.median(L))}/{max(L)}  "
                  f"> {MAX_NEW} toks: {over} ({100*over/len(L):.1f}%)")
        allL = [x for L in by.values() for x in L]
        over = sum(1 for x in allL if x > MAX_NEW)
        print(f"[tokens]   ==> gold SIDs that CANNOT fit in {MAX_NEW} new tokens "
              f"(guaranteed Acc@1 miss): {over}/{len(allL)} ({100*over/len(allL):.1f}%)")
        for g in sorted(set(golds), key=lambda s: -tlen(s))[:3]:
            print(f"[tokens]   longest e.g. {g} -> {tlen(g)} tokens")
except Exception as e:
    print(f"\n[tokens] WARN: tokenizer length check skipped ({type(e).__name__}: {e})")

# --- (3) prediction outcome / coverage --------------------------------------
if os.path.exists(pred_path):
    recs = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    n = len(recs)
    hit1 = sum(1 for r in recs if r["preds"][:1] == [r["gold"]])
    print(f"\n[pred] {n} predictions; overall Acc@1 = {hit1}/{n} = {hit1/n:.4f}")

    groups = collections.defaultdict(list)
    for r in recs:
        groups[natoms(r["gold"])].append(r)
    print("[pred] Acc@1 split by gold atom-count:")
    for k in sorted(groups):
        rs = groups[k]
        h = sum(1 for r in rs if r["preds"][:1] == [r["gold"]])
        print(f"[pred]   {k}-atom golds: n={len(rs):6d}  Acc@1={h/len(rs):.4f}")

    pdist = collections.Counter(natoms(r["preds"][0]) if r["preds"] else 0 for r in recs)
    print("[pred] COVERAGE -- atom-count of the model's top-1 prediction:")
    for k in sorted(pdist):
        lbl = f"{k}-atom" if k else "empty/none"
        print(f"[pred]   {lbl:11s}: {pdist[k]:6d}  ({100*pdist[k]/n:5.1f}%)")

    four = [r for r in recs if natoms(r["gold"]) >= 4]
    if four:
        missed = [r for r in four if r["preds"][:1] != [r["gold"]]]
        prefix = sum(1 for r in missed
                     if r["preds"] and r["preds"][0] != r["gold"]
                     and r["gold"].startswith(r["preds"][0]))
        dn = max(1, len(missed))
        print(f"[pred] 4-atom golds: {len(four)} total, {len(missed)} missed; of the misses, "
              f"{prefix} ({100*prefix/dn:.1f}%) had top-pred = a PREFIX of the gold "
              f"(truncation signature).")
else:
    print("\n[pred] no predictions.jsonl found at the V2 eval dir -- ran the")
    print("[pred] gold + token-length checks only. Produce predictions with:")
    print("[pred]     DATASET=$DATASET sbatch scripts/v2_eval.slurm")

print("\n[verify] done.")
PY

echo "[verify] analysis complete (see the [gold]/[tokens]/[pred] lines above)."
