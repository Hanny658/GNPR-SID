#!/usr/bin/env python
"""
rewrite_profile_sid.py -- offline, deterministic *SID-aware* user profiles.

Fuses the user side of GenUP (SIGSPATIAL'25) with the POI side of GNPR-SID: it
takes GenUP's pre-generated user profiles (Big-Five traits, demographics,
preferences, routines, a 200-word narrative) and makes them speak the model's
*Semantic-ID* vocabulary, so user-side affinities live in the same token space
the recommender generates in. No OpenAI key / network needed -- it only rewrites
text deterministically using the GNPR-SID codebook.

Per user it produces three texts plus the user's top Semantic IDs:

  * ``text_raw``       GenUP profile verbatim (raw integer "POI id N" mentions).
                       -> the B1 ablation arm (user side WITHOUT SID-awareness).
  * ``text_sid_core``  Same, but every "POI id N" mention rewritten to its SID
                       ``<a_..><b_..><c_..>`` via the codebook. No affinity line.
                       -> the alignment-stage INPUT (profile -> SID priors).
  * ``text_sid``       ``text_sid_core`` + an appended "Likely semantic codes:"
                       line listing the user's most-frequent SIDs.
                       -> injected into the recommender prompt (the B2 arm).
  * ``top_sids``       The user's top-N most-frequent training SIDs (the
                       alignment-stage OUTPUT, and the affinity line's content).

Inputs (artifacts left by build_dataset.py + GenUP's repo):
  --codebook   codebook.csv          [pid, sid, vector]
  --seq_csv    train_poi_sequence.csv [UserId, sequence_PoiId, ...]
  --genup_profiles  <GenUP>/data/<ds>/user_profiles/user_profile_<uid>.json

Output:
  --out  user_profiles_sid.json  ->  {uid: {text_raw, text_sid_core, text_sid, top_sids}}

Consumed by build_uaware_data.py (injection) and build_align_data.py (alignment).
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter

# pandas + the build_dataset codebook loader are imported lazily inside the
# functions that need them, so this module's pure-text helpers (rewrite_poi_ids,
# assemble_profile) import with only the standard library.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# "POI id 4482", "POI IDs 556, 256", "POI IDs 1348 and 534", "(e.g., POI ID 7)".
POI_ID_RE = re.compile(
    r"POI\s+IDs?\s+((?:\d+)(?:\s*(?:,|and|&|/)\s*\d+)*)", re.IGNORECASE
)


def rewrite_poi_ids(text, pid2code):
    """Replace 'POI id(s) N[, M ...]' mentions with their SID tokens.

    Numbers that aren't in the codebook (test-only POIs, or values the LLM
    hallucinated) are dropped; if none of a mention's numbers map, the original
    text is left untouched. Returns (new_text, n_mentions_rewritten).
    """
    n = 0

    def _repl(m):
        nonlocal n
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        sids = [pid2code[p] for p in nums if p in pid2code]
        if not sids:
            return m.group(0)
        n += 1
        return "semantic codes " + ", ".join(sids)

    return POI_ID_RE.sub(_repl, text), n


def assemble_profile(profile, uid, narrative):
    """GenUP-style system-prompt text (port of create_sft_dataset.create_system_prompt)."""
    attrs = list(profile.get("attributes", []))
    attrs = (attrs + ["unknown"] * 4)[:4]
    age, gender, edu, socio = attrs
    traits = ", ".join(str(t) for t in profile.get("traits", []))
    prefs = ", ".join(str(t) for t in profile.get("preferences", []))
    routines = ", ".join(str(t) for t in profile.get("routines", []))
    return (
        f"You are user {uid} and your basic information is as follows:\n"
        f"Age: {age}; Gender: {gender}; Education: {edu}; SocioEco: {socio}.\n"
        f"You have the following traits: {traits}.\n"
        f"You have the following preferences: {prefs}.\n"
        f"You have the following routines: {routines}.\n"
        f"{narrative}"
    )


def user_top_sids(seq_csv, pid2code, top_n):
    """{uid_str: [most-frequent SID, ...]} from the user's training check-ins."""
    import pandas as pd
    df = pd.read_csv(seq_csv)
    out = {}
    for uid, g in df.groupby("UserId"):
        cnt = Counter()
        for row in g.itertuples(index=False):
            try:
                seq = ast.literal_eval(row.sequence_PoiId)
            except Exception:
                continue
            for p in seq:
                code = pid2code.get(int(p))
                if code:
                    cnt[code] += 1
        out[str(uid)] = [s for s, _ in cnt.most_common(top_n)]
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Offline SID-aware user profiles")
    p.add_argument("--codebook", required=True, help="codebook.csv [pid,sid,vector]")
    p.add_argument("--seq_csv", required=True, help="train_poi_sequence.csv")
    p.add_argument("--genup_profiles", required=True,
                   help="dir of GenUP user_profile_<uid>.json")
    p.add_argument("--out", required=True, help="output user_profiles_sid.json")
    p.add_argument("--top_n", type=int, default=int(os.environ.get("PROFILE_TOP_N", "10")))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.out) and not args.force:
        print(f"[profile] {args.out} exists; nothing to do (use --force to rebuild).")
        return

    # Reuse the canonical pid->SID loader (codebook.csv -> {pid: "<a_..>..."}).
    from build_dataset import _load_pid2code
    pid2code = _load_pid2code(args.codebook)
    print(f"[profile] codebook: {len(pid2code)} POIs -> SID")
    top_sids = user_top_sids(args.seq_csv, pid2code, args.top_n)
    print(f"[profile] top-{args.top_n} SIDs computed for {len(top_sids)} users")

    files = sorted(
        f for f in os.listdir(args.genup_profiles)
        if re.fullmatch(r"user_profile_\d+\.json", f)
    )
    if not files:
        print(f"[profile] ERROR: no user_profile_<uid>.json under "
              f"{args.genup_profiles}", file=sys.stderr)
        sys.exit(1)

    out, n_rewritten, n_with_aff, n_bad = {}, 0, 0, 0
    for fn in files:
        uid = fn[len("user_profile_"):-len(".json")]
        try:
            with open(os.path.join(args.genup_profiles, fn), encoding="utf-8") as f:
                prof = json.load(f)
        except Exception as e:
            print(f"[profile]   skip {fn}: {e}")
            n_bad += 1
            continue
        narrative = str(prof.get("user_profile", "")).strip()

        text_raw = assemble_profile(prof, uid, narrative)
        sid_narr, n = rewrite_poi_ids(narrative, pid2code)
        n_rewritten += n
        text_sid_core = assemble_profile(prof, uid, sid_narr)

        tops = top_sids.get(str(uid), [])
        if tops:
            text_sid = text_sid_core + "\nLikely semantic codes: " + ", ".join(tops) + "."
            n_with_aff += 1
        else:
            text_sid = text_sid_core

        out[str(uid)] = {
            "text_raw": text_raw,
            "text_sid_core": text_sid_core,
            "text_sid": text_sid,
            "top_sids": tops,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[profile] {len(out)} profiles -> {args.out}")
    print(f"[profile]   POI-id mentions rewritten to SIDs: {n_rewritten}")
    print(f"[profile]   profiles with an affinity line     : {n_with_aff}/{len(out)}")
    if n_bad:
        print(f"[profile]   unreadable profiles skipped        : {n_bad}")


if __name__ == "__main__":
    main()
