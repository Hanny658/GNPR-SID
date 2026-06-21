#!/usr/bin/env python
"""
build_uaware_data.py -- assemble the GenUP-style fused next-POI dataset.

Reads a GNPR-SID ``llm_*.json`` (Alpaca {instruction,input,output}) and produces
a record that follows GenUP's recipe: the user's LONG-TERM history is summarised
*offline* into the profile (see rewrite_profile_sid.py), and the PROMPT carries
only the **recent trajectory** -- not the full history. Two transforms:

  * ``--field {text_sid|text_raw|none}`` adds a ``profile`` field (the SID-aware
    profile -> B2; GenUP's raw profile -> B1; empty -> the no-profile baseline).
  * ``--history_keep_last K`` truncates the in-prompt check-in history to the
    most recent K visits (K=0 drops it entirely; K<0 keeps the full history, the
    old GNPR-SID behaviour). The long-term signal still reaches the model through
    the profile, which is built from the user's *full* history.

This is how the pipeline leans from "GNPR-SID + profile" (full history in the
prompt) toward "GenUP + SID" (profile + recent trajectory) so one can study
whether Semantic IDs help the cold-start regime under GenUP's method.

A record whose user has no profile gets an empty ``profile`` (renders like the
baseline), so the output is always a drop-in.
"""

import argparse
import json
import os
import re

USER_RE = re.compile(r"User_(\d+)")
# "...checkin history: <v1>, <v2>, ..., <vN>.\nWhen <t> user_X is likely to visit:"
HIST_RE = re.compile(r"^(.*?checkin history: )(.*?)(\.\n.*)$", re.S)


def truncate_history(input_str, keep_last):
    """Keep only the most recent ``keep_last`` visits in the prompt history.

    keep_last < 0 -> unchanged (full history); 0 -> no visits. Visits are
    ", "-joined "<time> visited <SID>" items (neither field contains ", ").
    """
    if keep_last is None or keep_last < 0:
        return input_str
    m = HIST_RE.match(input_str)
    if not m:
        return input_str  # unexpected format -> leave as-is
    prefix, visits, suffix = m.group(1), m.group(2), m.group(3)
    parts = [v for v in visits.split(", ") if v]
    kept = "" if keep_last == 0 else ", ".join(parts[-keep_last:])
    return prefix + kept + suffix


def parse_args():
    p = argparse.ArgumentParser(description="Build the GenUP-style fused dataset")
    p.add_argument("--in_json", required=True, help="llm_train.json / llm_test.json")
    p.add_argument("--profiles", default=None, help="user_profiles_sid.json (not needed for --field none)")
    p.add_argument("--out_json", required=True)
    p.add_argument("--field", default="text_sid",
                   choices=["text_sid", "text_raw", "none"],
                   help="which profile text to inject (none = no profile)")
    p.add_argument("--history_keep_last", type=int,
                   default=int(os.environ.get("HISTORY_KEEP_LAST", "-1")),
                   help="keep only the last K visits in the prompt (<0 = full history)")
    return p.parse_args()


def main():
    args = parse_args()
    profiles = {}
    if args.field != "none":
        if not args.profiles:
            raise SystemExit(f"[uaware] --profiles is required for --field {args.field}")
        with open(args.profiles, encoding="utf-8") as f:
            profiles = json.load(f)
    with open(args.in_json, encoding="utf-8") as f:
        items = json.load(f)

    out, covered, no_user = [], 0, 0
    for ex in items:
        m = USER_RE.search(str(ex.get("input", "")))
        text = ""
        if args.field != "none":
            if m:
                text = str(profiles.get(m.group(1), {}).get(args.field, "")).strip()
            else:
                no_user += 1
        if text:
            covered += 1
        ex2 = dict(ex)
        ex2["input"] = truncate_history(str(ex.get("input", "")), args.history_keep_last)
        ex2["profile"] = text
        out.append(ex2)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    n = len(out)
    kl = "full" if args.history_keep_last < 0 else str(args.history_keep_last)
    print(f"[uaware] {n} records -> {args.out_json}  (history_keep_last={kl})")
    if args.field != "none":
        print(f"[uaware]   profile attached ({args.field}): {covered}/{n} "
              f"({100.0 * covered / n if n else 0:.1f}%)")
        if no_user:
            print(f"[uaware]   records with no User_<id> in input: {no_user}")


if __name__ == "__main__":
    main()
