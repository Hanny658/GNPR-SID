#!/usr/bin/env python
"""
build_uaware_data.py -- attach a user profile to each next-POI record.

Reads a GNPR-SID ``llm_*.json`` (Alpaca {instruction,input,output}), looks up
the record's user (the ``User_<id>`` already present in the ``input``), and adds
a ``profile`` field carrying that user's profile text. finetune_llm.py /
eval_llm.py render the profile in a protected ``### User Profile:`` block.

  --field text_sid   the SID-aware profile     -> B2 (our fusion)
  --field text_raw   GenUP's raw profile        -> B1 (user side, no SID-awareness)

A record whose user has no profile gets an empty ``profile`` (it then renders
exactly like the B0 baseline), so the output is always a superset-safe drop-in.
"""

import argparse
import json
import os
import re

USER_RE = re.compile(r"User_(\d+)")


def parse_args():
    p = argparse.ArgumentParser(description="Inject user profiles into llm_*.json")
    p.add_argument("--in_json", required=True, help="llm_train.json / llm_test.json")
    p.add_argument("--profiles", required=True, help="user_profiles_sid.json")
    p.add_argument("--out_json", required=True)
    p.add_argument("--field", default="text_sid",
                   choices=["text_sid", "text_raw"],
                   help="which profile text to inject")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.profiles, encoding="utf-8") as f:
        profiles = json.load(f)
    with open(args.in_json, encoding="utf-8") as f:
        items = json.load(f)

    out, covered, no_user = [], 0, 0
    for ex in items:
        m = USER_RE.search(str(ex.get("input", "")))
        text = ""
        if m:
            text = str(profiles.get(m.group(1), {}).get(args.field, "")).strip()
        else:
            no_user += 1
        if text:
            covered += 1
        ex2 = dict(ex)
        ex2["profile"] = text
        out.append(ex2)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    n = len(out)
    print(f"[uaware] {n} records -> {args.out_json}")
    print(f"[uaware]   profile attached ({args.field}): {covered}/{n} "
          f"({100.0 * covered / n if n else 0:.1f}%)")
    if no_user:
        print(f"[uaware]   records with no User_<id> in input: {no_user}")


if __name__ == "__main__":
    main()
