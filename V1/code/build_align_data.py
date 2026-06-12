#!/usr/bin/env python
"""
Build the SID <-> POI-attribute *alignment* dataset for the V2 pipeline.

Headless port of ``V2/dataprocess/get_align_data.ipynb``: for every POI it
emits two instruction samples --

  1. attributes -> Semantic ID   ("predict the POI semantic code")
  2. Semantic ID -> attributes   ("describe the POI semantic code attributes")

-- then shuffles and splits 90/10 into train/valid JSON (Alpaca schema), the
data ``V2/LLM/train/align_sid.py`` trains its embedding-only LoRA on.

Inputs are the artifacts ``build_dataset.py`` leaves in the dataset dir:
  poi_info.csv   [pid, category, latitude, longitude, visit_time_and_count]
  codebook.csv   [pid, sid, vector]   (sid = "[a, b, c]" or "[a, b, c, d]")

Note: the notebook's poi_info also had a ``region`` column; ours does not, so
the Region field is included only when present. The notebook shuffles without
a seed; we seed for reproducibility.
"""

import argparse
import ast
import json
import os
import random
import sys

import pandas as pd

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def code_to_tag(code_list):
    return "".join(f"<{_LETTERS[i]}_{int(v)}>" for i, v in enumerate(code_list))


def build_items(poi_info_csv, codebook_csv):
    info = pd.read_csv(poi_info_csv)
    codes = pd.read_csv(codebook_csv)
    merged = info.merge(codes[["pid", "sid"]], on="pid", how="left")
    has_region = "region" in merged.columns

    items, skipped = [], 0
    for row in merged.itertuples(index=False):
        sid = getattr(row, "sid")
        if not isinstance(sid, str):
            skipped += 1
            continue
        tag = code_to_tag(ast.literal_eval(sid))
        attrs = f"Category: {row.category}; "
        if has_region:
            attrs += f"Region: {row.region}; "
        attrs += (f"Latitude: {row.latitude}; "
                  f"Longitude: {row.longitude}; "
                  f"Visit_time_and_count: {row.visit_time_and_count}")
        items.append({
            "instruction": "Given a POI attributes, describe its semantic code.",
            "input": "Can you based on the attributes {" + attrs
                     + "} to predict the POI semantic code?",
            "output": tag,
        })
        items.append({
            "instruction": "Given a semantic code, describe its POI attributes.",
            "input": f"Can you describe the POI semantic code {tag} attributes?",
            "output": "{" + attrs + "}",
        })
    if skipped:
        print(f"[align] WARN: {skipped} POIs had no SID in {codebook_csv}; skipped.")
    return items


def parse_args():
    p = argparse.ArgumentParser(description="V2 SID<->attribute alignment data")
    p.add_argument("--poi_info", required=True)
    p.add_argument("--codebook", required=True)
    p.add_argument("--out_train", required=True)
    p.add_argument("--out_val", required=True)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.out_train) and os.path.exists(args.out_val):
        print(f"[align] {args.out_train} + {args.out_val} already exist; nothing to do.")
        return

    items = build_items(args.poi_info, args.codebook)
    if not items:
        print("[align] ERROR: no alignment samples produced; check inputs.", file=sys.stderr)
        sys.exit(1)

    random.Random(args.seed).shuffle(items)
    n_val = int(len(items) * args.val_frac)
    val, train = items[:n_val], items[n_val:]

    for path, data in ((args.out_train, train), (args.out_val, val)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[align] {len(train)} train -> {args.out_train}")
    print(f"[align] {len(val)} valid -> {args.out_val}")


if __name__ == "__main__":
    main()
