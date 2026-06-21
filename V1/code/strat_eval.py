#!/usr/bin/env python
"""
strat_eval.py -- stratified Acc@1 over a finished eval, for the GenUP x GNPR-SID
ablation. This is where the SID-aware user profile is expected to pay off:

  * by USER ACTIVITY     -- cold-start users (few training records) vs active
                            ones. The profile should help the thin-history users.
  * by GOLD-POI POPULARITY -- rare/cold POIs vs popular ones. SID prefix-sharing
                            should help the rare targets.

Read-only: consumes ``predictions.jsonl`` (from eval_llm.py), the test json (to
map each prediction to its user) and ``llm_train.json`` (to measure user
activity + SID popularity). Prints an Acc@1 breakdown and optionally writes it.
"""

import argparse
import json
import os
import re

SID_RUN_RE = re.compile(r"(?:<[abcd]_\d+>)+")
USER_RE = re.compile(r"User_(\d+)")


def _user_of(ex):
    m = USER_RE.search(str(ex.get("input", "")))
    return m.group(1) if m else None


def train_stats(train_json):
    """(records-per-user, SID-popularity) from the training set."""
    with open(train_json, encoding="utf-8") as f:
        items = json.load(f)
    activity, popularity = {}, {}
    for ex in items:
        inp = str(ex.get("input", ""))
        u = _user_of(ex)
        if u is not None:
            # user activity = total training check-ins (the cold-start axis),
            # counted as the "<time> visited <SID>" items in the history. Counting
            # records would be degenerate here (KEEP_LAST_K caps records/user).
            # NOTE: pass the BASE llm_train.json (full history), not a recent-only
            # uaware file, or this is capped at HISTORY_KEEP_LAST.
            activity[u] = activity.get(u, 0) + inp.count(" visited ")
        for field in ("input", "output"):
            for sid in SID_RUN_RE.findall(str(ex.get(field, ""))):
                popularity[sid] = popularity.get(sid, 0) + 1
    return activity, popularity


def terciles(values):
    """Two cut points splitting sorted values into low/mid/high thirds."""
    xs = sorted(values)
    if not xs:
        return (0, 0)
    return (xs[len(xs) // 3], xs[2 * len(xs) // 3])


def bin3(v, cuts, labels):
    lo, hi = cuts
    return labels[0] if v <= lo else (labels[1] if v <= hi else labels[2])


def _report(name, rows, cuts, labels):
    agg = {l: [0, 0] for l in labels}          # label -> [hits, n]
    for v, hit in rows:
        l = bin3(v, cuts, labels)
        agg[l][0] += hit
        agg[l][1] += 1
    out = {}
    print(f"\n[strat] by {name}  (tercile cuts at {cuts[0]} / {cuts[1]})")
    print(f"[strat]   {'stratum':<16}{'n':>7}{'Acc@1':>9}")
    for l in labels:
        hits, n = agg[l]
        acc = hits / n if n else 0.0
        out[l] = {"n": n, "Acc@1": acc}
        print(f"[strat]   {l:<16}{n:>7}{acc:>9.4f}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Stratified Acc@1 breakdown")
    p.add_argument("--pred", required=True, help="predictions.jsonl")
    p.add_argument("--test_json", required=True, help="test file used for eval")
    p.add_argument("--train_json", required=True, help="llm_train.json")
    p.add_argument("--out", default=None, help="optional metrics_strat.json")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.test_json, encoding="utf-8") as f:
        test = json.load(f)
    activity, popularity = train_stats(args.train_json)

    act_rows, pop_rows, hits, n = [], [], 0, 0
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds = rec.get("preds", [])
            hit = 1 if preds and preds[0] == rec.get("gold") else 0
            hits += hit
            n += 1
            idx = rec.get("idx")
            user = _user_of(test[idx]) if (isinstance(idx, int) and idx < len(test)) else None
            act_rows.append((activity.get(user, 0), hit))
            pop_rows.append((popularity.get(rec.get("gold"), 0), hit))

    overall = hits / n if n else 0.0
    print(f"[strat] n={n}  overall Acc@1={overall:.4f}")
    result = {"n": n, "Acc@1": overall}
    result["by_user_activity"] = _report(
        "user activity (train records/user)", act_rows,
        terciles([v for v, _ in act_rows]), ["cold", "mid", "active"])
    result["by_poi_popularity"] = _report(
        "gold-POI popularity (train SID count)", pop_rows,
        terciles([v for v, _ in pop_rows]), ["rare", "mid", "popular"])

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[strat] written -> {args.out}")


if __name__ == "__main__":
    main()
