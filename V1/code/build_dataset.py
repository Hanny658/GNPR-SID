#!/usr/bin/env python
"""
build_dataset.py -- headless Semantic-ID data builder (V2 CRQVAE module).

Turns the raw LLM4POI check-in CSVs into the ``llm_{train,val,test}.json`` that
``finetune_llm.py`` / ``eval_llm.py`` consume, using the **V2** CRQVAE Semantic-ID
module (cosine-similarity quantisation + EMA).  This is the path for datasets that
do NOT ship pre-baked SIDs (TKY, CA); NYC already ships the finished JSON.

Pipeline -- every stage skips if its output already exists, so the job is
idempotent / resumable:

  1. sequences + poi_info    raw sample.csv         -> {train,validation,test}_poi_sequence.csv, poi_info.csv
  2. category embeddings     poi_info.csv           -> category_to_embedding.pkl  (sentence-transformers + PCA)
  3. poi feature vectors     poi_info + cat pkl     -> poi_Emb_dict.pkl           (cat + geo3 + fourier-time12)
  4. train CRQVAE + emit SID poi_Emb_dict.pkl       -> codebook.csv [pid,sid,vector]
  5. llm json                seq csv + codebook.csv -> llm_{train,val,test}.json

The CRQVAE model / Trainer / dataset are imported from ``V2/SID`` so the actual
quantiser is the authors' code; only the (notebook) data wrangling and the
glue / path handling live here.
"""

import argparse
import ast
import json
import os
import pickle
import re
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INSTRUCTION = (
    "Here is a record of a user's POI accesses, your task is based on the history "
    "to predict the POI that the user is likely to access at the specified time."
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _exists(*paths):
    return all(os.path.exists(p) for p in paths)


def extract_category_name(x):
    """Categories are plain strings (NYC/TKY) or a JSON list-of-dicts (CA)."""
    if pd.isna(x):
        return None
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0].get("name", None)
    except Exception:
        pass
    return str(x)


def latlon_to_3d(lat, lon):
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    return np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])


def extract_time_features2(time_dict):
    """12-d Fourier encoding of the hour-of-day visit histogram (V2/POI2emb)."""
    if not time_dict:
        return np.zeros(12)
    hours = np.array(list(time_dict.keys()), dtype=float)
    counts = np.array(list(time_dict.values()), dtype=float)
    total = counts.sum()
    if total == 0:
        return np.zeros(12)
    weights = counts / total
    feats = []
    for k in range(1, 7):
        feats.append(np.sum(weights * np.sin(2 * np.pi * k * hours / 24.0)))
        feats.append(np.sum(weights * np.cos(2 * np.pi * k * hours / 24.0)))
    return np.array(feats)


def parse_time_dict(x):
    if pd.isna(x) or x in ("", "{}"):
        return {}
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}


def encode_item(item):
    """[12, 5, 7] -> '<a_12><b_5><c_7>' (4th level -> '<d_..>' for collisions)."""
    if not item:
        return ""
    labels = ["a", "b", "c", "d"]
    if len(item) > 4:
        raise ValueError(f"SID has more than 4 atoms: {item}")
    return "".join(f"<{labels[i]}_{v}>" for i, v in enumerate(item))


# --------------------------------------------------------------------------- #
# stage 1 -- raw check-ins -> trajectory sequences + poi_info
# --------------------------------------------------------------------------- #
def _load_raw(raw_dir):
    """Return (all_splits_df, train_split_df). Tolerant of file naming.

    Supports two packagings:
      * a combined sample.csv with a SplitTag column (original GNPR-SID notebook), and
      * the HF w11wo/LLM4POI layout, which ships only train_sample.csv (no SplitTag;
        the whole file IS the train split -- its test set lives in a separate
        test_qa_pairs_kqt.txt handled by stage_test_from_qa()).
    """
    sample = os.path.join(raw_dir, "sample.csv")
    train_sample = os.path.join(raw_dir, "train_sample.csv")
    if os.path.exists(sample):
        df = pd.read_csv(sample)
    else:
        parts = []
        for name in ("train_sample.csv", "validation_sample.csv", "test_sample.csv"):
            p = os.path.join(raw_dir, name)
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        if not parts:
            raise FileNotFoundError(
                f"No raw check-in CSVs found in {raw_dir}. Expected sample.csv "
                f"(all splits, with a SplitTag column) or train_sample.csv "
                f"(HF w11wo/LLM4POI layout). Download the dataset first."
            )
        df = pd.concat(parts, ignore_index=True)
    # HF layout has no SplitTag -> the file is entirely the train split.
    if "SplitTag" not in df.columns:
        df = df.copy()
        df["SplitTag"] = "train"
    train_df = pd.read_csv(train_sample) if os.path.exists(train_sample) \
        else df[df["SplitTag"] == "train"].copy()
    return df, train_df


def stage_sequences_and_poi_info(raw_dir, out_dir):
    seq_paths = {s: os.path.join(out_dir, f"{s}_poi_sequence.csv")
                 for s in ("train", "validation", "test")}
    poi_info_path = os.path.join(out_dir, "poi_info.csv")
    if _exists(poi_info_path, *seq_paths.values()):
        print("[build] stage 1 (sequences + poi_info): outputs present, skipping.")
        return seq_paths, poi_info_path

    print("[build] stage 1: building trajectory sequences + poi_info")
    df, train_df = _load_raw(raw_dir)
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"])
    df = df.sort_values(["UserId", "UTCTimeOffset"]).reset_index(drop=True)

    # keep only users/POIs seen in training (cold-start removal, per the notebook)
    train_users = set(df[df["SplitTag"] == "train"]["UserId"].unique())
    train_pois = set(df[df["SplitTag"] == "train"]["PoiId"].unique())
    df = df[df["UserId"].isin(train_users) & df["PoiId"].isin(train_pois)].copy()
    df = df.sort_values(["UserId", "UTCTimeOffset"]).reset_index(drop=True)

    records = []
    for _, user_df in df.groupby("UserId", sort=False):
        user_df = user_df.sort_values("UTCTimeOffset").reset_index(drop=True)
        for (traj_id, _), traj_df in user_df.groupby(
                ["pseudo_session_trajectory_id", "SplitTag"], sort=False):
            traj_df = traj_df.sort_values("UTCTimeOffset").reset_index(drop=True)
            split_tag = traj_df["SplitTag"].iloc[0]
            start_time = traj_df["UTCTimeOffset"].iloc[0]
            history_df = user_df[user_df["UTCTimeOffset"] < start_time]
            merged = pd.concat([history_df, traj_df], axis=0).sort_values("UTCTimeOffset")
            merged = merged.iloc[-50:].reset_index(drop=True) if len(merged) > 50 \
                else merged.reset_index(drop=True)
            if split_tag == "train" and len(merged) < 20:
                continue
            records.append({
                "UserId": traj_df["UserId"].iloc[0],
                "SplitTag": split_tag,
                "sequence_PoiId": merged["PoiId"].tolist(),
                "sequence_UTCTimeOffset": merged["UTCTimeOffset"].astype(str).tolist(),
            })
    result_df = pd.DataFrame(records)

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    cols = ["UserId", "sequence_PoiId", "sequence_UTCTimeOffset"]
    for tag, fname in split_map.items():
        sub = result_df[result_df["SplitTag"] == tag][cols]
        sub.to_csv(seq_paths[fname], index=False)
        print(f"[build]   {tag}: {len(sub)} sequences -> {seq_paths[fname]}")

    # ---- poi_info (built from the training split) ----
    p = train_df[["PoiId", "PoiCategoryName", "Latitude", "Longitude", "UTCTimeOffset"]].copy()
    p["UTCTimeOffset"] = pd.to_datetime(p["UTCTimeOffset"], errors="coerce")
    p["PoiCategoryName"] = p["PoiCategoryName"].apply(extract_category_name)
    rows = []
    for pid, group in p.groupby("PoiId"):
        group = group.dropna(subset=["UTCTimeOffset"])
        if len(group) == 0:
            continue
        row0 = group.iloc[0]
        hour_counts = group["UTCTimeOffset"].dt.hour.value_counts().to_dict()
        sorted_hours = dict(sorted(((int(h), int(c)) for h, c in hour_counts.items()),
                                   key=lambda kv: kv[1], reverse=True))
        rows.append({
            "pid": pid,
            "category": row0["PoiCategoryName"],
            "latitude": row0["Latitude"],
            "longitude": row0["Longitude"],
            "visit_time_and_count": sorted_hours,
        })
    pd.DataFrame(rows).to_csv(poi_info_path, index=False)
    print(f"[build]   poi_info: {len(rows)} POIs -> {poi_info_path}")
    return seq_paths, poi_info_path


# --------------------------------------------------------------------------- #
# stage 2 -- category text embeddings
# --------------------------------------------------------------------------- #
def stage_category_embeddings(poi_info_path, out_dir, model_name, cat_dim):
    cat_pkl = os.path.join(out_dir, "category_to_embedding.pkl")
    if os.path.exists(cat_pkl):
        print("[build] stage 2 (category embeddings): present, skipping.")
        return cat_pkl

    print(f"[build] stage 2: encoding categories with {model_name}")
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(poi_info_path)
    categories = [c for c in df["category"].dropna().unique().tolist()]
    model = SentenceTransformer(model_name)
    emb = model.encode(categories, show_progress_bar=True)

    n_comp = min(cat_dim, emb.shape[1], len(categories))
    if n_comp < emb.shape[1]:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=n_comp).fit_transform(emb)
        print(f"[build]   PCA -> {emb.shape[1]} dims")
    cat2emb = {c: emb[i] for i, c in enumerate(categories)}
    with open(cat_pkl, "wb") as f:
        pickle.dump(cat2emb, f)
    print(f"[build]   {len(cat2emb)} categories -> {cat_pkl}")
    return cat_pkl


# --------------------------------------------------------------------------- #
# stage 3 -- concatenated POI feature vectors
# --------------------------------------------------------------------------- #
def stage_poi_embeddings(poi_info_path, cat_pkl, out_dir):
    emb_pkl = os.path.join(out_dir, "poi_Emb_dict.pkl")
    if os.path.exists(emb_pkl):
        print("[build] stage 3 (poi feature vectors): present, skipping.")
        return emb_pkl

    print("[build] stage 3: building POI feature vectors")
    df = pd.read_csv(poi_info_path)
    with open(cat_pkl, "rb") as f:
        cat2emb = pickle.load(f)
    cat_dim = len(next(iter(cat2emb.values())))

    emb_dict = {}
    for row in df.itertuples(index=False):
        cat_vec = cat2emb.get(row.category, np.zeros(cat_dim))
        spatial = latlon_to_3d(row.latitude, row.longitude)
        time_vec = extract_time_features2(parse_time_dict(row.visit_time_and_count))
        emb_dict[int(row.pid)] = np.concatenate([cat_vec, spatial, time_vec])
    with open(emb_pkl, "wb") as f:
        pickle.dump(emb_dict, f)
    dim = len(next(iter(emb_dict.values())))
    print(f"[build]   {len(emb_dict)} POIs x {dim}-d -> {emb_pkl}")
    return emb_pkl


# --------------------------------------------------------------------------- #
# stage 4 -- train CRQVAE and emit the pid->sid codebook (V2 module)
# --------------------------------------------------------------------------- #
def _find_best_ckpt(ckpt_dir):
    """Prefer best_collision, then best_loss, else the newest epoch_*.pth."""
    best_coll, best_loss, epochs = None, None, []
    for root, _, files in os.walk(ckpt_dir):
        for fn in files:
            full = os.path.join(root, fn)
            if fn == "best_collision_model.pth":
                best_coll = full
            elif fn == "best_loss_model.pth":
                best_loss = full
            elif fn.startswith("epoch_") and fn.endswith(".pth"):
                epochs.append(full)
    if best_coll:
        return best_coll
    if best_loss:
        return best_loss
    if epochs:
        return max(epochs, key=os.path.getmtime)
    return None


def stage_train_and_emit_sid(emb_pkl, out_dir, ckpt_dir, v2_sid_dir, device, cfg):
    codebook_csv = os.path.join(out_dir, "codebook.csv")
    if os.path.exists(codebook_csv):
        print("[build] stage 4 (CRQVAE + SID): codebook present, skipping.")
        return codebook_csv

    sys.path.insert(0, v2_sid_dir)
    import torch
    from torch.utils.data import DataLoader
    from POIdatasets import EmbDataset
    from CRQVAE.crqvae import CRQVAE
    from SID_trainer import Trainer

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[build]   CUDA not available; falling back to CPU")
        device = "cpu"

    data = EmbDataset(emb_pkl)
    print(f"[build] stage 4: training CRQVAE on {len(data)} POIs ({data.dim}-d), "
          f"{cfg.epochs} epochs on {device}")

    def make_model():
        return CRQVAE(in_dim=data.dim, num_emb_list=cfg.num_emb_list, e_dim=cfg.e_dim,
                      layers=cfg.layers, dropout_prob=cfg.dropout_prob, bn=cfg.bn,
                      loss_type="mse", quant_loss_weight=cfg.quant_loss_weight,
                      beta=cfg.beta, kmeans_init=cfg.kmeans_init, kmeans_iters=cfg.kmeans_iters,
                      sk_epsilons=cfg.sk_epsilons, sk_iters=cfg.sk_iters, use_linear=cfg.use_linear)

    sid_args = SimpleNamespace(
        use_sk=False, lr=cfg.lr, learner="AdamW", lr_scheduler_type="constant",
        weight_decay=cfg.weight_decay, epochs=cfg.epochs, warmup_epochs=cfg.warmup_epochs,
        save_limit=5, eval_step=cfg.eval_step, device=device, ckpt_dir=ckpt_dir,
    )
    train_loader = DataLoader(data, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
                              shuffle=True, pin_memory=True)
    trainer = Trainer(sid_args, make_model(), len(train_loader))
    best_loss, best_collision = trainer.fit(train_loader)
    print(f"[build]   best_loss={best_loss:.4f} best_collision_rate={best_collision:.4f}")

    ckpt = _find_best_ckpt(ckpt_dir)
    if ckpt is None:
        raise RuntimeError(f"No CRQVAE checkpoint produced under {ckpt_dir}")
    print(f"[build]   loading best checkpoint: {ckpt}")
    model = make_model()
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model = model.to(device).eval()

    # emit SIDs (no shuffle so emission is deterministic)
    emit_loader = DataLoader(data, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
                             shuffle=False, pin_memory=True)
    sids, vectors = {}, {}
    for pids, x in emit_loader:
        x = x.to(device)
        vec, indices = model.get_indices(x)
        for i, pid in enumerate(pids.tolist()):
            sids[int(pid)] = indices[i].tolist()
            vectors[int(pid)] = vec[i].tolist()

    # disambiguate collisions by appending an extra atom (-> 4th '<d_..>' level)
    counts = Counter(tuple(v) for v in sids.values())
    seen, final = {}, {}
    for pid in sorted(sids.keys()):
        v = sids[pid]
        tup = tuple(v)
        if counts[tup] > 1:
            seen[tup] = seen.get(tup, -1) + 1
            final[pid] = v + [seen[tup]]
        else:
            final[pid] = v
    n_collision = sum(1 for t, c in counts.items() if c > 1)
    print(f"[build]   {len(final)} SIDs ({n_collision} colliding base codes disambiguated)")

    out = pd.DataFrame({
        "pid": list(final.keys()),
        "sid": [str(v) for v in final.values()],
        "vector": [str(vectors[k]) for k in final.keys()],
    })
    out.to_csv(codebook_csv, index=False)
    print(f"[build]   codebook -> {codebook_csv}")
    return codebook_csv


# --------------------------------------------------------------------------- #
# stage 5 -- assemble the LLM fine-tuning JSON
# --------------------------------------------------------------------------- #
def _load_pid2code(codebook_csv):
    df = pd.read_csv(codebook_csv)
    pid2code = {}
    for row in df.itertuples(index=False):
        try:
            pid2code[int(row.pid)] = encode_item(ast.literal_eval(row.sid))
        except Exception as e:
            print(f"[build]   skip pid={row.pid} sid={row.sid}: {e}")
    return pid2code


def _convert_split(seq_csv, out_json, pid2code, keep_last_k):
    df = pd.read_csv(seq_csv)
    if keep_last_k is not None:
        df = df.groupby("UserId", group_keys=False).tail(keep_last_k).reset_index(drop=True)
    samples, skipped = [], 0
    for row in df.itertuples(index=False):
        try:
            poi_seq = ast.literal_eval(row.sequence_PoiId)
            time_seq = ast.literal_eval(row.sequence_UTCTimeOffset)
        except Exception:
            skipped += 1
            continue
        if len(poi_seq) < 2 or len(poi_seq) != len(time_seq):
            skipped += 1
            continue
        hist_pids, hist_times = poi_seq[:-1], time_seq[:-1]
        target_pid, target_time = poi_seq[-1], time_seq[-1]
        if any(p not in pid2code for p in hist_pids) or target_pid not in pid2code:
            skipped += 1
            continue
        hist = ", ".join(f"{t} visited {pid2code[p]}" for t, p in zip(hist_times, hist_pids))
        samples.append({
            "instruction": INSTRUCTION,
            "input": (f"User_{row.UserId} checkin history: {hist}.\n"
                      f"When {target_time} user_{row.UserId} is likely to visit:"),
            "output": pid2code[target_pid],
        })
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[build]   {os.path.basename(out_json)}: {len(samples)} samples "
          f"({skipped} skipped) -> {out_json}")


# test set in LLM4POI's text-QA form (HF w11wo layout). Each line looks like:
#   <question>: ...At <t>, user <U> visited POI id <P>...  <answer>: At <t>, user <U> will visit POI id <T>.
_QA_USER_RE = re.compile(r"trajectory of user (\d+)")
_QA_VISIT_RE = re.compile(r"At (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d), user \d+ visited POI id (\d+)")
_QA_ANSWER_RE = re.compile(r"<answer>:.*?At (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?POI id (\d+)", re.S)


def _convert_qa_test(qa_txt, out_json, pid2code, max_hist):
    samples, skipped = [], 0
    with open(qa_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<answer>" not in line:
                continue
            q_part, a_part = line.split("<answer>", 1)
            um = _QA_USER_RE.search(q_part)
            am = _QA_ANSWER_RE.search("<answer>" + a_part)
            if not um or not am:
                skipped += 1
                continue
            uid = int(um.group(1))
            target_time, target_pid = am.group(1), int(am.group(2))
            # all past visits, chronological, last `max_hist` (mirrors the train recipe)
            visits = [(t, int(p)) for t, p in _QA_VISIT_RE.findall(q_part)]
            visits.sort(key=lambda tp: tp[0])
            if max_hist:
                visits = visits[-max_hist:]
            if len(visits) < 1 or target_pid not in pid2code \
                    or any(p not in pid2code for _, p in visits):
                skipped += 1
                continue
            hist = ", ".join(f"{t} visited {pid2code[p]}" for t, p in visits)
            samples.append({
                "instruction": INSTRUCTION,
                "input": (f"User_{uid} checkin history: {hist}.\n"
                          f"When {target_time} user_{uid} is likely to visit:"),
                "output": pid2code[target_pid],
            })
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[build]   llm_test.json: {len(samples)} samples ({skipped} skipped, "
          f"from QA text) -> {out_json}")


def _nonempty_csv(path):
    return os.path.exists(path) and len(pd.read_csv(path)) > 0


def stage_llm_json(seq_paths, codebook_csv, out_dir, keep_last_k, test_qa=None, max_hist=50):
    outs = {s: os.path.join(out_dir, f"llm_{s}.json")
            for s in ("train", "val", "test")}
    if _exists(*outs.values()):
        print("[build] stage 5 (llm json): outputs present, skipping.")
        return outs
    print("[build] stage 5: assembling llm_*.json")
    pid2code = _load_pid2code(codebook_csv)

    # train (always)
    if not os.path.exists(outs["train"]):
        _convert_split(seq_paths["train"], outs["train"], pid2code, keep_last_k)

    # validation (optional -- HF w11wo has no val split)
    if not os.path.exists(outs["val"]):
        if _nonempty_csv(seq_paths["validation"]):
            _convert_split(seq_paths["validation"], outs["val"], pid2code, None)
        else:
            print("[build]   no validation split available; skipping llm_val.json")

    # test (from QA text if provided, else from a SplitTag-based test CSV)
    if not os.path.exists(outs["test"]):
        if test_qa and os.path.exists(test_qa):
            _convert_qa_test(test_qa, outs["test"], pid2code, max_hist)
        elif _nonempty_csv(seq_paths["test"]):
            _convert_split(seq_paths["test"], outs["test"], pid2code, None)
        else:
            print("[build]   WARNING: no test source (no QA text, no test CSV); "
                  "llm_test.json NOT written.")
    return outs


# --------------------------------------------------------------------------- #
def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description="Build llm_*.json via the V2 CRQVAE SID module")
    p.add_argument("--dataset", default=env("DATASET", "tky"))
    p.add_argument("--raw_dir", required=True, help="dir with raw sample.csv / train_sample.csv")
    p.add_argument("--out_dir", required=True, help="DATA_DIR (where llm_*.json land)")
    p.add_argument("--test_qa", default="", help="LLM4POI test_qa_pairs_kqt.txt (HF layout); "
                   "builds llm_test.json from it. Default: <raw_dir>/test_qa_pairs_kqt.txt if present")
    p.add_argument("--max_hist", type=int, default=int(os.environ.get("MAX_HIST", "50")),
                   help="cap history length per sample (matches the 50-check-in recipe)")
    p.add_argument("--ckpt_dir", required=True, help="CRQVAE checkpoint dir (SID_DIR)")
    p.add_argument("--v2_sid_dir", default=os.path.join(REPO_ROOT, "V2", "SID"))
    p.add_argument("--device", default=env("SID_DEVICE", "cuda:0"))
    # SID / CRQVAE knobs (defaults mirror V2/SID/train_SID.py)
    p.add_argument("--cat_model", default=env("CAT_MODEL", "all-MiniLM-L6-v2"))
    p.add_argument("--cat_dim", type=int, default=int(env("CAT_DIM", "64")))
    p.add_argument("--keep_last_k", type=int, default=int(env("KEEP_LAST_K", "5")))
    p.add_argument("--epochs", type=int, default=int(env("SID_EPOCHS", "3000")))
    p.add_argument("--num_emb_list", type=int, nargs="+",
                   default=[int(x) for x in env("SID_NUM_EMB", "64 64 64").split()])
    p.add_argument("--e_dim", type=int, default=int(env("SID_E_DIM", "64")))
    p.add_argument("--layers", type=int, nargs="+",
                   default=[int(x) for x in env("SID_LAYERS", "512 256 128").split()])
    p.add_argument("--batch_size", type=int, default=int(env("SID_BATCH_SIZE", "128")))
    p.add_argument("--num_workers", type=int, default=int(env("SID_NUM_WORKERS", "4")))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # final outputs already there? then there's nothing to do. (val is optional --
    # the HF w11wo layout has no validation split, so we only require train+test.)
    finals = [os.path.join(args.out_dir, f"llm_{s}.json") for s in ("train", "test")]
    if _exists(*finals):
        print(f"[build] llm_train/test.json already present in {args.out_dir}; nothing to do.")
        return 0

    # default test source: the HF QA text dropped next to the raw CSV.
    test_qa = args.test_qa or os.path.join(args.raw_dir, "test_qa_pairs_kqt.txt")
    test_qa = test_qa if os.path.exists(test_qa) else None

    cfg = SimpleNamespace(
        epochs=args.epochs, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        layers=args.layers, dropout_prob=0.1, bn=True, quant_loss_weight=0.5, beta=0.25,
        kmeans_init=True, kmeans_iters=100, sk_epsilons=[0.1, 0.1, 0.1], sk_iters=50,
        use_linear=1, lr=1e-3, weight_decay=1e-4, warmup_epochs=100, eval_step=10,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    seq_paths, poi_info = stage_sequences_and_poi_info(args.raw_dir, args.out_dir)
    cat_pkl = stage_category_embeddings(poi_info, args.out_dir, args.cat_model, args.cat_dim)
    emb_pkl = stage_poi_embeddings(poi_info, cat_pkl, args.out_dir)
    codebook = stage_train_and_emit_sid(emb_pkl, args.out_dir, args.ckpt_dir,
                                        args.v2_sid_dir, args.device, cfg)
    stage_llm_json(seq_paths, codebook, args.out_dir, args.keep_last_k,
                   test_qa=test_qa, max_hist=args.max_hist)
    print(f"[build] DONE -> {args.out_dir}/llm_{{train,test}}.json"
          + ("" if test_qa else " (test built from CSV split if present)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
