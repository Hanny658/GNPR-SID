#!/usr/bin/env python
"""
Evaluation for generative next-POI recommendation with Semantic IDs (V1).

Loads the fine-tuned model (LoRA adapter or full model) produced by
``finetune_llm.py`` and, for every test example, generates the top-``num_beams``
Semantic-ID continuations with beam search.  The generated SID strings are
compared against the gold SID to compute the standard generative-recommendation
metrics: Acc@k, MRR and NDCG@k.

The job is resumable: predictions are streamed to ``predictions.jsonl`` (one
line per test example).  If the 6h job is killed and re-submitted, evaluation
skips the examples already present in that file and continues.  Metrics are
recomputed from the full file at the end, so they are always consistent.
"""

import argparse
import json
import math
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# First contiguous run of SID atoms, e.g. "<a_50><b_15><c_62>" (optionally <d_*>).
SID_RUN_RE = re.compile(r"(?:<[abcd]_\d+>)+")


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(ex):
    return PROMPT_TEMPLATE.format(
        instruction=str(ex.get("instruction", "")).strip(),
        input=str(ex.get("input", "")).strip(),
    )


def normalize_sid(text):
    """Extract the first contiguous '<a_..><b_..>...' SID string from raw text."""
    m = SID_RUN_RE.search(text)
    return m.group(0) if m else ""


def parse_args():
    p = argparse.ArgumentParser(description="GNPR-SID V1 evaluation")
    p.add_argument("--model_dir", required=True, help="finetune output_dir/final")
    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL", ""),
                   help="override base model (else read from training_meta.json)")
    p.add_argument("--test_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_beams", type=int, default=int(os.environ.get("NUM_BEAMS", "10")))
    p.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("EVAL_MAX_NEW_TOKENS", "12")))
    p.add_argument("--max_prompt_len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "2048")))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "4")))
    p.add_argument("--ks", type=int, nargs="+",
                   default=[int(k) for k in os.environ.get("EVAL_KS", "1 5 10").split()])
    p.add_argument("--flush_every", type=int, default=int(os.environ.get("EVAL_FLUSH_EVERY", "200")))
    return p.parse_args()


def load_model(model_dir, base_override):
    """Load tokenizer + model, supporting both LoRA-adapter and full-model dirs."""
    meta = {}
    meta_path = os.path.join(model_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16

    is_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    if is_adapter:
        base_model = base_override or meta.get("base_model")
        if not base_model:
            raise ValueError("LoRA adapter found but base model unknown; pass --base_model.")
        print(f"[eval] loading base {base_model} + LoRA adapter {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype,
                                                     trust_remote_code=True)
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        print(f"[eval] loading full model {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype,
                                                     trust_remote_code=True)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    model.config.use_cache = True
    return tokenizer, model


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, num_beams, max_new_tokens, max_prompt_len):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_prompt_len, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]          # strip the prompt
    texts = tokenizer.batch_decode(gen, skip_special_tokens=False)
    # regroup into [batch][num_beams]
    preds = []
    for b in range(len(prompts)):
        cand = []
        for r in range(num_beams):
            sid = normalize_sid(texts[b * num_beams + r])
            if sid and sid not in cand:        # ranked, de-duplicated
                cand.append(sid)
        preds.append(cand)
    return preds


def run_generation(args, tokenizer, model, items, pred_path):
    done = 0
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
    if done >= len(items):
        print(f"[eval] all {len(items)} predictions already present; skipping generation.")
        return
    print(f"[eval] resuming generation at example {done}/{len(items)}")

    fout = open(pred_path, "a", encoding="utf-8")
    try:
        i = done
        n = len(items)
        while i < n:
            batch = items[i: i + args.batch_size]
            prompts = [build_prompt(ex) for ex in batch]
            preds = generate_batch(model, tokenizer, prompts, args.num_beams,
                                   args.max_new_tokens, args.max_prompt_len)
            for j, ex in enumerate(batch):
                rec = {
                    "idx": i + j,
                    "gold": normalize_sid(str(ex.get("output", "")).strip()),
                    "preds": preds[j],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            i += len(batch)
            if (i // max(1, args.batch_size)) % max(1, args.flush_every // max(1, args.batch_size)) == 0 \
                    or i >= n:
                fout.flush()
                os.fsync(fout.fileno())
                print(f"[eval]   {i}/{n} done")
    finally:
        fout.flush()
        fout.close()


def compute_metrics(pred_path, ks):
    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    maxk = max(ks)
    acc = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = 0.0
    for r in records:
        gold, preds = r["gold"], r["preds"]
        rank = None
        for idx, p in enumerate(preds[:maxk], start=1):
            if p == gold:
                rank = idx
                break
        if rank is not None:
            mrr += 1.0 / rank
            for k in ks:
                if rank <= k:
                    acc[k] += 1
                    ndcg[k] += 1.0 / math.log2(rank + 1)
    metrics = {"n": n}
    for k in ks:
        metrics[f"Acc@{k}"] = acc[k] / n if n else 0.0
        metrics[f"NDCG@{k}"] = ndcg[k] / n if n else 0.0
    metrics["MRR"] = mrr / n if n else 0.0
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, "predictions.jsonl")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    items = load_items(args.test_file)
    print(f"[eval] test examples: {len(items)}")

    if args.num_beams < max(args.ks):
        print(f"[eval] WARNING: num_beams={args.num_beams} < max k={max(args.ks)}; "
              f"Acc@{max(args.ks)} will be capped.")

    tokenizer, model = load_model(args.model_dir, args.base_model)
    run_generation(args, tokenizer, model, items, pred_path)

    metrics = compute_metrics(pred_path, args.ks)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    open(os.path.join(args.out_dir, "_EVAL_COMPLETE"), "w").close()

    print("[eval] ===================== metrics =====================")
    for key, val in metrics.items():
        print(f"[eval]   {key:10s}: {val}")
    print(f"[eval] written -> {metrics_path}")


if __name__ == "__main__":
    sys.exit(main())
