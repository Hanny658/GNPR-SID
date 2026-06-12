#!/usr/bin/env python
"""
Merge a LoRA adapter into its base model and save the result as a full model.

Used by the V2 pipeline between the SID-alignment stage and the SFT stage
(the V2 recipe fine-tunes *on top of* the alignment-merged model). Clean
replacement for ``V2/LLM/train/merge_model.py``, which pins CUDA device 1 and
has its paths blanked out. Runs on CPU so it works regardless of GPU memory.

Idempotent: a completed merge leaves ``_MERGE_COMPLETE`` in --out_dir and the
script no-ops on re-run.
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    done_mark = os.path.join(args.out_dir, "_MERGE_COMPLETE")
    if os.path.exists(done_mark):
        print(f"[merge] {args.out_dir} already merged; nothing to do.")
        return
    if not os.path.exists(os.path.join(args.adapter_dir, "adapter_config.json")):
        print(f"[merge] ERROR: no adapter_config.json in {args.adapter_dir}", file=sys.stderr)
        sys.exit(1)

    # Tokenizer travels with the adapter (finetune_llm.py saves it there); it may
    # be larger than the base vocab when SID tokens were added.
    tok_src = args.adapter_dir if os.path.exists(
        os.path.join(args.adapter_dir, "tokenizer_config.json")) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    print(f"[merge] loading base {args.base_model} (cpu, bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    print(f"[merge] applying adapter {args.adapter_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()

    print(f"[merge] saving merged model -> {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)
    open(done_mark, "w").close()
    print("[merge] done.")


if __name__ == "__main__":
    main()
