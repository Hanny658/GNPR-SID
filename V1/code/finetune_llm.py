#!/usr/bin/env python
"""
LLM fine-tuning for generative next-POI recommendation with Semantic IDs (V1).

This is a self-contained Hugging Face ``Trainer`` driver that reproduces the
recipe used by the authors' reference SFT code (``V2/LLM/train/sft_without_alignment.py``):
LoRA, bf16, gradient-checkpointing, Alpaca-style prompt, and *completion-only*
supervision (the prompt tokens are masked out of the loss).

It is built for a single GPU within a ~6h wall-clock budget and is fully
resumable: checkpoints are written every ``--save_steps`` (default 200) with a
sliding window of ``--save_total_limit`` (default 5), and ``trainer.train``
auto-resumes from the newest checkpoint in ``--output_dir``.  Re-running the
job after it has finished is a no-op for the caller (the Slurm wrapper checks a
done-marker), but even invoking this script again will simply resume/no-op from
the last checkpoint.

Example (paths are normally supplied by scripts/train.slurm):

    python finetune_llm.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --train_file ../datasets/nyc/llm_train.json \
        --output_dir ../runs/nyc_.../sft \
        --save_steps 200 --save_total_limit 5
"""

import argparse
import json
import os
import re
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Alpaca prompt; the model is supervised only on what follows "### Response:\n".
PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

SID_RE = re.compile(r"<([abcd])_(\d+)>")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a JSON list of {{instruction,input,output}}")
    return items


def build_prompt(ex):
    return PROMPT_TEMPLATE.format(
        instruction=str(ex.get("instruction", "")).strip(),
        input=str(ex.get("input", "")).strip(),
    )


def collect_sid_tokens(*item_lists):
    """All distinct '<a_..>/<b_..>/<c_..>/<d_..>' atoms appearing in the data."""
    toks = set()
    for items in item_lists:
        for ex in items:
            for field in ("input", "output"):
                for m in SID_RE.finditer(str(ex.get(field, ""))):
                    toks.add((m.group(1), int(m.group(2))))
    return [f"<{l}_{n}>" for (l, n) in sorted(toks)]


class SFTDataset(Dataset):
    """Tokenizes on the fly; masks the prompt, supervises response + eos."""

    def __init__(self, items, tokenizer, max_len):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ex = self.items[i]
        eos = self.tok.eos_token or ""
        prompt_ids = self.tok(build_prompt(ex), add_special_tokens=False)["input_ids"]
        resp_ids = self.tok(str(ex.get("output", "")).strip() + eos,
                            add_special_tokens=False)["input_ids"]

        # Keep the (short) response intact; left-truncate the (long) history.
        max_prompt = self.max_len - len(resp_ids)
        if max_prompt < 1:
            resp_ids = resp_ids[: self.max_len - 1]
            max_prompt = 1
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + list(resp_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            d = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * d)
            labels.append(f["labels"] + [-100] * d)
            attn.append(f["attention_mask"] + [0] * d)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# training-args builder (robust to the eval_strategy/evaluation_strategy rename)
# --------------------------------------------------------------------------- #
def make_training_args(**kw):
    try:
        return TrainingArguments(**kw)
    except TypeError:
        if "eval_strategy" in kw:
            kw["evaluation_strategy"] = kw.pop("eval_strategy")
        return TrainingArguments(**kw)


def parse_args():
    p = argparse.ArgumentParser(description="GNPR-SID V1 LLM fine-tuning")
    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tuning", choices=["lora", "full"], default=os.environ.get("TUNING", "lora"))
    p.add_argument("--add_sid_tokens", type=int, default=int(os.environ.get("ADD_SID_TOKENS", "0")))
    p.add_argument("--num_epochs", type=float, default=float(os.environ.get("NUM_EPOCHS", "3")))
    p.add_argument("--per_device_bs", type=int, default=int(os.environ.get("PER_DEVICE_BS", "4")))
    p.add_argument("--grad_accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "4")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "2e-5")))
    p.add_argument("--max_seq_len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "2048")))
    p.add_argument("--save_steps", type=int, default=int(os.environ.get("SAVE_STEPS", "200")))
    p.add_argument("--save_total_limit", type=int, default=int(os.environ.get("SAVE_TOTAL_LIMIT", "5")))
    p.add_argument("--eval_during_train", type=int, default=int(os.environ.get("EVAL_DURING_TRAIN", "0")))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "final")

    # If a previous job already produced the final adapter, this is a no-op.
    if os.path.exists(os.path.join(final_dir, "_FINETUNE_COMPLETE")):
        print(f"[finetune] '{final_dir}' already complete; nothing to do.")
        return

    print(f"[finetune] base_model={args.base_model} tuning={args.tuning} "
          f"add_sid_tokens={args.add_sid_tokens}")
    train_items = load_items(args.train_file)
    val_items = load_items(args.val_file) if (args.val_file and os.path.exists(args.val_file)) else None
    print(f"[finetune] train={len(train_items)} val={len(val_items) if val_items else 0}")

    # ---------------- tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sid_tokens = []
    if args.add_sid_tokens:
        sid_tokens = collect_sid_tokens(train_items, val_items or [])
        added = tokenizer.add_special_tokens({"additional_special_tokens": sid_tokens})
        print(f"[finetune] added {added} atomic SID tokens (vocab now {len(tokenizer)})")

    # ---------------- model ----------------
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True
    )
    if args.add_sid_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False  # required with gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.tuning == "lora":
        from peft import LoraConfig, get_peft_model

        modules_to_save = ["embed_tokens", "lm_head"] if args.add_sid_tokens else None
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_cfg)
        model.enable_input_require_grads()  # so grad-checkpointing works with frozen base
        model.print_trainable_parameters()

    # ---------------- datasets ----------------
    train_ds = SFTDataset(train_items, tokenizer, args.max_seq_len)
    eval_ds = (SFTDataset(val_items, tokenizer, args.max_seq_len)
               if (val_items and args.eval_during_train) else None)
    collator = PadCollator(tokenizer.pad_token_id)

    # ---------------- trainer ----------------
    targs = make_training_args(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,     # <- sliding window of N checkpoints
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=args.save_steps,
        bf16=bf16_ok,
        fp16=(not bf16_ok),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        report_to="none",
        save_safetensors=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Resume from the newest checkpoint if one exists (handles 6h re-submits).
    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        print(f"[finetune] resuming from {last_ckpt}")
    else:
        print("[finetune] starting from scratch (no checkpoint found)")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # ---------------- persist final artifacts ----------------
    trainer.save_model(final_dir)        # LoRA adapter (or full model)
    tokenizer.save_pretrained(final_dir)
    meta = {
        "base_model": args.base_model,
        "tuning": args.tuning,
        "add_sid_tokens": bool(args.add_sid_tokens),
        "num_sid_tokens": len(sid_tokens),
        "max_seq_len": args.max_seq_len,
    }
    with open(os.path.join(final_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(final_dir, "_FINETUNE_COMPLETE"), "w").close()
    print(f"[finetune] done -> {final_dir}")


if __name__ == "__main__":
    sys.exit(main())
