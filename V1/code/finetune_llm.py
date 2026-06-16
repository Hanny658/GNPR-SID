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
    p.add_argument("--tuning", choices=["lora", "full", "embed"],
                   default=os.environ.get("TUNING", "lora"))
    p.add_argument("--add_sid_tokens", type=int, default=int(os.environ.get("ADD_SID_TOKENS", "0")))
    p.add_argument("--num_epochs", type=float, default=float(os.environ.get("NUM_EPOCHS", "8")))
    p.add_argument("--per_device_bs", type=int, default=int(os.environ.get("PER_DEVICE_BS", "2")))
    p.add_argument("--grad_accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "32")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-5")))
    p.add_argument("--lr_scheduler", default=os.environ.get("LR_SCHEDULER", "constant_with_warmup"))
    p.add_argument("--warmup_steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "20")))
    p.add_argument("--lora_r", type=int, default=int(os.environ.get("LORA_R", "16")))
    p.add_argument("--lora_alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    p.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.1")))
    # Comma/space-separated module names. The V2 pipeline overrides this: the
    # alignment stage trains "embed_tokens" only; its SFT drops o_proj/down_proj.
    p.add_argument("--lora_targets",
                   default=os.environ.get("LORA_TARGETS",
                                          "q_proj,k_proj,v_proj,o_proj,"
                                          "gate_proj,up_proj,down_proj"))
    p.add_argument("--max_seq_len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "2048")))
    p.add_argument("--save_steps", type=int, default=int(os.environ.get("SAVE_STEPS", "50")))
    p.add_argument("--save_total_limit", type=int, default=int(os.environ.get("SAVE_TOTAL_LIMIT", "5")))
    p.add_argument("--eval_during_train", type=int, default=int(os.environ.get("EVAL_DURING_TRAIN", "0")))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_resume_checkpoint(output_dir):
    """Latest *complete* checkpoint to resume from.

    The HF Trainer writes ``trainer_state.json`` last (after the model weights,
    optimizer and scheduler) and only deletes older checkpoints AFTER the new
    one is fully written. So a job killed mid-save leaves a half-written newest
    ``checkpoint-<step>`` (no ``trainer_state.json``) while the previous, complete
    checkpoint still exists. ``get_last_checkpoint`` would pick that half-written
    dir by step number and the resume would crash, so we instead take the newest
    checkpoint that actually has ``trainer_state.json``. This makes resume robust
    even with ``--save_total_limit 1``.
    """
    if not os.path.isdir(output_dir):
        return None
    def _complete(path):
        ts = os.path.join(path, "trainer_state.json")
        if not os.path.exists(ts):
            return False
        try:  # also reject a trainer_state.json truncated mid-write
            with open(ts, encoding="utf-8") as f:
                json.load(f)
            return True
        except Exception:
            return False

    complete, incomplete = [], []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        step, path = int(m.group(1)), os.path.join(output_dir, name)
        (complete if _complete(path) else incomplete).append((step, path))
    if not complete:
        return None
    best_step, best = max(complete, key=lambda sp: sp[0])
    for step, path in incomplete:
        if step > best_step:
            print(f"[finetune] WARNING: ignoring incomplete checkpoint {path} "
                  f"(no trainer_state.json; likely killed mid-save) and resuming "
                  f"from the last complete one instead")
    return best


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

    if args.tuning == "embed":
        # Train ONLY the token embeddings (and the untied output head). This is
        # the V2 SID-alignment stage: the freshly added SID tokens are OOV, so we
        # learn their embeddings on the attribute<->SID data BEFORE the SFT.
        # Deliberately NOT PEFT, because (a) the paper trains "only the embedding
        # module", and (b) LoRA-on-embed_tokens collides with modules_to_save and
        # is fragile to merge on tie_word_embeddings=True models. The result is a
        # full model whose embeddings carry the alignment -- the SFT stage then
        # uses it directly as its base, so no separate merge step is needed.
        if not args.add_sid_tokens:
            print("[finetune] WARNING: --tuning embed without --add_sid_tokens=1 "
                  "trains the entire embedding matrix (no new SID tokens to learn).")
        for p in model.parameters():
            p.requires_grad = False
        in_emb = model.get_input_embeddings()
        in_emb.weight.requires_grad = True
        out_emb = model.get_output_embeddings()
        tied = out_emb is None or out_emb.weight is in_emb.weight
        if not tied:
            out_emb.weight.requires_grad = True
        model.enable_input_require_grads()  # keep grad-checkpointing happy
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[finetune] embed-only: training {n_train:,}/{n_all:,} params "
              f"({'tied' if tied else 'untied'} lm_head)")

    elif args.tuning == "lora":
        from peft import LoraConfig, get_peft_model

        target_modules = [t for t in re.split(r"[,\s]+", args.lora_targets.strip()) if t]
        # don't mark a module both a LoRA target and to-save (PEFT excludes
        # modules_to_save from targeting -> "no modules targeted" if they overlap)
        modules_to_save = (["embed_tokens", "lm_head"] if args.add_sid_tokens else None)
        if modules_to_save:
            modules_to_save = [m for m in modules_to_save if m not in target_modules]
            modules_to_save = modules_to_save or None
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
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
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
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

    # Resume from the newest *complete* checkpoint if one exists (handles 6h
    # re-submits, and skips a half-written checkpoint left by a mid-save kill).
    last_ckpt = find_resume_checkpoint(args.output_dir)
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
