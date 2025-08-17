
from __future__ import annotations
import os, argparse, json, math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import yaml

def load_config(p):
    with open(p,"r") as f: return yaml.safe_load(f)

def format_example(ex):
    # Expect fields: "question", "contexts" (list of {text,meta}), "answer", "references" (list)
    ctx_lines = []
    for i, c in enumerate(ex.get("contexts", []), start=1):
        src = c.get("meta", {}).get("pmid") or c.get("meta", {}).get("source") or c.get("meta", {}).get("url","unknown")
        ctx_lines.append(f"[{i}] {c['text']}\n(Source: {src})")
    ctx = "\n\n".join(ctx_lines) if ctx_lines else "No context."
    prompt = f"You are a careful medical assistant. Use the CONTEXT to answer the QUESTION with inline numeric citations and a final References block.\n\nQUESTION:\n{ex['question']}\n\nCONTEXT:\n{ctx}\n\nAnswer:\n"
    target = ex["answer"]
    return prompt + target

def make_dataset(data_path: str):
    # data_path can be a json or jsonl
    if data_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=data_path, split="train")
    else:
        ds = load_dataset("json", data_files={"train": data_path})["train"]
    return ds

def tokenize(tokenizer, text, max_len):
    return tokenizer(text, truncation=True, max_length=max_len, return_tensors=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    base_model = cfg["base_model"]
    data_path = cfg["data_path"]
    output_dir = cfg["output_dir"]
    max_len = int(cfg.get("max_seq_len", 2048))

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg, device_map="auto")
    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("lora_target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = make_dataset(data_path)
    ds = ds.map(lambda ex: {"text": format_example(ex)})
    tokenized = ds.map(lambda ex: tokenize(tokenizer, ex["text"], max_len), batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_tr = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        learning_rate=cfg.get("lr", 2e-4),
        num_train_epochs=cfg.get("epochs", 2),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args_tr, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Finished training LoRA adapters at", output_dir)

if __name__ == "__main__":
    main()
