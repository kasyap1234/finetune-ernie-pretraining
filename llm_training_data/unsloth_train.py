#!/usr/bin/env python3
"""
Unsloth Continued Pretraining Script for Indian Stock Market Dataset

This script performs continued pretraining on Qwen 3 32B using QLoRA
to teach the model domain-specific knowledge about Indian stock markets.

Based on Unsloth documentation best practices for continued pretraining.

Requirements:
    pip install unsloth datasets torch accelerate bitsandbytes

Usage:
    python unsloth_train.py                     # Full training
    python unsloth_train.py --max_steps 10      # Quick test run
    python unsloth_train.py --resume            # Resume from checkpoint
"""

import argparse
import json
import os
import torch
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "unsloth/Qwen3-32B-unsloth-bnb-4bit"  # Qwen 3 32B with 4-bit quantization
MAX_SEQ_LENGTH = 4096
DTYPE = None  # Auto-detect (will use bf16 on A100/H100)
LOAD_IN_4BIT = True

# LoRA Configuration for Continued Pretraining
LORA_R = 16  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha (typically same as r)
LORA_DROPOUT = 0  # Dropout for LoRA layers

# Training hyperparameters
LEARNING_RATE = 5e-5
EMBEDDING_LEARNING_RATE = 5e-6  # 10x smaller for lm_head and embed_tokens
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 2  # Adjust based on GPU (2 for 80GB, 1 for 48GB)
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 16
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen 3 32B Continued Pretraining")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 for full training)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./india_stock_market_model",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Per-device batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Qwen 3 32B Continued Pretraining for Indian Stock Market")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max steps: {'Full training' if args.max_steps == -1 else args.max_steps}")
    print("=" * 60)
    
    # ========================================================================
    # Load Model with Unsloth Optimizations
    # ========================================================================
    print("\n[1/4] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # ========================================================================
    # Configure LoRA for Continued Pretraining
    # ========================================================================
    # For continued pretraining, we include lm_head and embed_tokens
    # to better adapt the model to new domain vocabulary
    print("\n[2/4] Configuring LoRA for continued pretraining...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head", "embed_tokens",  # Important for continued pretraining!
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=42,
    )
    
    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("\n[3/4] Loading dataset...")
    
    # Get the project directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Parent of llm_training_data
    
    # Primary training files
    train_file = os.path.join(script_dir, "unsloth_dataset_train.jsonl")
    val_file = os.path.join(script_dir, "unsloth_dataset_val.jsonl")
    
    # Load primary dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
        }
    )
    
    print(f"  Primary training samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")
    
    # ========================================================================
    # Load Additional Data from output/ folder
    # ========================================================================
    output_dir = os.path.join(project_dir, "output")
    output_files = [
        "fundamentals.jsonl",
        "price_history.jsonl",
        "macroeconomic.jsonl",
        "transcripts.jsonl",
        "extended_price_history.jsonl",
        "announcements.jsonl",
        "historical_financials.jsonl",
        "bse_filings.jsonl",
        "news.jsonl",
        "nse_announcements.jsonl",
        "mutual_fund_holdings.jsonl",
        "analyst_reports.jsonl",
    ]
    
    def convert_to_text(record: dict) -> str:
        """Convert structured record to text format for CPT."""
        parts = []
        
        # Common fields to include
        if "symbol" in record:
            parts.append(f"Company: {record['symbol']}")
        if "companyName" in record and record["companyName"] != record.get("symbol"):
            parts.append(f"Company Name: {record['companyName']}")
        if "date" in record and record["date"]:
            parts.append(f"Date: {record['date']}")
        if "category" in record:
            parts.append(f"Category: {record['category']}")
        if "title" in record:
            parts.append(f"Title: {record['title']}")
        if "summary" in record and record["summary"]:
            parts.append(f"Summary: {record['summary']}")
        if "content" in record and record["content"]:
            parts.append(f"\n{record['content']}")
        
        # If record already has 'text' field, use it
        if "text" in record:
            return record["text"]
        
        # Otherwise, join all parts
        return "\n".join(parts) if parts else str(record)
    
    # Load and convert output files
    additional_samples = []
    for filename in output_files:
        # Check for enriched version first
        enriched_filename = filename.replace(".jsonl", "_enriched.jsonl")
        enriched_filepath = os.path.join(output_dir, enriched_filename)
        
        target_filepath = os.path.join(output_dir, filename)
        target_filename = filename
        
        if os.path.exists(enriched_filepath):
            print(f"  Found enriched file: {enriched_filename}")
            target_filepath = enriched_filepath
            target_filename = enriched_filename
        
        if os.path.exists(target_filepath):
            try:
                with open(target_filepath, "r", encoding="utf-8") as f:
                    file_samples = 0
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                text = convert_to_text(record)
                                if text and len(text) > 50:  # Skip very short records
                                    additional_samples.append({"text": text})
                                    file_samples += 1
                            except json.JSONDecodeError:
                                continue
                print(f"  Loaded {target_filename}: {file_samples:,} records")
            except Exception as e:
                print(f"  Warning: Failed to load {target_filename}: {e}")
    
    # Combine with primary dataset
    if additional_samples:
        from datasets import Dataset, concatenate_datasets
        additional_dataset = Dataset.from_list(additional_samples)
        dataset["train"] = concatenate_datasets([dataset["train"], additional_dataset])
        print(f"  Combined training samples: {len(dataset['train']):,}")
    
    # ========================================================================
    # Configure Training
    # ========================================================================
    print("\n[4/4] Configuring trainer...")
    
    # Use UnslothTrainingArguments for embedding_learning_rate support
    training_args = UnslothTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=args.max_steps,  # -1 means use num_train_epochs
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LEARNING_RATE,
        embedding_learning_rate=EMBEDDING_LEARNING_RATE,  # Smaller LR for embeddings
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",  # Set to "wandb" for Weights & Biases logging
    )
    
    # Use UnslothTrainer for continued pretraining
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # ========================================================================
    # Start Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Starting continued pretraining...")
    print("=" * 60 + "\n")
    
    if args.resume:
        print("Resuming from last checkpoint...")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        trainer_stats = trainer.train()
    
    # ========================================================================
    # Save Model
    # ========================================================================
    final_model_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving model to {final_model_dir}...")
    
    # Save LoRA adapters
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Final model: {final_model_dir}")
    print(f"  Training time: {trainer_stats.metrics.get('train_runtime', 0):.2f}s")
    print(f"  Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    print("=" * 60)
    
    # Optional: Save merged 16-bit model for vLLM deployment
    # Uncomment if needed:
    # model.save_pretrained_merged(
    #     os.path.join(args.output_dir, "merged_16bit"),
    #     tokenizer,
    #     save_method="merged_16bit",
    # )


if __name__ == "__main__":
    main()
