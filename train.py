"""
Training script for Hive Personality Model
Uses Unsloth for fast and memory-efficient fine-tuning with LoRA/QLoRA
"""

import os
import json
import yaml
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import argparse
from pathlib import Path

def load_config(config_path="training_config.yaml"):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_and_prepare_dataset(config):
    """Load JSONL dataset and prepare for training"""
    data_path = config['dataset']['train_file']
    
    # Load JSONL file
    print(f"📂 Loading dataset from {data_path}...")
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # Apply max_samples limit if specified
    max_samples = config['dataset'].get('max_samples')
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✅ Loaded {len(dataset)} conversation pairs")
    
    # Split into train/validation
    val_split = config['dataset']['validation_split']
    if val_split > 0:
        split_dataset = dataset.train_test_split(test_size=val_split, seed=config['training']['seed'])
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"📊 Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    
    return dataset, None

def format_prompts(examples, tokenizer):
    """
    Format conversation into proper chat template
    Input format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    texts = []
    for messages in examples["messages"]:
        # Use the model's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

def main():
    parser = argparse.ArgumentParser(description="Train Hive Personality Model")
    parser.add_argument("--config", type=str, default="training_config.yaml", 
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    print("🔧 Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model and tokenizer
    print(f"🤖 Loading base model: {config['model']['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['base_model'],
        max_seq_length=config['model']['max_seq_length'],
        dtype=None,  # Auto-detect
        load_in_4bit=config['model']['load_in_4bit'],
    )
    
    # Apply LoRA
    print("🔗 Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=config['training']['seed'],
    )
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(config)
    
    # Format dataset using chat template
    print("📝 Formatting dataset with chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_prompts(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: format_prompts(x, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    
    # Training arguments (compatible with different transformers versions)
    training_args_dict = {
        'output_dir': output_dir,
        'num_train_epochs': config['training']['num_train_epochs'],
        'per_device_train_batch_size': config['training']['per_device_train_batch_size'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'optim': config['training']['optim'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'fp16': config['training']['fp16'],
        'bf16': config['training']['bf16'],
        'max_grad_norm': config['training']['max_grad_norm'],
        'warmup_steps': config['training']['warmup_steps'],
        'logging_steps': config['training']['logging_steps'],
        'save_strategy': config['training']['save_strategy'],
        'save_steps': config['training']['save_steps'],
        'save_total_limit': config['training']['save_total_limit'],
        'group_by_length': config['training']['group_by_length'],
        'report_to': config['training']['report_to'],
        'seed': config['training']['seed'],
    }
    
    # Add evaluation args only if evaluating (skip if parameter not supported)
    try:
        if eval_dataset:
            training_args_dict['evaluation_strategy'] = config['training']['evaluation_strategy']
            training_args_dict['eval_steps'] = config['training']['eval_steps']
        training_args = TrainingArguments(**training_args_dict)
    except TypeError as e:
        # Fallback: try without evaluation_strategy parameter (older transformers versions)
        if 'evaluation_strategy' in str(e):
            print("⚠️  Warning: evaluation_strategy not supported in this transformers version")
            print("   Disabling validation for faster training...")
            training_args_dict.pop('evaluation_strategy', None)
            training_args_dict.pop('eval_steps', None)
            training_args = TrainingArguments(**training_args_dict)
        else:
            raise
    
    # Initialize trainer
    print("🎯 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
        packing=False,  # Don't pack sequences for conversational data
    )
    
    # Show model info
    print("\n" + "="*50)
    print("📊 Model Information:")
    print("="*50)
    model.print_trainable_parameters()
    print("="*50 + "\n")
    
    # Start training
    print("🚀 Starting training...")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Training for {config['training']['num_train_epochs']} epochs\n")
    
    trainer.train(resume_from_checkpoint=args.resume)
    
    # Save final model
    print("\n💾 Saving final model...")
    final_output = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    
    print(f"✅ Training completed! Model saved to {final_output}")
    
    # Export options
    print("\n" + "="*50)
    print("📦 Export Options:")
    print("="*50)
    print("To export your model, run:")
    print(f"  python export_model.py --input {final_output}")
    print("\nAvailable export formats:")
    print("  - merged_16bit: Full precision merged model")
    print("  - merged_4bit: 4-bit quantized merged model")
    print("  - gguf: GGUF format for llama.cpp/Ollama")
    print("="*50)

if __name__ == "__main__":
    main()
