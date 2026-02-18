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
import inspect

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


def resolve_precision_settings(config):
    configured_fp16 = bool(config['training'].get('fp16', False))
    configured_bf16 = bool(config['training'].get('bf16', False))

    if not torch.cuda.is_available():
        return configured_fp16, False

    major, _minor = torch.cuda.get_device_capability(0)
    supports_bf16 = major >= 8

    if configured_bf16 and not supports_bf16:
        print("⚠️  BF16 requested but GPU does not support it. Switching to FP16.")
        return True, False

    return configured_fp16, configured_bf16


def get_eval_arg_name():
    params = inspect.signature(TrainingArguments.__init__).parameters
    if 'evaluation_strategy' in params:
        return 'evaluation_strategy'
    if 'eval_strategy' in params:
        return 'eval_strategy'
    return None

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
    
    fp16_value, bf16_value = resolve_precision_settings(config)

    # Training arguments (compatible with different transformers versions)
    training_args_dict = {
        'output_dir': output_dir,
        'num_train_epochs': config['training']['num_train_epochs'],
        'per_device_train_batch_size': config['training']['per_device_train_batch_size'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'optim': config['training']['optim'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'fp16': fp16_value,
        'bf16': bf16_value,
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
    
    eval_arg_name = get_eval_arg_name()
    if eval_dataset and eval_arg_name:
        training_args_dict[eval_arg_name] = config['training']['evaluation_strategy']
        training_args_dict['eval_steps'] = config['training']['eval_steps']
    elif eval_dataset and not eval_arg_name:
        print("⚠️  This transformers build has no evaluation strategy argument; disabling validation.")

    training_args = TrainingArguments(**training_args_dict)
    
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
