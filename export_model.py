"""
Export trained model to various formats
Supports: HuggingFace, GGUF (for Ollama/llama.cpp), merged models
"""

import os
import argparse
import yaml
from pathlib import Path
from unsloth import FastLanguageModel

def load_config(config_path="training_config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def export_model(model_path, export_type, output_name, quantization_method="q4_k_m"):
    """
    Export trained model to different formats
    
    Args:
        model_path: Path to the trained model (LoRA adapters)
        export_type: Type of export (lora, merged_16bit, merged_4bit, gguf)
        output_name: Name for the exported model
        quantization_method: Quantization method for GGUF export
    """
    print(f"🔄 Loading model from {model_path}...")
    
    # Load the trained model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Load in full precision for export
    )
    
    output_dir = f"exports/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    if export_type == "lora":
        print("💾 Exporting LoRA adapters only...")
        model.save_pretrained(f"{output_dir}_lora")
        tokenizer.save_pretrained(f"{output_dir}_lora")
        print(f"✅ LoRA adapters saved to {output_dir}_lora")
        
    elif export_type == "merged_16bit":
        print("🔗 Merging LoRA weights into base model (16-bit)...")
        model.save_pretrained_merged(
            f"{output_dir}_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✅ Merged 16-bit model saved to {output_dir}_16bit")
        
    elif export_type == "merged_4bit":
        print("🗜️ Merging and quantizing to 4-bit...")
        model.save_pretrained_merged(
            f"{output_dir}_4bit",
            tokenizer,
            save_method="merged_4bit",
        )
        print(f"✅ Merged 4-bit model saved to {output_dir}_4bit")
        
    elif export_type == "gguf":
        print(f"📦 Exporting to GGUF format ({quantization_method})...")
        gguf_path = f"{output_dir}.gguf"
        model.save_pretrained_gguf(
            gguf_path,
            tokenizer,
            quantization_method=quantization_method,
        )
        print(f"✅ GGUF model saved to {gguf_path}")
        print("\n🎉 Ready for Ollama!")
        print(f"To use with Ollama, create a Modelfile:")
        print(f"  FROM ./{gguf_path}/{output_name}-{quantization_method.upper()}.gguf")
        print(f"  ollama create {output_name} -f Modelfile")
        
    elif export_type == "all":
        print("📦 Exporting to all formats...")
        # Export LoRA
        model.save_pretrained(f"{output_dir}_lora")
        tokenizer.save_pretrained(f"{output_dir}_lora")
        print(f"✅ LoRA adapters saved")
        
        # Export merged 16-bit
        model.save_pretrained_merged(
            f"{output_dir}_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✅ Merged 16-bit model saved")
        
        # Export GGUF
        gguf_path = f"{output_dir}.gguf"
        model.save_pretrained_gguf(
            gguf_path,
            tokenizer,
            quantization_method=quantization_method,
        )
        print(f"✅ GGUF model saved")
        print(f"\n🎉 All exports completed in exports/{output_name}/")
    
    else:
        print(f"❌ Unknown export type: {export_type}")
        print("Available types: lora, merged_16bit, merged_4bit, gguf, all")

def main():
    parser = argparse.ArgumentParser(description="Export trained Hive model")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to trained model (e.g., outputs/final_model)")
    parser.add_argument("--type", type=str, default="gguf",
                       choices=["lora", "merged_16bit", "merged_4bit", "gguf", "all"],
                       help="Export type")
    parser.add_argument("--name", type=str, default="hive-personality",
                       help="Output model name")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                       choices=["q4_k_m", "q4_k_s", "q5_k_m", "q8_0", "f16"],
                       help="Quantization method for GGUF export")
    parser.add_argument("--config", type=str, default="training_config.yaml",
                       help="Config file (for default values)")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.input):
        print(f"❌ Model path not found: {args.input}")
        print("Make sure you've completed training first!")
        return
    
    print("="*60)
    print("🚀 Hive Model Export Tool")
    print("="*60)
    print(f"Input model: {args.input}")
    print(f"Export type: {args.type}")
    print(f"Output name: {args.name}")
    if args.type in ["gguf", "all"]:
        print(f"Quantization: {args.quantization}")
    print("="*60 + "\n")
    
    export_model(
        model_path=args.input,
        export_type=args.type,
        output_name=args.name,
        quantization_method=args.quantization
    )
    
    print("\n✅ Export completed successfully!")

if __name__ == "__main__":
    main()
