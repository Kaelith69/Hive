# Kaggle Notebook for Hive Personality Training
# Copy-paste this entire script into a Kaggle notebook cell
# GPU: T4 x2 (free), Internet: ON

# ============================================================================
# CELL 1: Clone Repository & Install Dependencies
# ============================================================================

!git clone https://github.com/Kaelith69/Hive.git
%cd Hive

print("📦 Installing PyTorch with CUDA support...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print("⚡ Installing Unsloth (fast fine-tuning library)...")
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

print("📚 Installing training dependencies...")
!pip install -q transformers datasets accelerate peft trl bitsandbytes pyyaml

print("✅ All dependencies installed!")

# ============================================================================
# CELL 2: Validate Dataset
# ============================================================================

print("\n📊 Validating dataset...")
!python validate_dataset.py --input dataset/final/personality.jsonl --sample 2

# ============================================================================
# CELL 3: Start Training (2-3 hours on Kaggle T4 x2)
# ============================================================================

print("\n🚀 Starting training... This will take ~2-3 hours")
print("=" * 70)

!python train.py

print("\n" + "=" * 70)
print("✅ Training completed!")
print("=" * 70)

# ============================================================================
# CELL 4: Export Model to GGUF (for Ollama)
# ============================================================================

print("\n📦 Exporting model to GGUF format...")
!python export_model.py --input outputs/final_model --type gguf --name hive-personality

print("\n✅ Model exported successfully!")

# ============================================================================
# CELL 5: Check Output Files
# ============================================================================

import os
from pathlib import Path

print("\n📁 Training outputs:")
if Path("outputs").exists():
    for item in os.listdir("outputs"):
        path = f"outputs/{item}"
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"  📄 {item} ({size:.1f} MB)")
        else:
            print(f"  📁 {item}/")

print("\n📁 Exported models:")
if Path("exports").exists():
    for item in os.listdir("exports"):
        path = f"exports/{item}"
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024*1024)
            if size > 1024:
                print(f"  📦 {item} ({size/1024:.1f} GB)")
            else:
                print(f"  📦 {item} ({size:.1f} MB)")
        else:
            print(f"  📁 {item}/")

# ============================================================================
# CELL 6: Download Model (Optional - if file size allows)
# ============================================================================

print("\n💾 To download your trained model:")
print("  1. Go to Output tab in Kaggle")
print("  2. All files in 'outputs/' and 'exports/' are saved")
print("  3. Click download to get the model")

print("\n" + "=" * 70)
print("🎉 Training pipeline complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Download the GGUF model from Kaggle outputs")
print("  2. Use with Ollama locally:")
print("     ollama create hive-personality -f Modelfile")
print("     ollama run hive-personality")
print("\n  See CLOUD_TRAINING.md for more details")
print("=" * 70)
