# 🐍 Kaggle Notebook Setup (Copy-Paste Ready)

Quick guide to run training on Kaggle with copy-paste cells.

## 📋 Step-by-Step

### 1. Create Kaggle Notebook
- Go to https://www.kaggle.com/code
- Click **New Notebook**

### 2. Configure GPU
Right panel settings:
- **Accelerator**: GPU T4 x2 ✅
- **Internet**: ON ✅
- **Persistence**: Files only ✅

### 3. Copy-Paste Cells

#### Cell 1: Setup & Install
```python
!git clone https://github.com/Kaelith69/Hive.git
%cd Hive

print("📦 Installing PyTorch...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print("⚡ Installing Unsloth...")
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

print("📚 Installing dependencies...")
!pip install -q transformers datasets accelerate peft trl bitsandbytes pyyaml

print("✅ Ready!")
```

#### Cell 2: Validate Dataset
```python
!python validate_dataset.py --input dataset/final/personality.jsonl --sample 2
```

#### Cell 3: Train Model (2-3 hours)
```python
print("🚀 Starting training...")
!python train.py
print("✅ Done!")
```

#### Cell 4: Export to GGUF
```python
print("📦 Exporting model...")
!python export_model.py --input outputs/final_model --type gguf --name hive-personality
print("✅ Exported!")
```

#### Cell 5: Check Files (Optional)
```python
import os
from pathlib import Path

print("📁 Outputs:")
if Path("outputs").exists():
    for item in os.listdir("outputs"):
        print(f"  {item}")

print("\n📁 Exports:")
if Path("exports").exists():
    for item in os.listdir("exports"):
        print(f"  {item}")
```

### 4. Download Model
- After training completes
- Go to **Output** tab on right
- Download the GGUF file from `exports/` folder
- (~3-4 GB GGUF file)

## ⏱️ Timeline

| Step | Time | What's Happening |
|------|------|------------------|
| Cell 1 | 3-5 min | Installing packages |
| Cell 2 | 30 sec | Dataset validation |
| Cell 3 | **2-3 hours** | Training (grab coffee ☕) |
| Cell 4 | 10-15 min | GGUF export (quantization) |
| Downloads | 5-10 min | Downloading files |

## 💾 File Sizes

After training, expect:
- `outputs/final_model/`: ~7-8 GB (full merged model)
- `exports/hive-personality.gguf/`: ~3-4 GB (GGUF quantized)
  - `*-Q4_K_M.gguf`: 3.8 GB ← **Download this one**
  - `*-Q4_K_S.gguf`: 3.1 GB (smaller, slightly lower quality)

## 🚀 Faster Option: Use Pre-Built Script

Instead of copy-pasting cells, you can run the notebook script:

```python
# Single cell approach
!git clone https://github.com/Kaelith69/Hive.git
%cd Hive
!pip install -q torch transformers datasets accelerate peft trl bitsandbytes pyyaml
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!python train.py
!python export_model.py --input outputs/final_model --type gguf --name hive-personality
```

Or run the provided script:
```python
exec(open('kaggle_training_notebook.py').read())
```

## ⚠️ Important Notes

### If Training Disconnects
Kaggle has time limits (~12 hours). If training disconnects:
1. Restart notebook
2. Run: `!python train.py --resume outputs/checkpoint-500`
3. Training resumes from last checkpoint

### Monitor Training
Watch the loss values - should decrease steadily:
- Start: ~2.5-3.0
- After 50%: ~1.0-1.5
- After 100%: ~0.5-0.8

### Memory Management
- Kaggle gives you ~15 GB free disk space
- Model files are auto-saved
- Large files may time out on download → use `!zip -r model.zip exports/`

## 📥 After Training: Download Model

### Option 1: Kaggle Output Tab (Recommended)
1. Click **Output** tab
2. All files auto-saved
3. Click file → Download

### Option 2: Download via Notebook
```python
from kaggle.api.kaggle_api_extended import KaggleApi

# Download all outputs
api = KaggleApi()
api.dataset_download_files('username/kernel-output', path='./downloads')
```

### Option 3: Manual Download
```python
import os
!ls -lh outputs/
!ls -lh exports/
```
Then download from Kaggle UI

## 🎯 After Download

Use model locally with Ollama:

```powershell
# Extract downloaded model
Expand-Archive hive-personality.gguf.zip

# Create Modelfile
cd exports/hive-personality.gguf
@"
FROM ./hive-personality-Q4_K_M.gguf
PARAMETER temperature 0.7
"@ | Out-File -Encoding UTF8 Modelfile

# Create in Ollama
ollama create hive-personality -f Modelfile
ollama run hive-personality "Hello!"
```

## 🐛 Troubleshooting

### "Out of memory" error
```python
# Reduce batch size in training_config.yaml before training
per_device_train_batch_size: 2  # instead of 4
```

### "CUDA error" during training
- Restart notebook and resume from checkpoint
- Check GPU is enabled in settings
- May be temporary Kaggle issue

### Download file too large
```python
# Compress before download
!zip -r hive-model.zip exports/hive-personality.gguf/hive-personality-Q4_K_M.gguf
```

### Training too slow
- Kaggle T4 is slower than RTX 4090, but still reasonable
- Consider RunPod RTX 4090 for 10x faster (costs ~$1)
- Use RunPod if you need results quickly

## ✅ Checklist

- [ ] Create Kaggle account
- [ ] Create notebook with GPU T4 x2
- [ ] Copy-paste installation cell
- [ ] Copy-paste training cell
- [ ] Wait 2-3 hours ☕
- [ ] Download GGUF model
- [ ] Use with Ollama locally

---

**You're all set! Training should be smooth sailing now.** 🚀
