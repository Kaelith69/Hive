# ☁️ Cloud Training Guide for Hive

Your training setup is now on GitHub! Here's how to train in the cloud with free or low-cost GPU access.

## 🎯 Quick Links

- **Your Repository**: https://github.com/Kaelith69/Hive
- **Dataset**: 21,220 conversation pairs ready to train

---

## 🚀 Option 1: Google Colab (Free/Pro)

**Best for**: Quick experimentation, free T4 GPU

### Setup (5 minutes)

1. **Open Google Colab**: https://colab.research.google.com

2. **Create new notebook** and run these cells:

```python
# Cell 1: Clone your repository
!git clone https://github.com/Kaelith69/Hive.git
%cd Hive

# Cell 2: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install transformers datasets accelerate peft trl bitsandbytes pyyaml

# Cell 3: Validate dataset
!python validate_dataset.py --input dataset/final/personality.jsonl

# Cell 4: Start training
!python train.py

# Cell 5: Export model (after training completes)
!python export_model.py --input outputs/final_model --type gguf --name hive-personality

# Cell 6: Download trained model
from google.colab import files
!zip -r hive-personality.zip exports/
files.download('hive-personality.zip')
```

### Pros & Cons
✅ Free tier available (T4 GPU)  
✅ No setup required  
✅ Easy to use  
❌ Session timeouts (12 hours max)  
❌ May disconnect during training  

**Tip**: Get Colab Pro ($10/month) for better GPUs (V100/A100) and longer sessions.

---

## 🚀 Option 2: Kaggle Notebooks (Free)

**Best for**: Reliable free GPU, no timeouts for committed users

### Setup

1. **Go to Kaggle**: https://www.kaggle.com
2. **Create new notebook**: Notebooks → New Notebook
3. **Settings** (right panel):
   - Accelerator: **GPU T4 x2** (free!)
   - Internet: **On**
   - Persistence: **Files only**

4. **Run these commands**:

```python
# Clone repository
!git clone https://github.com/Kaelith69/Hive.git
%cd Hive

# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q transformers datasets accelerate peft trl bitsandbytes pyyaml

# Validate and train
!python validate_dataset.py
!python train.py

# Export
!python export_model.py --input outputs/final_model --type gguf
```

5. **Download outputs**: Outputs are saved in Kaggle's output folder (auto-persisted)

### Pros & Cons
✅ Free 30 hours/week GPU (T4 x2)  
✅ More reliable than Colab free  
✅ Outputs auto-saved  
✅ Better internet bandwidth  
❌ Need phone verification  

---

## 🚀 Option 3: RunPod (Pay-as-you-go)

**Best for**: Serious training, full control, better GPUs

### Setup

1. **Sign up**: https://runpod.io
2. **Add credits**: $10 = ~10-20 hours of RTX 4090
3. **Deploy Pod**:
   - Template: **RunPod PyTorch**
   - GPU: **RTX 4090** (~$0.69/hour) or **RTX 3090** (~$0.39/hour)
   - Disk: 50GB

4. **Connect via JupyterLab** or SSH:

```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets accelerate peft trl bitsandbytes pyyaml

# Clone repository
git clone https://github.com/Kaelith69/Hive.git
cd Hive

# Train
python train.py

# Export
python export_model.py --input outputs/final_model --type gguf
```

5. **Download model**: Use JupyterLab file browser or SCP

### Pricing
- RTX 3090 (24GB): $0.39/hour → ~$2-4 for full training
- RTX 4090 (24GB): $0.69/hour → ~$3-6 for full training
- A100 (40GB): $1.89/hour → ~$5-10 for full training

### Pros & Cons
✅ Powerful GPUs (4090, A100)  
✅ No timeouts  
✅ Full control  
✅ Storage persists  
❌ Costs money (but cheap)  
❌ Need credit card  

---

## 🚀 Option 4: Vast.ai (Cheapest)

**Best for**: Budget training, many GPU options

### Setup

1. **Sign up**: https://vast.ai
2. **Search for instances**: 
   - GPU: **RTX 3090** or **RTX 4090**
   - VRAM: 24GB+
   - Sort by: **$ / hr**

3. **Rent instance** (~$0.20-0.30/hour for 3090)

4. **Connect** and run:

```bash
# Clone repo
git clone https://github.com/Kaelith69/Hive.git
cd Hive

# Install (if not pre-installed)
pip install torch transformers datasets accelerate peft trl bitsandbytes pyyaml
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

# Train
python train.py
```

### Pros & Cons
✅ Very cheap ($0.20-0.50/hour)  
✅ Many GPU options  
❌ Quality varies by host  
❌ Setup can be tricky  
❌ May have reliability issues  

---

## 🚀 Option 5: Lightning.ai (Hybrid)

**Best for**: Easy cloud development environment

### Setup

1. **Sign up**: https://lightning.ai
2. **Create Studio** with GPU
3. **Open terminal** and clone:

```bash
git clone https://github.com/Kaelith69/Hive.git
cd Hive
pip install -r requirements.txt
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
python train.py
```

### Pros & Cons
✅ Good free tier  
✅ Professional environment  
✅ Easy to use  
❌ Credits run out quickly  

---

## 📊 Cloud Platform Comparison

| Platform | Cost | GPU | Training Time* | Reliability | Ease of Use |
|----------|------|-----|---------------|-------------|-------------|
| **Colab Free** | Free | T4 | 3-4 hours ⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Colab Pro** | $10/mo | V100/A100 | 1-2 hours | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Kaggle** | Free | T4 x2 | 2-3 hours ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **RunPod** | $3-6 | RTX 4090 | 45-90 min | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Vast.ai** | $1-3 | RTX 3090 | 1.5-2.5 hours | ⭐⭐⭐ | ⭐⭐⭐ |
| **Lightning** | $5-10 | A10G | 2-3 hours | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

*Optimized config for 21k samples on 3B model (2 epochs)

---

## 🎯 Recommended Workflow

### For Beginners: Start with Kaggle
1. Free, reliable GPU
2. No payment required
3. Good for learning

### For Quick Tests: Use Colab
1. Fastest setup
2. Good for debugging
3. Upgrade to Pro if needed

### For Production: Use RunPod
1. Fast, reliable training
2. Cheap for quality
3. Download model when done

---

## 📦 After Training

Once training completes in the cloud:

### 1. Export Model
```python
!python export_model.py --input outputs/final_model --type gguf --name hive-personality
```

### 2. Download to Local Machine

**Colab**:
```python
from google.colab import files
!zip -r model.zip exports/
files.download('model.zip')
```

**Kaggle**: Check the `/kaggle/working/exports` folder

**RunPod/Vast**: Use SCP or download via JupyterLab

### 3. Use with Ollama (Local)
```powershell
# Extract downloaded model
# Create Modelfile
cd exports/hive-personality.gguf
ollama create hive-personality -f Modelfile
ollama run hive-personality
```

---

## 🔧 Training Configuration Tips

### For Cloud GPUs

**If you have 16GB+ VRAM** (Colab Pro, RunPod):
```yaml
# Edit training_config.yaml before training
per_device_train_batch_size: 4  # Increase batch size
gradient_accumulation_steps: 2
model:
  base_model: "unsloth/Mistral-7B-Instruct-v0.3"  # Use larger model
```

**If you have 8-12GB VRAM** (Colab Free, Kaggle):
```yaml
# Keep defaults or reduce
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
model:
  base_model: "unsloth/Llama-3.2-3B-Instruct"  # Smaller model
  max_seq_length: 1024  # Reduce if needed
```

---

## 🐛 Common Cloud Issues

### Colab Disconnects During Training
**Solution**: Save checkpoints frequently, resume training:
```python
# Resume from last checkpoint
!python train.py --resume outputs/checkpoint-500
```

### Out of Memory in Cloud
**Solution**: Reduce batch size in `training_config.yaml`:
```yaml
per_device_train_batch_size: 1
max_seq_length: 1024
```

### Slow Downloads
**Solution**: Upload directly to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r exports/ /content/drive/MyDrive/
```

---

## ✅ Quick Start Checklist

- [ ] Repository pushed to GitHub ✅ (Done!)
- [ ] Choose cloud platform
- [ ] Create account & verify
- [ ] Open notebook/terminal
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Validate dataset
- [ ] Start training
- [ ] Export model
- [ ] Download to local machine
- [ ] Deploy with Ollama

---

## 📞 Need Help?

- Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed troubleshooting
- Unsloth Discord: https://discord.gg/unsloth
- Your repo: https://github.com/Kaelith69/Hive

---

**Happy Cloud Training! ☁️🚀**

*Training a personality has never been easier!*
