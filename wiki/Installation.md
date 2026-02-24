# ⚡ Installation

Getting Hive running takes about 10 minutes if your environment cooperates. Add 45 minutes if Python dependencies decide to have opinions.

---

## Prerequisites

- Python **3.10 or 3.11** (3.12 has known issues with some ML libs, avoid it for now)
- A CUDA-capable GPU with **8GB+ VRAM** for local training, OR a free Kaggle/Colab account
- Your WhatsApp chat exports (`.txt` format, exported without media)

---

## Option A: Local Setup

### 1. Clone the Repo

```bash
git clone https://github.com/Kaelith69/Hive.git
cd Hive
```

### 2. Create a Virtual Environment

Don't skip this. You've been burned before. You know.

```bash
python -m venv venv

# Activate:
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows (PowerShell)
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs: `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `pyyaml`, `numpy`, `pandas`, `tqdm`.

### 4. Install Unsloth

Unsloth is installed from source. Choose the right variant:

**Google Colab (any):**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Local — CUDA 12.1 + PyTorch 2.3:**
```bash
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

**Local — CUDA 12.1 + PyTorch 2.1:**
```bash
pip install "unsloth[cu121-torch210] @ git+https://github.com/unslothai/unsloth.git"
```

**Local — CUDA 11.8:**
```bash
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

Not sure which CUDA you have? `nvcc --version` or `nvidia-smi`.

### 5. Windows (PowerShell)

A setup script is included:

```powershell
.\setup.ps1
```

---

## Option B: Kaggle (Free T4×2 — Recommended for Most People)

Kaggle gives you 2× T4 GPUs for free, 30 hours/week. This is the sweet spot.

1. Go to [kaggle.com](https://kaggle.com) and create an account
2. Create a new notebook
3. Go to **Settings → Accelerator → GPU T4 x2**
4. Upload `dataset/final/personality.jsonl` as a Kaggle dataset, or copy its content into the notebook
5. Copy cells from `KAGGLE_CELL.txt` into your notebook, OR upload `kaggle_training_notebook.py`
6. Run all cells
7. Download `outputs/` when it finishes

Full instructions in `KAGGLE_SETUP.md`.

---

## Option C: Google Colab

### Free Tier (T4 — 3-4 hours)

1. Create a new Colab notebook
2. Runtime → Change runtime type → **T4 GPU**
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```
4. Upload your dataset and scripts
5. Run

**Warning:** Colab free tier disconnects after ~90 minutes of inactivity. Use `train.py --resume` to continue from checkpoint.

### Colab Pro (V100 — 1-2 hours)

Same setup, just faster. Use `bf16: true` in `training_config.yaml` for V100+ (it supports BF16 unlike T4). `train.py` will auto-detect this at runtime.

---

## Option D: RunPod

For when you want it done in under an hour and don't mind spending a few dollars.

1. Go to [runpod.io](https://runpod.io)
2. Deploy a PyTorch pod with RTX 4090 (or similar)
3. SSH in, clone the repo, install deps
4. Run `python train.py`
5. Done in 45-90 minutes

Full guide in `CLOUD_TRAINING.md`.

---

## Verifying the Installation

```bash
# Check Python
python --version  # Should be 3.10.x or 3.11.x

# Check CUDA (if running locally)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check Unsloth
python -c "from unsloth import FastLanguageModel; print('unsloth ok')"

# Validate your dataset (run this before training)
python validate_dataset.py
```

If `torch.cuda.is_available()` returns `False` on a machine with an NVIDIA GPU, your CUDA drivers need attention. Not a Hive problem — that's between you and NVIDIA.

---

## Minimum Hardware Requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU VRAM | 8 GB (RTX 3060) | 16 GB+ |
| System RAM | 16 GB | 32 GB |
| Disk Space | 20 GB free | 50 GB+ |
| CUDA | 11.8 | 12.1+ |

Training `outputs/` and `exports/` can get large. Make sure you have space.
