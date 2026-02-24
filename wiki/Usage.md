# 🚀 Usage

The full pipeline, step by step. From WhatsApp export to running model. Let's go.

---

## Step 0: Export Your WhatsApp Chats

In WhatsApp (mobile):
- Open any chat
- Tap the three dots → More → Export chat
- Choose **Without Media**
- Export to a `.txt` file
- Transfer to your machine

Drop all exported `.txt` files into `dataset/raw/`. Subdirectories are fine — `extract.py` walks recursively. You can have multiple chats in different subfolders.

---

## Step 1: Extract Your Messages

Open `extract.py` and set your WhatsApp display name:

```python
YOUR_NAME = "Sayanth"   # ← change this to YOUR display name in WhatsApp
```

Then run:

```bash
python extract.py
```

**What this does:**
- Reads every `.txt` in `dataset/raw/` and subdirectories
- Keeps only messages where the sender matches `YOUR_NAME`
- Filters out: `<Media omitted>`, deleted messages, system messages
- Strips all URLs
- Skips empty messages (after URL removal)
- Writes cleaned lines to `dataset/cleaned/`

**Check your output:**
```bash
wc -l dataset/cleaned/*.txt       # Linux/macOS
# Or on Windows PowerShell:
(Get-Content .\dataset\cleaned\*.txt | Measure-Object -Line).Lines
```

You want thousands of lines, not dozens. If you have less than ~1000 messages, results will be rough.

---

## Step 2: Convert to JSONL

```bash
python convert.py
```

Creates `dataset/final/personality.jsonl`. Each line is a JSON object with a `messages` array:

```json
{"messages": [{"role": "user", "content": "okay that's wild"}, {"role": "assistant", "content": "lmao true tho"}]}
```

**Count your entries:**
```bash
wc -l dataset/final/personality.jsonl   # Linux/macOS
```

Aim for 10,000+ entries minimum. 20,000+ is great.

---

## Step 3: Validate the Dataset

```bash
python validate_dataset.py
```

This checks for malformed JSON, missing fields, and gives you entry statistics. Fix any errors before training.

**If validation finds issues:**
```bash
python fix_dataset.py      # Attempts to repair malformed entries
python clean_incomplete.py # Removes entries with missing user or assistant messages
```

Re-run `validate_dataset.py` afterward to confirm.

---

## Step 4: Configure Training

Edit `training_config.yaml`. The defaults work well — only touch these if you know what you're doing:

```yaml
model:
  base_model: "unsloth/Llama-3.2-3B-Instruct"
  # Alternatives: "unsloth/Qwen2.5-3B-Instruct", "unsloth/Phi-4", "unsloth/Mistral-7B-Instruct-v0.3"

lora:
  r: 16          # Rank. Higher = more params, better quality, slower. 8-32 is reasonable.
  lora_alpha: 16 # Usually keep equal to r

training:
  num_train_epochs: 2          # 2 is fast (2-3h on T4). 3 is better quality.
  learning_rate: 3.0e-4        # Higher LR = faster convergence, riskier
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 4×4 = 16
```

---

## Step 5: Train

```bash
python train.py
```

Or with a custom config file:
```bash
python train.py --config my_config.yaml
```

Resume from a checkpoint if training was interrupted:
```bash
python train.py --resume outputs/checkpoint-500
```

**Watch for:**
- Loss should decrease over time (typical range: starts ~2.0, ends ~0.5-1.0)
- If loss goes up or NaN appears, something is wrong (see Troubleshooting wiki)
- Checkpoints save every 500 steps to `outputs/`

Training ends with:
```
✅ Training completed! Model saved to outputs/final_model
```

---

## Step 6: Export the Model

```bash
# GGUF (for Ollama / llama.cpp) — recommended for most users
python export_model.py --input outputs/final_model --type gguf --name hive-personality

# Merged 16-bit HuggingFace model
python export_model.py --input outputs/final_model --type merged_16bit --name hive-personality

# All formats at once
python export_model.py --input outputs/final_model --type all --name hive-personality

# Different GGUF quantization level
python export_model.py --input outputs/final_model --type gguf --quantization q5_k_m
```

**Quantization options:**

| Method | Size | Quality | Use case |
|--------|------|---------|----------|
| `q4_k_m` | ~2 GB | Good | Default, best balance |
| `q5_k_m` | ~2.5 GB | Better | If you have disk space |
| `q8_0` | ~4 GB | Near-lossless | Maximum quality |
| `f16` | ~7 GB | Lossless | For further fine-tuning |

---

## Step 7: Run Your Trained Model

**With Ollama:**

```bash
# Create Modelfile
echo "FROM ./exports/hive-personality.gguf/hive-personality-Q4_K_M.gguf" > Modelfile
ollama create hive-personality -f Modelfile

# Chat with yourself
ollama run hive-personality
```

**With llama.cpp:**
```bash
./llama-cli -m exports/hive-personality.gguf/hive-personality-Q4_K_M.gguf \
  --chat-template llama3 -i
```

---

## Tips

- Multiple chat files? Great. Just put them all in `dataset/raw/` and let `extract.py` handle it.
- Want to filter differently (keep URLs, different message patterns)? Edit `extract.py`.
- Training on Kaggle? See [Installation](Installation) and `KAGGLE_SETUP.md`.
- Model sounds weird? You need more data. Or fewer epochs. Or both. Check [Troubleshooting](Troubleshooting).
