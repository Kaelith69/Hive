# 🏗️ Architecture

Hive is a linear pipeline with four main stages and three utility scripts. There's no server, no API, no cloud service phoning home — just Python scripts chained together like a functional assembly line for your personality.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HIVE PIPELINE                            │
│                                                                 │
│  WhatsApp .txt  →  extract.py  →  convert.py  →  train.py  →  export  │
│                                                                 │
│  dataset/raw/   →  dataset/cleaned/  →  personality.jsonl  →  outputs/ → exports/ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: `extract.py` — The Filter

The WhatsApp export format looks like this:
```
3/14/24, 10:42 AM - Sayanth: okay that's wild
3/14/24, 10:43 AM - Friend: lmao right
3/14/24, 10:43 AM - Sayanth: <Media omitted>
```

`extract.py` does the following:
- Walks `dataset/raw/` recursively (handles multi-chat setups)
- Matches lines with regex: `".* - (.*?): (.*)"`
- Keeps only lines where the sender matches `YOUR_NAME`
- Drops: `<Media omitted>`, `You deleted this message`, any line containing `deleted this message`
- Strips all URLs (regex: `https?://\S+|www\.\S+`)
- Skips empty messages (after URL stripping)
- Writes cleaned messages, one per line, to `dataset/cleaned/`

**Output:** plain text files with one message per line.

---

## Stage 2: `convert.py` — The Converter

Takes the cleaned message files and creates conversational pairs using a **sliding window** approach:

```
Message[i]   → user
Message[i+1] → assistant
```

This treats consecutive messages as a conversation: "what you said" followed by "what you said next" becomes a Q→A pair. It's a simplification of real conversation dynamics, but it gives the model 21,000+ examples of your communication style.

**Output format (JSONL):**
```json
{"messages": [{"role": "user", "content": "okay that's wild"}, {"role": "assistant", "content": "lmao true tho"}]}
```

This is the standard HuggingFace `messages` format used by `apply_chat_template()`.

---

## Stage 3: `train.py` — The Brain Forge

The core ML stage. Uses three key libraries:

### Unsloth
Fast, memory-efficient fine-tuning. Speeds up training 2-5× vs vanilla HuggingFace training by using custom CUDA kernels. This is why you can run on a T4 instead of needing an A100.

### PEFT (Parameter-Efficient Fine-Tuning) — LoRA
Instead of updating all 3 billion parameters (which would require enormous GPU memory), LoRA freezes the base model and injects small trainable matrices (rank decompositions) into the attention layers:

```
W_output = W_base (frozen) + B×A (trainable, rank r=16)
```

With `r=16`, only ~8 million parameters are actually trained. The base model stays frozen. This is why 8GB VRAM is enough.

**LoRA target modules:**
```
q_proj, k_proj, v_proj, o_proj     ← attention
gate_proj, up_proj, down_proj      ← feedforward (MLP)
```

### 4-bit Quantization (QLoRA)
The base model is loaded in 4-bit precision via `bitsandbytes`. This halves (roughly) the memory needed to hold the model in VRAM. The LoRA adapters themselves train in full precision.

### Training Flow

```
Load base model (4-bit) + tokenizer
        │
Apply LoRA adapters (freeze base, init A×B)
        │
Load personality.jsonl → apply chat template → tokenize
        │
SFTTrainer (Supervised Fine-Tuning)
  - AdamW 8-bit optimizer
  - LR: 3e-4, warmup: 10 steps
  - 2 epochs, batch 4, grad accum 4 (effective batch 16)
  - FP16, grad clip 0.3
  - Save checkpoint every 500 steps
        │
Save final model → outputs/final_model/
```

---

## Stage 4: `export_model.py` — The Packager

Converts the trained model into deployable formats:

| Format | What it is | Use case |
|--------|-----------|----------|
| `lora` | Raw LoRA adapter weights only | Smallest export, needs base model to run |
| `merged_16bit` | Base + LoRA merged, full precision | HuggingFace Hub publishing |
| `merged_4bit` | Base + LoRA merged, 4-bit | Smaller HF model |
| `gguf` | Quantized binary format | Ollama, llama.cpp, local inference |

GGUF quantization options: `q4_k_m` (default, good balance), `q5_k_m` (better quality), `q8_0` (near-lossless), `f16` (full precision).

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `validate_dataset.py` | Checks JSONL format, reports entry count, flags malformed lines |
| `fix_dataset.py` | Attempts to repair malformed JSONL entries |
| `clean_incomplete.py` | Removes entries with missing `user` or `assistant` messages |
| `kaggle_training_notebook.py` | Pre-configured notebook for Kaggle T4×2 training |

---

## Configuration

All training hyperparameters live in `training_config.yaml`. The script reads this at runtime — nothing is hardcoded in `train.py` (except the defaults in the YAML).

Key configurable fields:

```yaml
model.base_model          # swap to Qwen2.5, Phi-4, Mistral-7B, etc.
model.load_in_4bit        # set false for full-precision (needs more VRAM)
lora.r                    # LoRA rank (16 = good default)
training.num_train_epochs # 2 = fast, 3 = better quality
training.learning_rate    # 3e-4 = aggressive but stable
export.save_method        # lora | merged_16bit | merged_4bit | gguf
```

---

## Dependencies

| Library | Role |
|---------|------|
| `torch >= 2.1.0` | Deep learning engine |
| `transformers >= 4.37.0` | Model architecture, tokenizer, chat templates |
| `unsloth` | Fast LoRA training (git install) |
| `peft >= 0.8.0` | LoRA implementation |
| `trl >= 0.7.10` | SFTTrainer for supervised fine-tuning |
| `datasets >= 2.16.0` | JSONL loading and preprocessing |
| `accelerate >= 0.26.0` | Distributed training support |
| `bitsandbytes >= 0.42.0` | 4-bit/8-bit quantization |
| `pyyaml >= 6.0.1` | Config file parsing |
| `numpy, pandas, tqdm` | Utilities |
