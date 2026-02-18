# 🚀 Hive Personality LLM Training Guide

Complete guide to fine-tune a language model on your personality dataset using modern techniques (LoRA/QLoRA).

## 📋 Table of Contents
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Configuration](#configuration)
- [Training Options](#training-options)
- [Export & Deployment](#export--deployment)
- [Troubleshooting](#troubleshooting)

---

## 🛠️ Requirements

### Hardware Requirements
- **Minimum**: 16GB RAM, GPU with 8GB VRAM (RTX 3060/4060 or better)
- **Recommended**: 32GB RAM, GPU with 16GB+ VRAM (RTX 4070 Ti, 4080, or better)
- **Cloud Alternative**: Google Colab Pro, Kaggle, or RunPod

### Software Requirements
- Python 3.10 or 3.11
- CUDA 12.1 or higher (for GPU training)
- Git

---

## ⚡ Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (fast fine-tuning library)
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

# Install other requirements
pip install transformers datasets accelerate peft trl bitsandbytes pyyaml
```

### 2. Validate Your Dataset

```powershell
python validate_dataset.py --input dataset/final/personality.jsonl --sample 3
```

This will check your dataset for issues and show statistics.

### 3. Start Training

```powershell
# Basic training with default settings
python train.py

# Custom configuration
python train.py --config training_config.yaml
```

### 4. Export Model

```powershell
# Export to GGUF format (for Ollama/llama.cpp)
python export_model.py --input outputs/final_model --type gguf --name hive-personality

# Export to HuggingFace format
python export_model.py --input outputs/final_model --type merged_16bit --name hive-personality

# Export all formats
python export_model.py --input outputs/final_model --type all --name hive-personality
```

---

## 📖 Step-by-Step Guide

### Step 1: Prepare Your Environment

1. **Check CUDA availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

2. **Verify dataset**:
   ```powershell
   python validate_dataset.py
   ```
   
   Your dataset should have:
   - ✅ At least 500-1000 conversation pairs
   - ✅ Proper JSONL format with "messages" field
   - ✅ User and assistant roles in each conversation

### Step 2: Configure Training

Edit `training_config.yaml` to customize:

**Key parameters to adjust**:

- **Base Model**: Change `model.base_model` to your preferred model
  ```yaml
  base_model: "unsloth/Llama-3.2-3B-Instruct"  # Fast, efficient
  # Or: "unsloth/Mistral-7B-Instruct-v0.3"       # More powerful
  # Or: "unsloth/Qwen2.5-3B-Instruct"            # Good for non-English
  ```

- **Training Duration**: Adjust epochs based on dataset size
  ```yaml
  num_train_epochs: 3  # Increase to 5-10 for small datasets
  ```

- **Batch Size**: Adjust based on your GPU memory
  ```yaml
  per_device_train_batch_size: 2  # Decrease to 1 if out of memory
  gradient_accumulation_steps: 4   # Increase to 8 for smaller batch size
  ```

- **Learning Rate**: Fine-tune for your data
  ```yaml
  learning_rate: 2.0e-4  # Lower for more stable training (1e-4)
  ```

### Step 3: Train the Model

```powershell
# Start training
python train.py

# Resume from checkpoint if interrupted
python train.py --resume outputs/checkpoint-100
```

**Training will**:
- Load the base model in 4-bit quantization
- Apply LoRA adapters (only trains ~1% of parameters)
- Save checkpoints every 100 steps
- Show training progress and loss

**Expected training time**:
- 21k samples on RTX 4090: ~2-4 hours
- 21k samples on RTX 3060: ~6-8 hours
- Depends on model size and batch size

### Step 4: Monitor Training

Watch the training loss in the terminal. Good training shows:
- ✅ Loss decreasing steadily
- ✅ Converges to < 1.0 for conversational data
- ⚠️ If loss < 0.1: May be overfitting
- ⚠️ If loss stuck > 2.0: Try lower learning rate

### Step 5: Export Your Model

After training completes, export to your preferred format:

#### For Ollama (Recommended)

```powershell
python export_model.py --input outputs/final_model --type gguf --quantization q4_k_m
```

Then create Modelfile:
```modelfile
FROM ./exports/hive-personality.gguf/hive-personality-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
```

Create in Ollama:
```powershell
cd exports/hive-personality.gguf
ollama create hive-personality -f Modelfile
ollama run hive-personality
```

#### For Python/Transformers

```powershell
python export_model.py --input outputs/final_model --type merged_16bit
```

Use in Python:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exports/hive-personality_16bit")
tokenizer = AutoTokenizer.from_pretrained("exports/hive-personality_16bit")

messages = [{"role": "user", "content": "Hello!"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

---

## ⚙️ Configuration

### Model Selection

| Model | Size | VRAM | Speed | Quality | Best For |
|-------|------|------|-------|---------|----------|
| Llama-3.2-3B-Instruct | 3B | 6GB | Fast | Good | Quick training, limited GPU |
| Qwen2.5-3B-Instruct | 3B | 6GB | Fast | Good | Multilingual, code |
| Mistral-7B-Instruct | 7B | 12GB | Medium | Better | More context, better quality |
| Llama-3.1-8B-Instruct | 8B | 14GB | Medium | Better | Balanced performance |

### LoRA Configuration

**For better quality** (slower training):
```yaml
lora:
  r: 32              # Higher rank = more parameters
  lora_alpha: 32     # Match with r
  lora_dropout: 0.05
```

**For faster training** (lower quality):
```yaml
lora:
  r: 8               # Lower rank = fewer parameters
  lora_alpha: 16     # 2x r for more stable training
  lora_dropout: 0.1
```

### Training Hyperparameters

**Small dataset (<5k samples)**:
```yaml
training:
  num_train_epochs: 5
  learning_rate: 3.0e-4
  warmup_steps: 10
```

**Large dataset (>20k samples)**:
```yaml
training:
  num_train_epochs: 2
  learning_rate: 2.0e-4
  warmup_steps: 50
```

**Out of memory?**:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 1024  # Also reduce in model section
```

---

## 📦 Export & Deployment

### Export Formats

| Format | Size | Use Case | Command |
|--------|------|----------|---------|
| **LoRA only** | Small | Share adapters, apply to base model | `--type lora` |
| **Merged 16-bit** | Large | Python inference, upload to HF | `--type merged_16bit` |
| **Merged 4-bit** | Medium | Memory-constrained inference | `--type merged_4bit` |
| **GGUF** | Medium | Ollama, llama.cpp, local use | `--type gguf` |

### Quantization Methods (GGUF)

| Method | Size | Speed | Quality |
|--------|------|-------|---------|
| **q4_k_m** | Medium | Fast | Good (recommended) |
| q4_k_s | Smaller | Faster | Slightly lower |
| q5_k_m | Larger | Medium | Better |
| q8_0 | Large | Slower | Best |
| f16 | Largest | Slowest | Original |

### Deploy to Ollama

```powershell
# 1. Export to GGUF
python export_model.py --input outputs/final_model --type gguf --quantization q4_k_m

# 2. Create Modelfile (see Step 5 above)

# 3. Create in Ollama
cd exports/hive-personality.gguf
ollama create hive-personality -f Modelfile

# 4. Test
ollama run hive-personality "Hey, what's up?"
```

---

## 🐛 Troubleshooting

### Out of Memory Errors

**Symptom**: `CUDA out of memory` error during training

**Solutions**:
1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 8
   ```

2. Reduce sequence length:
   ```yaml
   max_seq_length: 1024  # or even 512
   ```

3. Use smaller model or enable more aggressive quantization

### Training Loss Not Decreasing

**Symptom**: Loss stays high (>2.0) or increases

**Solutions**:
1. Lower learning rate:
   ```yaml
   learning_rate: 1.0e-4
   ```

2. Increase warmup:
   ```yaml
   warmup_steps: 20
   ```

3. Check dataset quality with `validate_dataset.py`

### Model Outputs Gibberish

**Symptom**: Trained model produces nonsensical text

**Solutions**:
1. Train for more epochs (underfitting):
   ```yaml
   num_train_epochs: 5
   ```

2. Check if you're using the correct chat template

3. Validate dataset format - ensure messages are in correct order

### Export Fails

**Symptom**: Error during model export

**Solutions**:
1. Check available disk space (GGUF exports need significant temp space)
2. Try exporting to `lora` first, then `merged_16bit`
3. Update Unsloth: `pip install --upgrade "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"`

### Slow Training

**Symptom**: Training is very slow

**Solutions**:
1. Check you're using GPU: `torch.cuda.is_available()` should be `True`
2. Use Unsloth-optimized model (models with `unsloth/` prefix)
3. Enable packing for short conversations (not recommended for conversational data though)

---

## 📊 Performance Tips

### Speed Up Training
- Use `gradient_checkpointing` (already enabled with Unsloth)
- Use `bf16=true` instead of `fp16=true` on modern GPUs
- Use `group_by_length=true` to batch similar sequence lengths

### Improve Quality
- Clean your dataset (remove duplicates, fix formatting)
- Increase LoRA rank (r=32 or r=64)
- Train for more epochs on small datasets
- Use larger base model if you have VRAM

### Reduce Overfitting
- Add more validation data (increase `validation_split`)
- Use dropout (increase `lora_dropout`)
- Reduce training epochs
- Use weight decay: `weight_decay: 0.01`

---

## 🎯 Next Steps

After successful training:

1. **Test thoroughly**: Chat with your model extensively
2. **Iterate**: Adjust hyperparameters based on results
3. **Share**: Upload to HuggingFace Hub
4. **Deploy**: Use in applications via Ollama or Python

---

## 📚 Additional Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Ollama Documentation](https://ollama.ai/docs)

---

## 🆘 Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Validate your dataset with `python validate_dataset.py`
3. Search existing issues on Unsloth GitHub
4. Ask in relevant Discord communities (Ollama, HuggingFace)

---

**Happy Training! 🚀**
