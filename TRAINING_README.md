# 🧠 Hive - Personality LLM Training System

Train your own AI personality model using your WhatsApp chat data! This repository provides a complete pipeline from data extraction to fine-tuned LLM deployment.

## 🎯 What This Does

Transforms your WhatsApp conversations into a fine-tuned language model that mimics your personality and communication style.

**Pipeline**: WhatsApp Export → Data Cleaning → JSONL Dataset → LoRA Fine-tuning → Deployable Model

## 📊 Dataset Stats

- **Total Conversations**: 21,220 pairs
- **Format**: JSONL with user/assistant messages
- **Ready for**: LLaMA, Mistral, Qwen, and other instruction-tuned models

## 🚀 Quick Start

### Option 1: Automated Setup

```powershell
# Run the setup script
.\setup.ps1
```

### Option 2: Manual Setup

1. **Validate your dataset**:
   ```powershell
   python validate_dataset.py --input dataset/final/personality.jsonl --sample 3
   ```

2. **Install dependencies** (see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)):
   ```powershell
   # Create virtual environment
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Install Unsloth (fast fine-tuning)
   pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
   
   # Install other requirements
   pip install transformers datasets accelerate peft trl bitsandbytes pyyaml
   ```

3. **Configure training** (optional):
   Edit [training_config.yaml](training_config.yaml) to customize:
   - Base model selection
   - Training hyperparameters
   - LoRA configuration
   - Export settings

4. **Start training**:
   ```powershell
   python train.py
   ```

5. **Export your model**:
   ```powershell
   # For Ollama (recommended)
   python export_model.py --input outputs/final_model --type gguf --name hive-personality
   
   # For Python/HuggingFace
   python export_model.py --input outputs/final_model --type merged_16bit --name hive-personality
   ```

## 📁 Project Structure

```
Hive/
├── dataset/
│   ├── raw/                    # Original WhatsApp exports
│   ├── cleaned/                # Processed messages
│   └── final/
│       └── personality.jsonl   # Training dataset (21k+ pairs)
├── training_config.yaml        # Training configuration
├── train.py                    # Main training script
├── export_model.py            # Model export utility
├── validate_dataset.py        # Dataset validation tool
├── extract.py                 # Extract messages from WhatsApp
├── convert.py                 # Convert to JSONL format
├── setup.ps1                  # Automated setup script
├── TRAINING_GUIDE.md          # Comprehensive training guide
├── REPORT.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## 🎓 Training Details

### Default Configuration

- **Base Model**: Llama-3.2-3B-Instruct (fast, efficient)
- **Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (fits on 8GB VRAM)
- **Epochs**: 3
- **Batch Size**: 2 (effective: 8 with gradient accumulation)

### Hardware Requirements

| Config | VRAM | Training Time* |
|--------|------|---------------|
| Minimum | 8GB | ~6-8 hours |
| Recommended | 16GB+ | ~2-4 hours |

*For 21k samples on 3B model

### Supported Models

- ✅ Llama 3.2 (3B, 8B)
- ✅ Mistral (7B)
- ✅ Qwen 2.5 (3B, 7B)
- ✅ Phi-4
- ✅ Any instruction-tuned model compatible with Unsloth

## 📦 Export Formats

| Format | Use Case |
|--------|----------|
| **GGUF** | Ollama, llama.cpp, local deployment |
| **HuggingFace** | Python inference, API deployment |
| **LoRA** | Share adapters, minimal size |
| **4-bit Merged** | Memory-efficient deployment |

## 🛠️ Scripts Reference

### validate_dataset.py
Validate your dataset format and show statistics.

```powershell
python validate_dataset.py --input dataset/final/personality.jsonl --sample 3
```

**Output**:
- Total conversations count
- Message length statistics
- Format validation
- Sample conversations

### train.py
Main training script using Unsloth for efficient fine-tuning.

```powershell
# Basic training
python train.py

# Custom config
python train.py --config my_config.yaml

# Resume from checkpoint
python train.py --resume outputs/checkpoint-500
```

### export_model.py
Export trained model to various formats.

```powershell
# Export to GGUF (Ollama)
python export_model.py --input outputs/final_model --type gguf --quantization q4_k_m

# Export to HuggingFace format
python export_model.py --input outputs/final_model --type merged_16bit

# Export all formats
python export_model.py --input outputs/final_model --type all
```

## 🎯 Usage Examples

### Using with Ollama

```powershell
# After exporting to GGUF
cd exports/hive-personality.gguf

# Create Modelfile
@"
FROM ./hive-personality-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"@ | Out-File -Encoding UTF8 Modelfile

# Create model
ollama create hive-personality -f Modelfile

# Run
ollama run hive-personality "Hey, what's up?"
```

### Using with Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("exports/hive-personality_16bit")
tokenizer = AutoTokenizer.from_pretrained("exports/hive-personality_16bit")

# Generate response
messages = [{"role": "user", "content": "Hello!"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Complete training guide with troubleshooting
- **[REPORT.md](REPORT.md)**: Detailed project report and methodology
- **[training_config.yaml](training_config.yaml)**: Configuration file with comments

## ⚙️ Configuration Options

Edit `training_config.yaml` to customize:

**Model Settings**:
- Base model selection
- Sequence length
- Quantization options

**Training Hyperparameters**:
- Learning rate
- Batch size
- Number of epochs
- Optimizer settings

**LoRA Configuration**:
- Rank (r)
- Alpha
- Target modules
- Dropout

**Export Options**:
- Save format
- Quantization method
- Output directory

## 🐛 Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 1024 or 512
- Use smaller base model (3B instead of 7B)

### Training Loss Not Decreasing
- Lower learning rate to `1.0e-4`
- Increase warmup steps
- Check dataset quality

### Model Outputs Nonsense
- Train for more epochs
- Increase LoRA rank
- Validate dataset format

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed troubleshooting.

## 🤝 Contributing

This is a personal project, but feel free to:
- Fork and customize for your data
- Submit improvements
- Share your results

## ⚠️ Privacy Notice

Your dataset contains personal conversations. Before sharing:
- Remove sensitive information
- Anonymize names and personal details
- Don't upload raw WhatsApp exports

## 📄 License

See repository for license information.

---

## 🎓 Learning Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ollama Docs](https://ollama.ai/docs)

---

**Made with ❤️ for personality preservation**

For detailed instructions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
