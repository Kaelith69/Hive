# 🐝 Hive Wiki — Home

Welcome to the Hive wiki. You found the documentation that lives behind the documentation. Very meta.

**Hive** is a WhatsApp-to-LLM personality fine-tuning pipeline. You give it your chats. It gives back a dataset. You train a model. You now have a language model that talks like you. The existential implications are your problem.

---

## 📚 Wiki Pages

| Page | What it covers |
|------|----------------|
| [Architecture](Architecture) | How the pipeline is structured, what each script does, LoRA explained like you're five |
| [Installation](Installation) | Getting the thing running — local, Kaggle, Colab, RunPod |
| [Usage](Usage) | The full pipeline walkthrough from chat export to model export |
| [Privacy](Privacy) | What data goes where, what to do before going public |
| [Troubleshooting](Troubleshooting) | When things go sideways (and they will) |
| [Roadmap](Roadmap) | What's coming next, what's being considered, what's a pipe dream |
| [DataSpecs](DataSpecs) | JSONL format, input format, filtering rules |

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Set YOUR_NAME in extract.py
# 2. Drop WhatsApp .txt exports in dataset/raw/
# 3. Run the pipeline:
python extract.py
python convert.py
python validate_dataset.py
python train.py
python export_model.py --input outputs/final_model --type gguf
```

That's it. Your personality is now a `.gguf` file.

---

## 🏗️ What This Actually Does

```
WhatsApp .txt
     │
     ▼ extract.py
dataset/cleaned/   ← just your messages, cleaned
     │
     ▼ convert.py
dataset/final/personality.jsonl  ← 21,009 conv pairs
     │
     ▼ train.py  (Unsloth + LoRA + 4-bit QLoRA)
outputs/final_model/
     │
     ▼ export_model.py
exports/hive-personality.gguf  ← run in Ollama
```

---

*Enjoy the memes. Take the privacy section seriously. 🐝*