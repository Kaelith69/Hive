# Changelog 🐝

All notable changes to Hive are documented here.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning is [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

Things cooking in the lab right now:

- Multi-person dataset support
- Automatic chat partner detection (stop hardcoding `YOUR_NAME`, you know who you are)
- Interactive inference script (chat with your trained self in the terminal, very existential)
- HuggingFace Hub auto-push after training

---

## [1.2.0] — 2026-02-24

### Added
- `fix_dataset.py` — repairs malformed JSONL entries so you don't cry after a 3-hour training run
- `clean_incomplete.py` — removes incomplete conversation pairs before they ruin your loss curve
- `kaggle_training_notebook.py` — pre-configured Kaggle notebook with all the right cells, so you don't have to figure it out at 2 AM
- `KAGGLE_SETUP.md`, `CLOUD_TRAINING.md`, `FAST_TRAINING.md` — actually useful cloud training docs

### Changed
- `train.py` now auto-detects BF16/FP16 support at runtime based on GPU capability — no more silent training failures on T4s
- `train.py` dynamically resolves `evaluation_strategy` vs `eval_strategy` for different `transformers` versions
- `training_config.yaml` tuned for Kaggle T4×2: 2 epochs, batch 4, grad accum 4, LR 3e-4

### Fixed
- `export_model.py` GGUF path handling for Ollama Modelfile instructions

---

## [1.1.0] — 2025-12-01

### Added
- `export_model.py` — multi-format model export: `lora`, `merged_16bit`, `merged_4bit`, `gguf`, `all`
- `validate_dataset.py` — JSONL integrity checker with entry stats and error reporting
- `training_config.yaml` — all hyperparameters in one place, finally
- Support for quantization options: `q4_k_m`, `q4_k_s`, `q5_k_m`, `q8_0`, `f16`

### Changed
- `train.py` refactored to be fully config-driven via YAML (goodbye hardcoded values)
- LoRA target modules expanded to include `gate_proj`, `up_proj`, `down_proj` for better coverage

---

## [1.0.0] — 2025-10-15

### The beginning 🐣

- `extract.py` — first working version: parse WhatsApp exports, filter to your messages only, strip URLs / deleted messages / `<Media omitted>`
- `convert.py` — sliding-window JSONL pair generation, HuggingFace `messages` format
- `train.py` — Unsloth + PEFT LoRA fine-tuning on Llama-3.2-3B-Instruct with 4-bit quantization
- `dataset/final/personality.jsonl` — 21,009 conversation pairs extracted from real chat history
- `wiki/` — initial Home, Usage, DataSpecs pages
- `README.md` — the world's most honest project description

---

*"The first version barely worked. The second version mostly worked. The current version works unless your WhatsApp display name has a colon in it."*
