# 🔮 Roadmap

Things that could make Hive better. Organized by likelihood and how much caffeine it would take to implement.

---

## Near Term (Planned)

These are reasonably scoped and make obvious sense:

### Automatic chat partner detection

Right now you have to set `YOUR_NAME = "Sayanth"` manually. Which works, but breaks if your display name changed mid-conversation history, or if you're running this on someone else's export. A better approach would be auto-detect the most frequent sender name.

### Dataset merge tool

If you have 10 different chat exports, you currently run `extract.py` once and it walks all of them. But there's no clean way to merge the resulting cleaned files before converting. A proper merge utility would handle deduplication and ordering across multiple chat files.

### Interactive inference script

After all this work, there's no easy way to actually *chat* with the trained model in the terminal. A small `chat.py` that loads the exported GGUF or HF model and runs an interactive session would round out the pipeline nicely.

### Quality filtering

Currently all messages above 0 characters get included. A message like "k" is technically training data. A minimum message length filter (configurable, default ~5 chars) and maybe a maximum filter (for when you copy-pasted an entire article) would improve dataset quality.

---

## Medium Term (Being Considered)

Bigger pieces that need more thought:

### Multi-person fine-tuning

What if you want a model that can switch between two people's communication styles? Or blend them? This would require changes to the dataset format (adding speaker identification) and the prompt template. Interesting ML problem. Not trivial.

### HuggingFace Hub auto-push

`export_model.py` already saves to `exports/`. Adding a `--push-to-hub` flag that does `model.push_to_hub("username/model-name")` would let people share their personality models with one command. Needs HF token handling.

### Evaluation metrics

Right now there's no way to objectively measure "does this sound like me?" beyond vibes. Adding BLEU/ROUGE scores on the held-out validation set would at least give a quantitative sanity check. Won't fully capture personality matching, but better than nothing.

### Support for other chat formats

WhatsApp `.txt` exports have a specific format. Telegram exports (JSON), Signal exports, iMessage exports, Discord export scripts — all have different formats. A modular extractor with format-specific parsers would make Hive usable beyond just WhatsApp.

---

## Long Term (Pipe Dreams)

Things that would be cool but require significant effort or dependencies:

### Web UI

A browser-based interface for the pipeline. Upload chats → configure settings → train (on your local GPU) → download model. Click-clack instead of terminal. Would make this accessible to non-programmers. Requires a whole web framework layer.

### Automated hyperparameter search

Run multiple training configurations in sequence and pick the one with the lowest validation loss. Nice to have, but requires multiple GPU-hours per search. More of a "cloud budget" problem than a code problem.

### Persona blending

Train on chats from multiple periods of your life (2018 you vs. 2024 you) and blend them with different weights. Requires significant dataset metadata work.

### Continuous learning

Add new chats to the dataset and fine-tune the existing LoRA adapters without full retraining from scratch. Requires careful data mixing to avoid catastrophic forgetting.

---

## What's Explicitly Not Planned

To set expectations:

- **Any server component** — this is a local tool, not a SaaS product
- **Non-WhatsApp platforms** without a community request and clear format spec
- **Automatic data collection** from messaging platforms — that would be a different (and concerning) project

---

*Have an idea? Open an issue. The roadmap isn't gospel — it's a direction. 🐝*
