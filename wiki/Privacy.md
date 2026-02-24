# 🔒 Privacy

Hive processes personal chat data. This page is the paranoia checklist you'll thank yourself for reading.

---

## What Data Does Hive Handle?

| Data | Where it lives | Sensitive? |
|------|---------------|------------|
| WhatsApp `.txt` exports | `dataset/raw/` | ⚠️ Yes — your private conversations |
| Cleaned message lines | `dataset/cleaned/` | ⚠️ Yes — your words, filtered |
| JSONL conversational pairs | `dataset/final/personality.jsonl` | ⚠️ Yes — your communication style |
| Training checkpoints | `outputs/` | 🟡 Contains learned patterns from your messages |
| Exported model weights | `exports/` | 🟡 Encodes your communication style |

---

## Where Does Your Data Go?

**Nowhere you don't explicitly send it.**

- All processing runs locally (on your machine or your cloud account)
- No data is uploaded to any external server by Hive's scripts
- `report_to: "none"` is the default in `training_config.yaml` — W&B and TensorBoard are disabled out of the box
- The only network calls are model downloads from HuggingFace Hub when you first run training (the base model weights, not your data)

If you enable Weights & Biases (`report_to: "wandb"` in the config), training metrics (loss curves, step counts) will be sent to Weights & Biases' servers. That's metrics only — not your dataset content. But it's worth knowing.

---

## What's Gitignored By Default

The `.gitignore` excludes these by default so they don't accidentally get committed:

```
dataset/raw/        ← your original WhatsApp exports
dataset/cleaned/    ← your filtered messages
outputs/            ← model checkpoints
exports/            ← trained model files
*.gguf              ← GGUF model files
*.safetensors       ← HuggingFace model files
*.bin               ← legacy model files
wandb/              ← W&B logs
```

**`dataset/final/personality.jsonl` is NOT gitignored by default.** This is intentional — it's the output artifact of the pipeline. But it contains your words. Review it before committing.

---

## Before Making This Repo Public

Go through this checklist:

- [ ] **Delete `dataset/raw/`** — never push original WhatsApp exports publicly. They may contain other people's messages.
- [ ] **Delete `dataset/cleaned/`** — your filtered messages in plaintext.
- [ ] **Review `personality.jsonl`** — open it and scroll through. Would you be okay if this appeared in a web search? If not, delete it or add it to `.gitignore`.
- [ ] **Check `exports/`** is empty or excluded.
- [ ] **Check `outputs/`** is empty or excluded.
- [ ] Run `git status` and make sure no sensitive files are staged.

**The golden rule:** if it contains other people's messages (even in conversation context), don't push it publicly without their knowledge.

---

## Other People's Messages

The pipeline only *keeps* your messages (the `YOUR_NAME` filter in `extract.py`). But the raw export files in `dataset/raw/` contain the full conversations — including what others said. Don't commit `dataset/raw/`.

Even `personality.jsonl` uses your previous message as "user" input — which might technically contain context from a conversation. Review accordingly.

---

## Model Privacy

The trained model (LoRA adapters or exported GGUF) has learned patterns from your messages. Sharing the model publicly is essentially sharing a statistical summary of how you communicate. Not the raw messages, but arguably more interesting to an adversary. Decide accordingly.

---

## Local Cloud Accounts (Kaggle / Colab / RunPod)

When you train on Kaggle or Colab, your dataset is uploaded to their platform temporarily. Review their privacy policies:

- [Kaggle Privacy Policy](https://www.kaggle.com/privacy)
- [Google Colab Terms](https://research.google.com/colaboratory/faq.html)
- [RunPod Privacy Policy](https://www.runpod.io/privacy-policy)

Use private datasets on Kaggle, and don't leave sensitive data in public Colab notebooks.

---

## Summary

> **Short version:** Your data stays on your machine unless you push it to GitHub or enable external logging. The scripts don't phone home. Just don't commit your raw chats. You'll be fine. 🐝
