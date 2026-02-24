# 🔧 Troubleshooting

When things go wrong (and they will), this is the page you want. Organized by stage because that's how you'll find it at 2 AM.

---

## Installation Issues

### `unsloth` install fails

Unsloth requires specific CUDA/PyTorch version combinations. The generic `pip install unsloth` usually fails.

**Fix:** Use the versioned install command matching your setup:

```bash
# Check your PyTorch + CUDA versions first:
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Then install the matching variant:
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
# Replace cu121 with your CUDA version, torch230 with your PyTorch version
```

See the [Unsloth GitHub](https://github.com/unslothai/unsloth) for the full version matrix.

---

### `bitsandbytes` errors on Windows

bitsandbytes has historically been Linux-only. On Windows, you may see CUDA load errors.

**Fix:** Use WSL2 (Windows Subsystem for Linux) for GPU training on Windows.

---

### `CUDA out of memory`

```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Fix options (pick one or combine):**
1. Reduce `per_device_train_batch_size` to `2` or `1` in `training_config.yaml`
2. Reduce `max_seq_length` from `2048` to `1024`
3. Make sure `load_in_4bit: true` is set
4. Close other GPU processes (`nvidia-smi` to check what's using VRAM)

---

## Data Pipeline Issues

### `extract.py` finds 0 messages

Your `YOUR_NAME` doesn't match exactly.

**Fix:** Open your `.txt` export file and copy your display name character for character. It's case-sensitive and space-sensitive. Check for invisible Unicode characters if you copy-pasted from WhatsApp.

---

### `personality.jsonl` is empty or tiny

Likely no matching messages were found, or `convert.py` had nothing to work with.

**Debug:**
```bash
# Check if cleaned/ has content:
ls -la dataset/cleaned/
wc -l dataset/cleaned/*.txt

# Check convert output:
wc -l dataset/final/personality.jsonl
```

If `cleaned/` is empty → fix `YOUR_NAME` in `extract.py`.
If `cleaned/` has content but `personality.jsonl` is small → check for encoding issues in the source files.

---

### `validate_dataset.py` reports errors

Common errors:

- `Invalid JSON on line N` → Run `fix_dataset.py`
- `Missing 'messages' key` → Run `fix_dataset.py`
- `Incomplete entry (missing user or assistant)` → Run `clean_incomplete.py`
- `Empty content field` → Run `clean_incomplete.py`

Always re-run `validate_dataset.py` after running the fix/clean scripts to confirm.

---

### Encoding errors when reading .txt files

```
UnicodeDecodeError: 'utf-8' codec can't decode ...
```

WhatsApp exports from some regions / devices may use different encodings.

**Fix:** Open `extract.py` and change `encoding="utf-8"` to `encoding="utf-8-sig"` or `encoding="latin-1"` depending on the file.

---

## Training Issues

### Training loss is NaN

```
{'loss': nan, 'learning_rate': ..., 'epoch': ...}
```

**Causes and fixes:**
- Learning rate too high → reduce `learning_rate` to `1e-4` or `2e-4`
- Bad data in dataset → run `validate_dataset.py` and `clean_incomplete.py`
- `max_grad_norm` too high → try `0.1`

---

### Training loss is not decreasing

After the first few hundred steps, loss should trend downward. If it plateaus immediately:

- Increase epochs from 2 to 3
- Try a higher learning rate (e.g., `5e-4`)
- Make sure your dataset has enough variety (lots of short repetitive messages won't help)

---

### `BF16 requested but GPU does not support it`

This is a warning, not an error. `train.py` automatically falls back to FP16. You can safely ignore it, or set `bf16: false` in the config explicitly.

---

### Training is extremely slow

- Make sure you have GPU enabled: `python -c "import torch; print(torch.cuda.is_available())"`
- Unsloth not properly installed → reinstall with the correct versioned command
- `group_by_length: true` helps speed — make sure it's set
- On Kaggle: use T4 × 2, not CPU/TPU

---

### Checkpoint not found on resume

```
❌ No checkpoint found at outputs/checkpoint-500
```

Checkpoints save every `save_steps` (default: 500). If training crashed before step 500, no checkpoint was saved.

**Fix:** Start training from scratch, or reduce `save_steps` to `100` for shorter checkpoint intervals.

---

## Export Issues

### `❌ Model path not found`

You need to complete training before exporting. The trained model should be at `outputs/final_model`.

```bash
ls outputs/final_model/  # Check it exists
```

---

### GGUF file not appearing where expected

GGUF files are saved to `exports/<output_name>.gguf/`. The Modelfile path should point to the `.gguf` file inside that directory.

```bash
find exports/ -name "*.gguf"  # Find the actual file
```

---

### Ollama refuses to load the model

- Verify the GGUF file is complete (not truncated — check file size is > 1 GB for a 3B model)
- Make sure your Ollama version supports the model architecture
- Check the Modelfile path is correct (absolute path is safer)

---

## General Tips

- **Always validate before training.** `python validate_dataset.py` takes 5 seconds and saves hours.
- **More data is usually better.** If results are poor, collect more chats.
- **Short messages cause noise.** One-word replies as training targets teach the model to be monosyllabic.
- **Epoch 2 is fast but epoch 3 is usually noticeably better.** Trade off based on your GPU time budget.

---

Still stuck? Open an issue on [GitHub](https://github.com/Kaelith69/Hive/issues). Include the error message, your Python/CUDA/GPU version, and which step failed. 🐝
