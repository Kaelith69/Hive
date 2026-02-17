# Hive — Personality Fine-tuning Dataset

📂 Project: transform WhatsApp exports into fine-tuning-ready conversational pairs.

**What this repo contains**
- `dataset/raw/` — original exports (kept as-is)
- `dataset/cleaned/` — messages from *you* after filtering (deleted/media/URLs removed)
- `dataset/final/personality.jsonl` — final JSONL ready for fine-tuning (one JSON per line)
- `extract.py` — extracts only your messages and cleans them
- `convert.py` — converts cleaned messages into JSONL conversational pairs
- `REPORT.md` — detailed, colorful, funny project report (read this first)
- `wiki/` — markdown pages you can import into GitHub wiki

Quick stats (from this run):
- Lines in `dataset/cleaned/whatsapp.txt`: ~21k
- Entries in `dataset/final/personality.jsonl`: 21,009

Getting started

1. Inspect raw chat(s) in `dataset/raw/` (do not share private data publicly).
2. Set your WhatsApp display name in `extract.py` (`YOUR_NAME` variable).
3. Run extractor then converter:

```powershell
python extract.py
python convert.py
```

4. Validate JSONL count:

```powershell
(Get-Content .\dataset\final\personality.jsonl | Measure-Object -Line).Lines
```

How to publish to GitHub (example):

```bash
git init
git add .
git commit -m "Add dataset and pipeline"
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

To publish the wiki (manual step):
- Clone the wiki repo and copy files from `wiki/`:

```bash
git clone git@github.com:<your-user>/<your-repo>.wiki.git
cd <your-repo>.wiki
cp -r ../wiki/* .
git add .
git commit -m "Add wiki pages"
git push
```

License & privacy

- This repo may contain personal conversations — remove or anonymize sensitive data before pushing publically.

Have fun 👋 — see `REPORT.md` for the full circus 🎪 and diagrams.