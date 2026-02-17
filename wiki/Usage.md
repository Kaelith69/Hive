# Usage

1. Edit `extract.py` and set `YOUR_NAME` to your WhatsApp display name.
2. Run:

```powershell
python extract.py
python convert.py
```

3. Inspect `dataset/cleaned/` and `dataset/final/personality.jsonl`.

Tips:
- If you have multiple chats, `extract.py` walks directories recursively.
- To change filters (e.g., keep URLs), edit `extract.py`.
