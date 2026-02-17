# DataSpecs

## Raw format
WhatsApp exported plain-text lines typically look like:

```
12/03/24, 10:32 pm - Friend: hello
```

## Extracted fields
- `name` — the speaker name captured by regex
- `msg` — the message text

## Filters applied by `extract.py`
- Remove system messages: `You deleted this message`, `<Media omitted>`
- Strip URLs: `https?://...` and `www...`
- Only keep lines where `name == YOUR_NAME`
- Skip empty results

## Final encoding
Each JSON line is:

```json
{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Example
See `dataset/final/personality.jsonl` for real examples.


Thank you for using Hive — keep your data safe!