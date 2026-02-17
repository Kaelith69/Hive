import json
import os

INPUT_DIR = "dataset/cleaned"
OUTPUT_FILE = "dataset/final/personality.jsonl"

os.makedirs("dataset/final", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for file in os.listdir(INPUT_DIR):

        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:

            lines = [line.strip() for line in f if line.strip()]

            for i in range(len(lines) - 1):

                entry = {
                    "messages": [
                        {"role": "user", "content": lines[i]},
                        {"role": "assistant", "content": lines[i+1]}
                    ]
                }

                out.write(json.dumps(entry) + "\n")
