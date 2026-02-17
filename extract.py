import os
import re

INPUT_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/cleaned"
YOUR_NAME = "Sayanth"   # replace with your WhatsApp name

os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = re.compile(r".* - (.*?): (.*)")

# Walk through INPUT_DIR recursively to find .txt files inside subfolders
for root, dirs, files in os.walk(INPUT_DIR):
    for fname in files:
        if fname.endswith(".txt"):
            src_path = os.path.join(root, fname)
            with open(src_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # filter out system messages like deleted/media and strip URLs
            IGNORED = {"You deleted this message", "<Media omitted>"}
            url_pattern = re.compile(r"https?://\S+|www\.\S+")
            extracted = []
            for line in lines:
                match = pattern.match(line)
                if not match:
                    continue
                name, msg = match.groups()
                if name != YOUR_NAME:
                    continue
                msg_text = msg.strip()
                if not msg_text:
                    continue
                if msg_text in IGNORED:
                    continue
                # also defensively skip lines that mention deletion or media
                if "deleted this message" in msg_text.lower():
                    continue
                if msg_text.startswith("<Media omitted>"):
                    continue
                # remove URLs from the message text
                msg_text = url_pattern.sub("", msg_text).strip()
                # skip if message becomes empty after removing URLs
                if not msg_text:
                    continue
                extracted.append(msg_text)

            out_path = os.path.join(OUTPUT_DIR, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                for msg in extracted:
                    f.write(msg + "\n")
