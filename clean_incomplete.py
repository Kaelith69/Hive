"""
Remove incomplete conversation entries from JSONL
"""

import json
from pathlib import Path

def clean_incomplete_entries(input_file, output_file):
    """Remove entries that don't have proper user+assistant pairs"""
    
    valid_count = 0
    removed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # Check if messages field exists and has at least 2 messages
                if "messages" in entry and isinstance(entry["messages"], list):
                    messages = entry["messages"]
                    
                    # Must have at least 2 messages (user + assistant)
                    if len(messages) >= 2:
                        # Check that roles are valid
                        has_user = any(m.get("role") == "user" for m in messages)
                        has_assistant = any(m.get("role") == "assistant" for m in messages)
                        
                        if has_user and has_assistant:
                            valid_count += 1
                            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        else:
                            removed_count += 1
                    else:
                        removed_count += 1
                else:
                    removed_count += 1
                    
            except json.JSONDecodeError:
                removed_count += 1
    
    return valid_count, removed_count

def main():
    input_file = "dataset/final/personality.jsonl"
    output_file = "dataset/final/personality_cleaned.jsonl"
    
    print(f"🧹 Cleaning incomplete entries from {input_file}")
    print("=" * 70)
    
    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return
    
    valid, removed = clean_incomplete_entries(input_file, output_file)
    
    print(f"\n📊 Results:")
    print(f"  ✅ Complete conversations kept: {valid}")
    print(f"  ❌ Incomplete entries removed: {removed}")
    print("=" * 70)
    
    # Replace original with cleaned version
    import shutil
    print(f"\n💾 Replacing original file...")
    shutil.move(output_file, input_file)
    
    print(f"✅ Done! Clean dataset with {valid} entries ready for training")
    print(f"\n🚀 Dataset is now perfectly clean!")

if __name__ == "__main__":
    main()
