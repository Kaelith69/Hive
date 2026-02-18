"""
Fix corrupted JSONL dataset by removing duplicate keys and invalid entries
"""

import json
import re
from pathlib import Path

def fix_jsonl(input_file, output_file, verbose=False):
    """
    Fix JSONL file by:
    1. Detecting and removing duplicate keys
    2. Removing invalid JSON entries
    3. Keeping valid entries
    """
    
    valid_count = 0
    invalid_count = 0
    fixed_count = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Try to parse as-is first
                entry = json.loads(line)
                valid_count += 1
                fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                # Try to fix duplicate keys by parsing and reconstructing
                try:
                    # Remove duplicate key patterns like "role": "...", "role": "..."
                    # Use regex to handle this
                    fixed_line = line
                    
                    # Pattern: "key": value, {more}, "key": value
                    # Replace duplicate keys with just the first occurrence
                    
                    # Parse the messages array part specifically
                    messages_match = re.search(r'"messages"\s*:\s*\[(.*)\]', fixed_line)
                    if messages_match:
                        messages_str = messages_match.group(1)
                        
                        # Try to fix by removing duplicate role/content entries
                        # This is a simple heuristic - look for patterns like },"role":...
                        messages_str = re.sub(r'},\s*"(role|content)":\s*"([^"]*)",\s*"(role|content)"', 
                                             r'},"\1":"\2"', messages_str)
                        
                        fixed_line = fixed_line[:messages_match.start(1)] + messages_str + fixed_line[messages_match.end(1):]
                    
                    # Try to parse the fixed version
                    entry = json.loads(fixed_line)
                    
                    # Validate structure
                    if "messages" in entry and isinstance(entry["messages"], list):
                        if all("role" in msg and "content" in msg for msg in entry["messages"]):
                            fixed_count += 1
                            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            if verbose:
                                print(f"✅ Fixed line {line_num}")
                        else:
                            invalid_count += 1
                            if verbose:
                                print(f"❌ Line {line_num}: Invalid message structure")
                    else:
                        invalid_count += 1
                        if verbose:
                            print(f"❌ Line {line_num}: Missing or invalid messages field")
                
                except Exception as fix_error:
                    invalid_count += 1
                    if verbose and line_num in [1393, 1394, 1395]:
                        print(f"❌ Line {line_num}: Could not fix: {fix_error}")
                        print(f"   Content: {line[:100]}...")
    
    return valid_count, fixed_count, invalid_count

def main():
    input_file = "dataset/final/personality.jsonl"
    output_file = "dataset/final/personality_fixed.jsonl"
    
    print(f"🔧 Fixing corrupted JSONL file: {input_file}")
    print("=" * 70)
    
    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return
    
    valid, fixed, invalid = fix_jsonl(input_file, output_file, verbose=True)
    
    print("\n" + "=" * 70)
    print("📊 Results:")
    print(f"  ✅ Valid entries (kept as-is): {valid}")
    print(f"  🔧 Fixed entries: {fixed}")
    print(f"  ❌ Invalid entries (removed): {invalid}")
    print(f"  📝 Total entries saved: {valid + fixed}")
    print("=" * 70)
    
    # Backup original and replace
    import shutil
    print(f"\n💾 Saving backup: {input_file}.bak")
    shutil.copy(input_file, f"{input_file}.bak")
    
    print(f"📋 Replacing original with fixed version...")
    shutil.move(output_file, input_file)
    
    print(f"✅ Done! Fixed file saved to {input_file}")
    print(f"\n🚀 You can now train without JSON errors!")

if __name__ == "__main__":
    main()
