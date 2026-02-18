"""
Data validation and preparation script
Checks your personality.jsonl dataset for issues and provides statistics
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import statistics

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Line {i} is invalid JSON: {e}")
    return data

def validate_format(data):
    """Validate data format"""
    issues = []
    valid_count = 0
    
    for i, item in enumerate(data, 1):
        if "messages" not in item:
            issues.append(f"Line {i}: Missing 'messages' field")
            continue
            
        messages = item["messages"]
        if not isinstance(messages, list):
            issues.append(f"Line {i}: 'messages' is not a list")
            continue
        
        if len(messages) < 2:
            issues.append(f"Line {i}: Need at least 2 messages (user + assistant)")
            continue
        
        # Check message structure
        for j, msg in enumerate(messages):
            if "role" not in msg:
                issues.append(f"Line {i}, Message {j}: Missing 'role' field")
            if "content" not in msg:
                issues.append(f"Line {i}, Message {j}: Missing 'content' field")
            if msg.get("role") not in ["user", "assistant", "system"]:
                issues.append(f"Line {i}, Message {j}: Invalid role '{msg.get('role')}'")
        
        valid_count += 1
    
    return valid_count, issues

def analyze_dataset(data):
    """Analyze dataset statistics"""
    total_conversations = len(data)
    message_counts = []
    user_lengths = []
    assistant_lengths = []
    roles = Counter()
    
    for item in data:
        messages = item.get("messages", [])
        message_counts.append(len(messages))
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            roles[role] += 1
            
            if role == "user":
                user_lengths.append(len(content))
            elif role == "assistant":
                assistant_lengths.append(len(content))
    
    stats = {
        "total_conversations": total_conversations,
        "total_messages": sum(message_counts),
        "avg_messages_per_conversation": statistics.mean(message_counts) if message_counts else 0,
        "min_messages": min(message_counts) if message_counts else 0,
        "max_messages": max(message_counts) if message_counts else 0,
        "role_distribution": dict(roles),
        "avg_user_message_length": statistics.mean(user_lengths) if user_lengths else 0,
        "avg_assistant_message_length": statistics.mean(assistant_lengths) if assistant_lengths else 0,
        "median_user_message_length": statistics.median(user_lengths) if user_lengths else 0,
        "median_assistant_message_length": statistics.median(assistant_lengths) if assistant_lengths else 0,
    }
    
    return stats

def print_report(stats, valid_count, issues):
    """Print validation report"""
    print("\n" + "="*70)
    print("📊 DATASET VALIDATION REPORT")
    print("="*70)
    
    print(f"\n✅ Valid Conversations: {valid_count}")
    print(f"📝 Total Messages: {stats['total_messages']}")
    print(f"💬 Average Messages per Conversation: {stats['avg_messages_per_conversation']:.2f}")
    print(f"📏 Message Count Range: {stats['min_messages']} - {stats['max_messages']}")
    
    print(f"\n🗣️  Role Distribution:")
    for role, count in stats['role_distribution'].items():
        print(f"  - {role}: {count}")
    
    print(f"\n📐 Message Length Statistics (characters):")
    print(f"  User Messages:")
    print(f"    - Average: {stats['avg_user_message_length']:.1f}")
    print(f"    - Median: {stats['median_user_message_length']:.1f}")
    print(f"  Assistant Messages:")
    print(f"    - Average: {stats['avg_assistant_message_length']:.1f}")
    print(f"    - Median: {stats['median_assistant_message_length']:.1f}")
    
    if issues:
        print(f"\n⚠️  Issues Found: {len(issues)}")
        print("First 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("\n✅ No issues found! Dataset looks good.")
    
    print("\n" + "="*70)
    
    # Training readiness check
    print("\n🚀 Training Readiness:")
    if valid_count >= 1000:
        print("  ✅ Dataset size is good (>1000 conversations)")
    elif valid_count >= 500:
        print("  ⚠️  Dataset is small but usable (500-1000 conversations)")
    else:
        print("  ❌ Dataset is very small (<500 conversations) - consider adding more data")
    
    if stats['avg_messages_per_conversation'] >= 2:
        print("  ✅ Average conversation length is sufficient")
    else:
        print("  ⚠️  Conversations are very short")
    
    if len(issues) == 0:
        print("  ✅ No format issues detected")
    elif len(issues) < valid_count * 0.01:
        print("  ⚠️  Minor format issues detected (<1% of data)")
    else:
        print("  ❌ Significant format issues detected - fix before training")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Validate personality dataset")
    parser.add_argument("--input", type=str, default="dataset/final/personality.jsonl",
                       help="Path to JSONL dataset")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--sample", type=int, default=0,
                       help="Show N sample conversations")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        return
    
    print(f"📂 Loading dataset from {args.input}...")
    data = load_jsonl(args.input)
    
    if not data:
        print("❌ No data loaded! Check your file.")
        return
    
    print(f"✅ Loaded {len(data)} entries")
    
    print("\n🔍 Validating format...")
    valid_count, issues = validate_format(data)
    
    print("📊 Analyzing statistics...")
    stats = analyze_dataset(data)
    
    print_report(stats, valid_count, issues)
    
    # Show samples if requested
    if args.sample > 0:
        print(f"\n📖 Sample Conversations (showing {min(args.sample, len(data))}):")
        print("="*70)
        for i, item in enumerate(data[:args.sample], 1):
            print(f"\n--- Conversation {i} ---")
            for msg in item.get("messages", []):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                # Truncate long messages
                if len(content) > 100:
                    content = content[:97] + "..."
                print(f"[{role}]: {content}")
        print("="*70)

if __name__ == "__main__":
    main()
