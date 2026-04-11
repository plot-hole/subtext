"""
Step 1: Extract — Concatenate split JSON conversation files into a single canonical file.
Adapted for pre-extracted split JSON files (conversations-000.json through conversations-015.json).
"""
import json
import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT.parent / "conversations"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RAW_DIR / "conversations.json"


def main(source_dir: Path = CONVERSATIONS_DIR):
    print(f"=== Step 1: Extract ===")
    print(f"Source directory: {source_dir}")

    # Find all conversation JSON files, sorted
    json_files = sorted(source_dir.glob("conversations-*.json"))
    if not json_files:
        # Fallback: any .json file
        json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No JSON files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files")

    all_conversations = []
    total_bytes = 0

    for fp in json_files:
        size = fp.stat().st_size
        total_bytes += size
        print(f"  Loading {fp.name} ({size / 1024 / 1024:.1f} MB)...")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_conversations.extend(data)
        elif isinstance(data, dict):
            all_conversations.append(data)
        else:
            print(f"  WARNING: Unexpected type {type(data)} in {fp.name}, skipping")

    print(f"\nConcatenation complete:")
    print(f"  Files processed:        {len(json_files)}")
    print(f"  Total source size:      {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Total conversations:    {len(all_conversations)}")

    # Validate structure
    if not all_conversations:
        print("ERROR: No conversations found!")
        sys.exit(1)

    sample = all_conversations[0]
    print(f"  Sample keys:            {list(sample.keys())[:10]}...")
    has_mapping = "mapping" in sample
    print(f"  Has 'mapping' field:    {has_mapping}")

    # Write canonical file
    print(f"\nWriting to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f)

    out_size = OUTPUT_PATH.stat().st_size
    print(f"  Output size:            {out_size / 1024 / 1024:.1f} MB")
    print(f"\n=== Step 1 Complete ===")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main()
