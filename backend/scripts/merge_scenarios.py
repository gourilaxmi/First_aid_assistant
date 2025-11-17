"""
Merge Multiple Scenario JSON Files
Merges all checkpoint files from data directory into all_authoritative_scenarios.json

Usage:
    python merge_scenarios.py
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict


def deduplicate_scenarios(scenarios: List[Dict]) -> List[Dict]:
    """Remove duplicates from scenario list"""
    
    print("\n🔍 Deduplicating scenarios...")
    
    seen = set()
    unique = []
    duplicates = 0
    
    for scenario in scenarios:
        # Create signature
        emergency = scenario.get('emergency_type', scenario.get('title', '')).lower().strip()
        desc = scenario.get('description', '')[:150].lower().strip()
        symptoms = str(scenario.get('symptoms', []))[:50].lower()
        aug_type = scenario.get('augmentation_type', '')
        
        # Normalize
        emergency = re.sub(r'\s+', ' ', emergency)
        desc = re.sub(r'\s+', ' ', desc)
        
        sig = f"{emergency}:{desc}:{symptoms}:{aug_type}"
        
        if sig not in seen:
            seen.add(sig)
            unique.append(scenario)
        else:
            duplicates += 1
    
    print(f"  Removed {duplicates} duplicates")
    print(f"  Kept {len(unique)} unique scenarios")
    
    return unique


def merge_all_checkpoints(data_dir: str = "./data", output_file: str = "all_authoritative_scenarios.json"):
    """
    Merge all checkpoint JSON files from data directory
    
    Args:
        data_dir: Directory containing checkpoint files
        output_file: Output filename
    """
    
    print("\n" + "="*70)
    print("🔄 MERGING ALL CHECKPOINT FILES")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # Find all JSON checkpoint files
    checkpoint_patterns = [
        "checkpoint_*.json",
        "*_scenarios.json"
    ]
    
    all_files = []
    for pattern in checkpoint_patterns:
        all_files.extend(data_path.glob(pattern))
    
    # Exclude output file if it exists
    all_files = [f for f in all_files if f.name != output_file]
    
    if not all_files:
        print("❌ No checkpoint files found!")
        print(f"   Looking in: {data_path.absolute()}")
        return
    
    print(f"\n📂 Found {len(all_files)} checkpoint files:")
    for f in all_files:
        print(f"   - {f.name}")
    
    # Merge all scenarios
    print(f"\n📥 Loading and merging...")
    all_scenarios = []
    source_counts = defaultdict(int)
    
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                scenarios = data
            elif isinstance(data, dict):
                scenarios = data.get('scenarios', [])
            else:
                scenarios = []
            
            # Track source
            source_name = filepath.stem.replace('checkpoint_', '').replace('_scenarios', '')
            source_counts[source_name] = len(scenarios)
            
            all_scenarios.extend(scenarios)
            print(f"   ✅ {filepath.name}: {len(scenarios)} scenarios")
            
        except Exception as e:
            print(f"   ❌ {filepath.name}: Error - {e}")
            continue
    
    print(f"\n📊 Total scenarios before deduplication: {len(all_scenarios)}")
    
    # Deduplicate
    unique_scenarios = deduplicate_scenarios(all_scenarios)
    
    # Save merged file
    output_path = data_path / output_file
    
    print(f"\n💾 Saving merged data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_scenarios': len(unique_scenarios),
            'source_breakdown': dict(source_counts),
            'merged_files': [f.name for f in all_files],
            'scenarios': unique_scenarios
        }, f, indent=2, ensure_ascii=False)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    
    # Print summary
    print(f"\n{'='*70}")
    print("MERGE COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n📊 Summary:")
    print(f"   Files merged:        {len(all_files)}")
    print(f"   Total scenarios:     {len(unique_scenarios)}")
    print(f"   Duplicates removed:  {len(all_scenarios) - len(unique_scenarios)}")
    print(f"   Output file:         {output_path}")
    print(f"   File size:           {file_size:.2f} MB")
    
    print(f"\n📚 Source breakdown:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_scenarios) * 100) if all_scenarios else 0
        bar = "█" * int(percentage / 2)
        print(f"   {source:30s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Category breakdown
    categories = defaultdict(int)
    for scenario in unique_scenarios:
        cat = scenario.get('category', 'Unknown')
        categories[cat] += 1
    
    if categories:
        print(f"\n📋 Top categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(unique_scenarios) * 100)
            print(f"   {cat:25s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("✅ MERGE SUCCESSFUL!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"   1. To augment: python master_pipeline.py --augment")
    print(f"   2. Or upload to Pinecone: python master_pipeline.py --phase2")


def main():
    """Main merge function"""
    
    print("\n" + "="*70)
    print("🔄 SCENARIO CHECKPOINT MERGER")
    print("="*70)
    print("\nThis will merge ALL checkpoint files in ./data/")
    print("into: all_authoritative_scenarios.json")
    print("\nCheckpoint files include:")
    print("  • checkpoint_*.json")
    print("  • *_scenarios.json")
    print("  • fast_scenarios.json")
    print("\nDuplicates will be automatically removed.")
    print("="*70)
    
    # Confirm
    response = input("\nProceed with merge? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Merge cancelled")
        return
    
    # Run merge
    merge_all_checkpoints(data_dir="./data")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Merge interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()