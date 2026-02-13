import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

from utils.logger_config import setup_logger

logger = setup_logger(__name__, log_level=logging.INFO)


def deduplicate_scenarios(scenarios: List[Dict]) -> List[Dict]:
    
    logger.info("Deduplicating scenarios...")
    
    seen = set()
    unique = []
    duplicates = 0
    
    for scenario in scenarios:
        
        emergency = scenario.get('emergency_type', scenario.get('title', '')).lower().strip()
        desc = scenario.get('description', '')[:150].lower().strip()
        symptoms = str(scenario.get('symptoms', []))[:50].lower()
        aug_type = scenario.get('augmentation_type', '')
        
        emergency = re.sub(r'\s+', ' ', emergency)
        desc = re.sub(r'\s+', ' ', desc)
        
        sig = f"{emergency}:{desc}:{symptoms}:{aug_type}"
        
        if sig not in seen:
            seen.add(sig)
            unique.append(scenario)
        else:
            duplicates += 1
    
    logger.info(f"Removed {duplicates} duplicates")
    logger.info(f"Kept {len(unique)} unique scenarios")
    
    return unique


def merge_all_checkpoints(
    data_dir: str = "./data",
    output_file: str = "all_authoritative_scenarios.json"
) -> None:
    
    logger.info("=" * 70)
    logger.info("MERGING ALL CHECKPOINT FILES")
    logger.info("=" * 70)
    
    data_path = Path(data_dir)
    
    checkpoint_patterns = [
        "checkpoint_*.json",
        "*_scenarios.json"
    ]
    
    all_files = []
    for pattern in checkpoint_patterns:
        all_files.extend(data_path.glob(pattern))
    
    all_files = [f for f in all_files if f.name != output_file]
    
    if not all_files:
        logger.error("No checkpoint files found!")
        logger.error(f"Looking in: {data_path.absolute()}")
        return
    
    logger.info(f"Found {len(all_files)} checkpoint files:")
    for f in all_files:
        logger.info(f"  - {f.name}")
    
    logger.info("Loading and merging...")
    all_scenarios = []
    source_counts = defaultdict(int)
    
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                scenarios = data
            elif isinstance(data, dict):
                scenarios = data.get('scenarios', [])
            else:
                scenarios = []
            
            source_name = filepath.stem.replace('checkpoint_', '').replace('_scenarios', '')
            source_counts[source_name] = len(scenarios)
            
            all_scenarios.extend(scenarios)
            logger.info(f"  Loaded {filepath.name}: {len(scenarios)} scenarios")
            
        except Exception as e:
            logger.error(f"  Error loading {filepath.name}: {e}")
            continue
    
    logger.info(f"Total scenarios before deduplication: {len(all_scenarios)}")
    
    unique_scenarios = deduplicate_scenarios(all_scenarios)
    
    output_path = data_path / output_file
    
    logger.info(f"Saving merged data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_scenarios': len(unique_scenarios),
            'source_breakdown': dict(source_counts),
            'merged_files': [f.name for f in all_files],
            'scenarios': unique_scenarios
        }, f, indent=2, ensure_ascii=False)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    
    logger.info("Merge Complete")
    logger.info(f"  Files merged:        {len(all_files)}")
    logger.info(f"  Total scenarios:     {len(unique_scenarios)}")
    logger.info(f"  Duplicates removed:  {len(all_scenarios) - len(unique_scenarios)}")
    logger.info(f"  Output file:         {output_path}")
    logger.info(f"  File size:           {file_size:.2f} MB")
    
    logger.info("Source breakdown:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_scenarios) * 100) if all_scenarios else 0
        bar = "█" * int(percentage / 2)
        logger.info(f"  {source:30s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    categories = defaultdict(int)
    for scenario in unique_scenarios:
        cat = scenario.get('category', 'Unknown')
        categories[cat] += 1
    
    if categories:
        logger.info("Top categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(unique_scenarios) * 100)
            logger.info(f"  {cat:25s}: {count:4d} ({percentage:5.1f}%)")
    
    logger.info("MERGE SUCCESSFUL!")


def main():
    
    # Confirm
    response = input("\nProceed with merge? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        logger.info("Merge cancelled")
        return 1
    
    merge_all_checkpoints(data_dir="./data")
    return 0


if __name__ == "__main__":
    try:
        import sys
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Merge interrupted by user")
        import sys
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        import sys
        sys.exit(1)