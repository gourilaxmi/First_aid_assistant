"""
Analyze collected scenario data
Professional data analysis tool with comprehensive metrics

Usage: python scripts/analyze_data.py [filepath]
"""

import json
import sys
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger_config import setup_logger

logger = setup_logger(__name__, log_level=logging.INFO)


def load_scenarios(filepath: str) -> Optional[List[Dict]]:
    """
    Load scenarios from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of scenarios or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            scenarios = data
        elif isinstance(data, dict):
            scenarios = data.get('scenarios', [])
        else:
            logger.error(f"Unexpected JSON structure in {filepath}")
            return None
            
        logger.info(f"Loaded {len(scenarios)} scenarios from {filepath}")
        return scenarios
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {str(e)}")
        return None


def analyze_basic_stats(scenarios: List[Dict]) -> None:
    """
    Analyze basic statistics
    
    Args:
        scenarios: List of scenario dictionaries
    """
    logger.info("=" * 70)
    logger.info("BASIC STATISTICS")
    logger.info("=" * 70)
    
    logger.info(f"Total Scenarios: {len(scenarios)}")
    
    # Count fields
    with_symptoms = sum(1 for s in scenarios if s.get('symptoms'))
    with_steps = sum(1 for s in scenarios if s.get('immediate_steps'))
    with_warnings = sum(1 for s in scenarios if s.get('when_to_seek_help'))
    with_donts = sum(1 for s in scenarios if s.get('do_not'))
    
    logger.info("Completeness:")
    logger.info(f"  With symptoms:        {with_symptoms:4d} ({with_symptoms/len(scenarios)*100:.1f}%)")
    logger.info(f"  With immediate steps: {with_steps:4d} ({with_steps/len(scenarios)*100:.1f}%)")
    logger.info(f"  With warnings:        {with_warnings:4d} ({with_warnings/len(scenarios)*100:.1f}%)")
    logger.info(f"  With don'ts:          {with_donts:4d} ({with_donts/len(scenarios)*100:.1f}%)")


def analyze_categories(scenarios: List[Dict]) -> None:
    """
    Analyze category distribution
    
    Args:
        scenarios: List of scenario dictionaries
    """
    logger.info("=" * 70)
    logger.info("CATEGORY BREAKDOWN")
    logger.info("=" * 70)
    
    categories = Counter(s.get('category', 'Unknown') for s in scenarios)
    
    logger.info(f"Total categories: {len(categories)}")
    logger.info("Distribution:")
    
    for cat, count in categories.most_common():
        percentage = (count / len(scenarios)) * 100
        bar = "█" * int(percentage / 2)
        logger.info(f"  {cat:25s}: {count:4d} ({percentage:5.1f}%) {bar}")


def analyze_severity(scenarios: List[Dict]) -> None:
    """
    Analyze severity level distribution
    
    Args:
        scenarios: List of scenario dictionaries
    """
    logger.info("=" * 70)
    logger.info("SEVERITY LEVELS")
    logger.info("=" * 70)
    
    severities = Counter(s.get('severity', 'Unknown') for s in scenarios)
    
    logger.info("Distribution:")
    
    severity_order = ['critical', 'severe', 'moderate', 'minor', 'Unknown']
    for sev in severity_order:
        if sev in severities:
            count = severities[sev]
            percentage = (count / len(scenarios)) * 100
            bar = "█" * int(percentage / 2)
            logger.info(f"  {sev:15s}: {count:4d} ({percentage:5.1f}%) {bar}")


def analyze_sources(scenarios: List[Dict]) -> None:
    """
    Analyze data source distribution
    
    Args:
        scenarios: List of scenario dictionaries
    """
    logger.info("=" * 70)
    logger.info("DATA SOURCES")
    logger.info("=" * 70)
    
    # Extract source names
    source_names = []
    for s in scenarios:
        source = s.get('source', 'Unknown')
        source_name = source.split(':')[0].split(',')[0].strip()
        source_names.append(source_name)
    
    sources = Counter(source_names)
    
    logger.info(f"Total sources: {len(sources)}")
    logger.info("Distribution:")
    
    for source, count in sources.most_common():
        percentage = (count / len(scenarios)) * 100
        bar = "█" * int(percentage / 2)
        logger.info(f"  {source:30s}: {count:4d} ({percentage:5.1f}%) {bar}")


def analyze_content_quality(scenarios: List[Dict]) -> None:
    """
    Analyze content quality metrics
    
    Args:
        scenarios: List of scenario dictionaries
    """
    logger.info("=" * 70)
    logger.info("CONTENT QUALITY")
    logger.info("=" * 70)
    
    # Content counts
    steps_counts = [len(s.get('immediate_steps', [])) for s in scenarios]
    symptoms_counts = [len(s.get('symptoms', [])) for s in scenarios]
    warnings_counts = [len(s.get('when_to_seek_help', [])) for s in scenarios]
    donts_counts = [len(s.get('do_not', [])) for s in scenarios]
    
    logger.info("Average content per scenario:")
    if steps_counts:
        logger.info(f"  Immediate steps:     {sum(steps_counts)/len(steps_counts):5.1f}")
    if symptoms_counts:
        logger.info(f"  Symptoms:            {sum(symptoms_counts)/len(symptoms_counts):5.1f}")
    if warnings_counts:
        logger.info(f"  Warning conditions:  {sum(warnings_counts)/len(warnings_counts):5.1f}")
    if donts_counts:
        logger.info(f"  Don'ts:              {sum(donts_counts)/len(donts_counts):5.1f}")
    
    # Quality score
    quality_scores = []
    for s in scenarios:
        score = 0
        score += min(len(s.get('immediate_steps', [])) * 20, 40)
        score += min(len(s.get('symptoms', [])) * 10, 20)
        score += min(len(s.get('when_to_seek_help', [])) * 10, 20)
        score += min(len(s.get('do_not', [])) * 10, 20)
        quality_scores.append(score)
    
    avg_quality = sum(quality_scores) / len(quality_scores)
    logger.info(f"Average quality score: {avg_quality:.1f}/100")
    
    # Quality distribution
    high_quality = sum(1 for q in quality_scores if q >= 70)
    medium_quality = sum(1 for q in quality_scores if 40 <= q < 70)
    low_quality = sum(1 for q in quality_scores if q < 40)
    
    logger.info("Quality distribution:")
    logger.info(f"  High (70+):   {high_quality:4d} ({high_quality/len(scenarios)*100:.1f}%)")
    logger.info(f"  Medium (40-70): {medium_quality:4d} ({medium_quality/len(scenarios)*100:.1f}%)")
    logger.info(f"  Low (<40):    {low_quality:4d} ({low_quality/len(scenarios)*100:.1f}%)")


def find_sample_scenarios(scenarios: List[Dict]) -> None:
    """
    Display sample scenarios from each category
    
    Args:
        scenarios: List of scenario dictionaries
    """
    logger.info("=" * 70)
    logger.info("SAMPLE SCENARIOS")
    logger.info("=" * 70)
    
    categories = {}
    for s in scenarios:
        cat = s.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = s
    
    for cat, scenario in list(categories.items())[:5]:
        logger.info(f"\n{cat}:")
        logger.info(f"  Title: {scenario.get('title', 'N/A')}")
        logger.info(f"  Severity: {scenario.get('severity', 'N/A')}")
        steps = scenario.get('immediate_steps', [])
        if steps:
            logger.info(f"  First step: {steps[0][:60]}...")


def export_summary_report(scenarios: List[Dict], output_path: str) -> None:
    """
    Export summary report to JSON
    
    Args:
        scenarios: List of scenario dictionaries
        output_path: Output file path
    """
    report = {
        'total_scenarios': len(scenarios),
        'categories': dict(Counter(s.get('category', 'Unknown') for s in scenarios)),
        'severities': dict(Counter(s.get('severity', 'Unknown') for s in scenarios)),
        'sources': dict(Counter(
            s.get('source', 'Unknown').split(':')[0].split(',')[0].strip() 
            for s in scenarios
        )),
        'completeness': {
            'with_symptoms': sum(1 for s in scenarios if s.get('symptoms')),
            'with_steps': sum(1 for s in scenarios if s.get('immediate_steps')),
            'with_warnings': sum(1 for s in scenarios if s.get('when_to_seek_help')),
            'with_donts': sum(1 for s in scenarios if s.get('do_not'))
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Summary report saved to: {output_path}")


def main():
    """Main analysis function"""
    # Get filepath
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "./data/all_authoritative_scenarios.json"
    
    logger.info("=" * 70)
    logger.info("DATA ANALYSIS REPORT")
    logger.info("=" * 70)
    logger.info(f"File: {filepath}")
    
    # Load scenarios
    scenarios = load_scenarios(filepath)
    
    if scenarios is None:
        return 1
    
    # Run analyses
    analyze_basic_stats(scenarios)
    analyze_categories(scenarios)
    analyze_severity(scenarios)
    analyze_sources(scenarios)
    analyze_content_quality(scenarios)
    find_sample_scenarios(scenarios)
    
    # Export report
    report_path = filepath.replace('.json', '_summary.json')
    export_summary_report(scenarios, report_path)
    
    logger.info("=" * 70)
    logger.info("Analysis complete!")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())