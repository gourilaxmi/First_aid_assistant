"""
Analyze collected scenario data
Usage: python scripts/analyze_data.py [filepath]
"""

import json
import sys
import os
from collections import Counter
from pathlib import Path


def load_scenarios(filepath):
    """Load scenarios from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {filepath}")
        return None


def analyze_basic_stats(scenarios):
    """Basic statistics"""
    print("\n" + "="*70)
    print("📊 BASIC STATISTICS")
    print("="*70)
    
    print(f"\nTotal Scenarios: {len(scenarios)}")
    
    # Count fields
    with_symptoms = sum(1 for s in scenarios if s.get('symptoms'))
    with_steps = sum(1 for s in scenarios if s.get('immediate_steps'))
    with_warnings = sum(1 for s in scenarios if s.get('when_to_seek_help'))
    with_donts = sum(1 for s in scenarios if s.get('do_not'))
    
    print(f"\nCompleteness:")
    print(f"  With symptoms:        {with_symptoms:4d} ({with_symptoms/len(scenarios)*100:.1f}%)")
    print(f"  With immediate steps: {with_steps:4d} ({with_steps/len(scenarios)*100:.1f}%)")
    print(f"  With warnings:        {with_warnings:4d} ({with_warnings/len(scenarios)*100:.1f}%)")
    print(f"  With don'ts:          {with_donts:4d} ({with_donts/len(scenarios)*100:.1f}%)")


def analyze_categories(scenarios):
    """Analyze categories"""
    print("\n" + "="*70)
    print("🏷️  CATEGORY BREAKDOWN")
    print("="*70)
    
    categories = Counter(s.get('category', 'Unknown') for s in scenarios)
    
    print(f"\nTotal categories: {len(categories)}")
    print(f"\nDistribution:")
    
    for cat, count in categories.most_common():
        percentage = (count / len(scenarios)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {cat:25s}: {count:4d} ({percentage:5.1f}%) {bar}")


def analyze_severity(scenarios):
    """Analyze severity levels"""
    print("\n" + "="*70)
    print("⚠️  SEVERITY LEVELS")
    print("="*70)
    
    severities = Counter(s.get('severity', 'Unknown') for s in scenarios)
    
    print(f"\nDistribution:")
    
    severity_order = ['critical', 'severe', 'moderate', 'minor', 'Unknown']
    for sev in severity_order:
        if sev in severities:
            count = severities[sev]
            percentage = (count / len(scenarios)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {sev:15s}: {count:4d} ({percentage:5.1f}%) {bar}")


def analyze_sources(scenarios):
    """Analyze data sources"""
    print("\n" + "="*70)
    print("📁 DATA SOURCES")
    print("="*70)
    
    # Extract source names
    source_names = []
    for s in scenarios:
        source = s.get('source', 'Unknown')
        # Get just the source name (before : or ,)
        source_name = source.split(':')[0].split(',')[0].strip()
        source_names.append(source_name)
    
    sources = Counter(source_names)
    
    print(f"\nTotal sources: {len(sources)}")
    print(f"\nDistribution:")
    
    for source, count in sources.most_common():
        percentage = (count / len(scenarios)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {source:30s}: {count:4d} ({percentage:5.1f}%) {bar}")


def analyze_content_quality(scenarios):
    """Analyze content quality metrics"""
    print("\n" + "="*70)
    print("📋 CONTENT QUALITY")
    print("="*70)
    
    # Steps count
    steps_counts = [len(s.get('immediate_steps', [])) for s in scenarios]
    symptoms_counts = [len(s.get('symptoms', [])) for s in scenarios]
    warnings_counts = [len(s.get('when_to_seek_help', [])) for s in scenarios]
    donts_counts = [len(s.get('do_not', [])) for s in scenarios]
    
    print(f"\nAverage content per scenario:")
    if steps_counts:
        print(f"  Immediate steps:     {sum(steps_counts)/len(steps_counts):5.1f}")
    if symptoms_counts:
        print(f"  Symptoms:            {sum(symptoms_counts)/len(symptoms_counts):5.1f}")
    if warnings_counts:
        print(f"  Warning conditions:  {sum(warnings_counts)/len(warnings_counts):5.1f}")
    if donts_counts:
        print(f"  Don'ts:              {sum(donts_counts)/len(donts_counts):5.1f}")
    
    # Quality score
    quality_scores = []
    for s in scenarios:
        score = 0
        score += min(len(s.get('immediate_steps', [])) * 20, 40)  # Max 40
        score += min(len(s.get('symptoms', [])) * 10, 20)         # Max 20
        score += min(len(s.get('when_to_seek_help', [])) * 10, 20) # Max 20
        score += min(len(s.get('do_not', [])) * 10, 20)           # Max 20
        quality_scores.append(score)
    
    avg_quality = sum(quality_scores) / len(quality_scores)
    print(f"\nAverage quality score: {avg_quality:.1f}/100")
    
    # Quality distribution
    high_quality = sum(1 for q in quality_scores if q >= 70)
    medium_quality = sum(1 for q in quality_scores if 40 <= q < 70)
    low_quality = sum(1 for q in quality_scores if q < 40)
    
    print(f"\nQuality distribution:")
    print(f"  High (70+):   {high_quality:4d} ({high_quality/len(scenarios)*100:.1f}%)")
    print(f"  Medium (40-70): {medium_quality:4d} ({medium_quality/len(scenarios)*100:.1f}%)")
    print(f"  Low (<40):    {low_quality:4d} ({low_quality/len(scenarios)*100:.1f}%)")


def find_sample_scenarios(scenarios):
    """Show sample scenarios from each category"""
    print("\n" + "="*70)
    print("📝 SAMPLE SCENARIOS")
    print("="*70)
    
    categories = {}
    for s in scenarios:
        cat = s.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = s
    
    for cat, scenario in list(categories.items())[:5]:  # Show first 5
        print(f"\n{cat}:")
        print(f"  Title: {scenario.get('title', 'N/A')}")
        print(f"  Severity: {scenario.get('severity', 'N/A')}")
        steps = scenario.get('immediate_steps', [])
        if steps:
            print(f"  First step: {steps[0][:60]}...")


def export_summary_report(scenarios, output_path):
    """Export summary report"""
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
    
    print(f"\n💾 Summary report saved to: {output_path}")


def main():
    """Main analysis function"""
    
    # Get filepath
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default to merged scenarios
        filepath = "./data/all_authoritative_scenarios.json"
    
    print("\n" + "="*70)
    print(" "*20 + "DATA ANALYSIS REPORT")
    print("="*70)
    print(f"\nFile: {filepath}")
    
    # Load scenarios
    scenarios = load_scenarios(filepath)
    
    if scenarios is None:
        return
    
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
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()