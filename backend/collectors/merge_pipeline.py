"""
Master Data Pipeline - Collects and merges data from all sources
This is the CLASS that orchestrates data collection from multiple sources
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from .red_cross import RedCrossCollector
from .web_collector import (
    EnhancedWebCollector,
    MayoClinicCollector,
    ClevelandClinicCollector,
    HealthlineCollector,
    CDCEmergencyCollector,
    NHSCollector,
    StJohnCollector,
    WebMDCollector
)


class MasterDataPipeline:
    """
    Orchestrates data collection from all sources
    
    Usage:
        pipeline = MasterDataPipeline(data_dir="./data")
        scenarios = pipeline.collect_all(sources=['redcross', 'mayo', 'nhs'])
        pipeline.deduplicate()
        pipeline.save_merged("output.json")
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the pipeline
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_scenarios = []
        self.sources_collected = []
        
        print(f"\n📁 Data directory: {self.data_dir.absolute()}")
    
    def collect_all(self, sources: Optional[List[str]] = None) -> List[Dict]:
        """
        Collect from all or specified sources
        
        Args:
            sources: List of source names to collect from
                    Options: 'redcross', 'mayo', 'cleveland', 
                            'healthline', 'cdc', 'nhs', 'stjohn', 'webmd'
                    If None, collects from all sources
        
        Returns:
            List of all collected scenarios
        """
        if sources is None:
            sources = ['redcross', 'mayo', 'cleveland', 
                      'healthline', 'cdc', 'nhs', 'stjohn', 'webmd']
        
        print(f"\n🔄 Collecting from {len(sources)} sources: {', '.join(sources)}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Red Cross PDF
        if 'redcross' in sources:
            print("\n" + "─"*60)
            print("📕 RED CROSS PDF")
            print("─"*60)
            try:
                collector = RedCrossCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect()
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('redcross')
                print(f"✅ Red Cross: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ Red Cross failed: {e}")
        
        # Mayo Clinic
        if 'mayo' in sources:
            print("\n" + "─"*60)
            print("🏥 MAYO CLINIC")
            print("─"*60)
            try:
                collector = MayoClinicCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('mayo')
                print(f"✅ Mayo Clinic: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ Mayo Clinic failed: {e}")
        
        # Cleveland Clinic
        if 'cleveland' in sources:
            print("\n" + "─"*60)
            print("🏥 CLEVELAND CLINIC")
            print("─"*60)
            try:
                collector = ClevelandClinicCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('cleveland')
                print(f"✅ Cleveland Clinic: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ Cleveland Clinic failed: {e}")
        
        # Healthline
        if 'healthline' in sources:
            print("\n" + "─"*60)
            print("🏥 HEALTHLINE")
            print("─"*60)
            try:
                collector = HealthlineCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('healthline')
                print(f"✅ Healthline: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ Healthline failed: {e}")
        
        # CDC Emergency Preparedness
        if 'cdc' in sources:
            print("\n" + "─"*60)
            print("🏛️ CDC EMERGENCY PREPAREDNESS")
            print("─"*60)
            try:
                collector = CDCEmergencyCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('cdc')
                print(f"✅ CDC: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ CDC failed: {e}")
        
        # NHS UK
        if 'nhs' in sources:
            print("\n" + "─"*60)
            print("🏥 NHS UK")
            print("─"*60)
            try:
                collector = NHSCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('nhs')
                print(f"✅ NHS UK: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ NHS UK failed: {e}")
        
        # St John Ambulance
        if 'stjohn' in sources:
            print("\n" + "─"*60)
            print("🚑 ST JOHN AMBULANCE")
            print("─"*60)
            try:
                collector = StJohnCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('stjohn')
                print(f"✅ St John: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ St John failed: {e}")
        
        # WebMD
        if 'webmd' in sources:
            print("\n" + "─"*60)
            print("🏥 WEBMD")
            print("─"*60)
            try:
                collector = WebMDCollector(data_dir=str(self.data_dir))
                scenarios = collector.collect(max_pages=100)
                self.all_scenarios.extend(scenarios)
                self.sources_collected.append('webmd')
                print(f"✅ WebMD: {len(scenarios)} scenarios collected")
            except Exception as e:
                print(f"❌ WebMD failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"✅ Collection complete: {len(self.all_scenarios)} total scenarios")
        print(f"   Sources: {', '.join(self.sources_collected)}")
        print(f"{'='*60}")
        
        return self.all_scenarios
    
    def load_checkpoints(self) -> Dict[str, List[Dict]]:
        """
        Load existing checkpoint files from data directory
        
        Returns:
            Dictionary mapping source name to list of scenarios
        """
        checkpoint_files = {
            'redcross': 'redcross_scenarios.json',
            'mayo': 'mayo_scenarios.json',
            'cleveland': 'cleveland_scenarios.json',
            'healthline': 'healthline_scenarios.json',
            'cdc': 'cdc_scenarios.json',
            'nhs': 'nhs_scenarios.json',
            'stjohn': 'stjohn_scenarios.json',
            'webmd': 'webmd_scenarios.json',
        }
        
        checkpoint_data = {}
        
        print("\n📂 Loading checkpoint files...")
        
        for source, filename in checkpoint_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        checkpoint_data[source] = data
                        print(f"  ✅ {source:15s}: {len(data):4d} scenarios from {filename}")
                except Exception as e:
                    print(f"  ❌ {source:15s}: Error loading - {e}")
        
        if not checkpoint_data:
            print("  ⚠️  No checkpoint files found")
        else:
            print(f"\n✅ Loaded {len(checkpoint_data)} checkpoint files")
        
        return checkpoint_data
    
    def merge_from_checkpoints(self, checkpoint_data: Dict[str, List[Dict]]):
        """
        Merge data from checkpoint files into all_scenarios
        
        Args:
            checkpoint_data: Dictionary from load_checkpoints()
        """
        print("\n🔄 Merging checkpoint data...")
        
        for source, scenarios in checkpoint_data.items():
            self.all_scenarios.extend(scenarios)
            self.sources_collected.append(source)
            print(f"  ✅ Merged {len(scenarios)} scenarios from {source}")
        
        print(f"\n✅ Total merged: {len(self.all_scenarios)} scenarios from {len(checkpoint_data)} sources")
    
    def deduplicate(self):
        """
        Remove duplicate scenarios based on title similarity
        Uses case-insensitive title matching
        """
        print(f"\n🔄 Deduplicating scenarios...")
        original_count = len(self.all_scenarios)
        
        # Simple deduplication by normalized title
        seen_titles = set()
        unique_scenarios = []
        duplicates_removed = 0
        
        for scenario in self.all_scenarios:
            # Normalize title for comparison
            title = scenario.get('title', '').lower().strip()
            
            # Also check for very similar content
            title_key = title[:50] if len(title) > 50 else title
            
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_scenarios.append(scenario)
            else:
                duplicates_removed += 1
        
        self.all_scenarios = unique_scenarios
        
        print(f"  ✅ Removed {duplicates_removed} duplicates")
        print(f"  ✅ {len(self.all_scenarios)} unique scenarios remaining")
    
    def save_merged(self, filename: str):
        """
        Save merged scenarios to file
        
        Args:
            filename: Output filename (saved in data_dir)
        """
        output_path = self.data_dir / filename
        
        print(f"\n💾 Saving merged data...")
        
        # Add scenario IDs if missing
        for i, scenario in enumerate(self.all_scenarios):
            if 'scenario_id' not in scenario:
                scenario['scenario_id'] = f"scenario_{i+1:06d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_scenarios, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"  ✅ Saved to: {output_path}")
        print(f"  ✅ File size: {file_size_mb:.2f} MB")
        print(f"  ✅ Scenarios: {len(self.all_scenarios)}")
    
    def print_summary(self):
        print("\n" + "="*70)
        print(" "*20 + "COLLECTION SUMMARY")
        print("="*70)
        
        print(f"\n📊 Total scenarios: {len(self.all_scenarios)}")
        print(f"📂 Sources collected: {len(self.sources_collected)}")
        
        # Handle empty scenarios
        if len(self.all_scenarios) == 0:
            print("\n⚠️  No scenarios were collected!")
            print("\n📝 Possible reasons:")
            print("   • PDF file not found or not readable")
            print("   • Ollama not running (needs: ollama serve)")
            print("   • Data sources require manual download")
            print("   • Collection filters too strict")
            print("\n💡 Next steps:")
            print("   1. Check that PDF exists in data directory")
            print("   2. Verify Ollama is running: ollama serve")
            print("   3. Check error messages above")
            return
        
        if self.sources_collected:
            print(f"\n✅ Sources breakdown:")
            
            # Count scenarios per source
            source_counts = {}
            for scenario in self.all_scenarios:
                source = scenario.get('source', 'Unknown')
                # Extract source name (before : or ,)
                source_name = source.split(':')[0].split(',')[0].strip()
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
            
            for source in self.sources_collected:
                # Try to match source in counts
                count = 0
                for src_name, cnt in source_counts.items():
                    if source.lower() in src_name.lower():
                        count = cnt
                        break
                
                if count > 0:
                    percentage = (count / len(self.all_scenarios)) * 100
                    bar = "█" * int(percentage / 2)
                    print(f"  {source:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Category breakdown
        from collections import Counter
        categories = Counter(s.get('category', 'Unknown') for s in self.all_scenarios)
        
        print(f"\n📋 Top categories:")
        for cat, count in categories.most_common(5):
            percentage = (count / len(self.all_scenarios)) * 100
            print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # Severity breakdown
        severities = Counter(s.get('severity', 'Unknown') for s in self.all_scenarios)
        
        print(f"\n⚠️  Severity levels:")
        for sev in ['critical', 'severe', 'moderate', 'minor']:
            if sev in severities:
                count = severities[sev]
                percentage = (count / len(self.all_scenarios)) * 100
                print(f"  {sev:12s}: {count:4d} ({percentage:5.1f}%)")
        
        # Quality metrics - only if we have scenarios
        if len(self.all_scenarios) > 0:
            avg_steps = sum(len(s.get('immediate_steps', [])) for s in self.all_scenarios) / len(self.all_scenarios)
            avg_symptoms = sum(len(s.get('symptoms', [])) for s in self.all_scenarios) / len(self.all_scenarios)
            
            print(f"\n📈 Content quality:")
            print(f"  Avg steps per scenario:    {avg_steps:.1f}")
            print(f"  Avg symptoms per scenario: {avg_symptoms:.1f}")
        
        print("\n" + "="*70)