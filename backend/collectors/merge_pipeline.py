import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("merge_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MasterDataPipeline:
   
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def _normalize_scenario_signature(self, scenario: Dict) -> str:
        title = scenario.get('title', scenario.get('emergency_type', '')).lower().strip()
        desc = scenario.get('description', scenario.get('additional_info', ''))[:100].lower().strip()
        return f"{title}:{desc}"
    
    def load_scenarios_from_file(self, filepath: str) -> List[Dict]:
        try:
            file_path = self.data_dir / filepath
            if not file_path.exists():
                logger.warning(f"File not found: {filepath}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                scenarios = data
            elif 'scenarios' in data:
                scenarios = data['scenarios']
            else:
                logger.warning(f"Unexpected JSON structure in {filepath}")
                return []
            
            logger.info(f"Loaded {len(scenarios)} scenarios from {filepath}")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []
    
    def merge_scenarios(self, scenario_files: List[str]) -> Dict:
       
        logger.info("STARTING MASTER DATA PIPELINE")
        
        all_scenarios = []
        source_stats = defaultdict(int)
        seen_signatures = set()
        duplicates_removed = 0
        
        for filename in scenario_files:
            scenarios = self.load_scenarios_from_file(filename)
            
            # Extract source name from filename
            source_name = filename.replace('_scenarios.json', '').replace('_', ' ').title()
            original_count = len(scenarios)
            
            # Deduplicate and add to collection
            for scenario in scenarios:
                signature = self._normalize_scenario_signature(scenario)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    all_scenarios.append(scenario)
                    source_stats[source_name] += 1
                else:
                    duplicates_removed += 1
        
     
        
        for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {source:30s}: {count:4d} unique scenarios")
        
        logger.info(f"  Total Unique Scenarios: {len(all_scenarios)}")
        logger.info(f"  Duplicates Removed: {duplicates_removed}")
        
        return {
            'total_scenarios': len(all_scenarios),
            'duplicates_removed': duplicates_removed,
            'source_breakdown': dict(source_stats),
            'scenarios': all_scenarios
        }
    
    def save_merged_data(self, merged_data: Dict, output_filename: str = "all_scenarios_merged.json"):
 
        output_path = self.data_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved merged data to: {output_path}")
            logger.info(f"Total scenarios in merged file: {merged_data['total_scenarios']}")
            
        except Exception as e:
            logger.error(f"Error saving merged data: {e}")
    
    def run_full_pipeline(self, output_filename: str = "all_scenarios_merged.json") -> Dict:
        
        scenario_files = [
            "redcross_scenarios.json",
            "mayo_scenarios.json",
            "cleveland_scenarios.json",
            "healthline_scenarios.json",
            "nhs_scenarios.json",
            "stjohn_scenarios.json",
            "webmd_scenarios.json",
            "cdc_scenarios.json",
            "checkpoint_american_heart_association.json",
            "checkpoint_who_emergency_care.json",
            "checkpoint_medlineplus_nih.json",
            "checkpoint_red_cross_online.json",
            "checkpoint_poison_control.json",
            "checkpoint_kidshealth.json",
            "checkpoint_familydoctor.json",
            "checkpoint_emergencycareforyou.json",
            "checkpoint_aap_healthychildren.json",
            "checkpoint_johns_hopkins_medicine.json",
        ]
        
        # Merge all scenarios
        merged_data = self.merge_scenarios(scenario_files)
        
        # Save to file
        self.save_merged_data(merged_data, output_filename)
        
        return merged_data
    
    def create_categorized_dataset(self, merged_data: Dict, output_filename: str = "scenarios_by_category.json"):
        
        logger.info("Creating categorized dataset...")
        
        categories = defaultdict(list)
        
        for scenario in merged_data['scenarios']:
            category = scenario.get('category', 'General')
            categories[category].append(scenario)
        
        categorized_data = {
            'total_scenarios': len(merged_data['scenarios']),
            'categories': dict(categories),
            'category_counts': {cat: len(scenarios) for cat, scenarios in categories.items()}
        }
        
        output_path = self.data_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(categorized_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved categorized data to: {output_path}")
            logger.info(f"Categories: {len(categories)}")
            
            for category, count in sorted(categorized_data['category_counts'].items(), 
                                         key=lambda x: x[1], reverse=True):
                logger.info(f"  {category:30s}: {count:4d} scenarios")
                
        except Exception as e:
            logger.error(f"Error saving categorized data: {e}")
    
    def validate_scenarios(self, scenarios: List[Dict]) -> Dict:
        
        total = len(scenarios)
        valid = 0
        missing_fields = defaultdict(int)
        
        required_fields = ['title', 'category', 'immediate_steps']
        
        for scenario in scenarios:
            is_valid = True
            
            for field in required_fields:
                if not scenario.get(field):
                    missing_fields[field] += 1
                    is_valid = False
            
            if is_valid:
                valid += 1
        
        validation_results = {
            'total_scenarios': total,
            'valid_scenarios': valid,
            'invalid_scenarios': total - valid,
            'validation_rate': (valid / total * 100) if total > 0 else 0,
            'missing_fields': dict(missing_fields)
        }
        
        logger.info(f"Validation complete: {valid}/{total} scenarios valid ({validation_results['validation_rate']:.1f}%)")
        
        if missing_fields:
            logger.warning("Missing fields found:")
            for field, count in missing_fields.items():
                logger.warning(f"  {field}: {count} scenarios")
        
        return validation_results


if __name__ == "__main__":
    pipeline = MasterDataPipeline(data_dir="../data")
    
    merged_data = pipeline.run_full_pipeline(output_filename="all_scenarios_merged.json")
    
    pipeline.create_categorized_dataset(merged_data, output_filename="scenarios_by_category.json")
    
    validation_results = pipeline.validate_scenarios(merged_data['scenarios'])
    
    logger.info("PIPELINE COMPLETE")
 