import json
import requests
import time
import logging
import sys
from typing import List, Dict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("augmentation_backend.log"), 
        logging.StreamHandler(sys.stdout)               
    ]
)
logger = logging.getLogger(__name__)

class ScenarioAugmentation:
    
    def __init__(self, data_dir: str = "./data", ollama_model: str = "llama3.2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def _ollama_(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False,
                    'temperature': 0.3,
                    'format': 'json'
                },
                timeout=90
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            logger.warning(f"Ollama returned status code {response.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ollama connection error: {str(e)[:100]}")
            return ""
    
    def augment_single_scenario(self, scenario: Dict) -> List[Dict]:
        variations = []
        if scenario.get('severity') == 'critical':
            return []
        
        age_prompt = f"Adapt this medical scenario for ELDERLY patients (65+). Return ONLY JSON: {json.dumps(scenario)}"
        response = self._ollama_(age_prompt)
        self._process_variant(response, variations, 'age_specific_elderly', scenario)
        
        severity_prompt = f"Create a SEVERE version of this medical scenario. Return ONLY JSON: {json.dumps(scenario)}"
        response = self._ollama_(severity_prompt)
        self._process_variant(response, variations, 'severity_increased', scenario)
        
        return variations

    def _process_variant(self, response, variations, aug_type, original):
        """Helper to parse LLM response and log failures"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                variant = json.loads(response[start:end])
                variant['augmentation_type'] = aug_type
                variant['base_scenario_id'] = original.get('title', 'unknown')
                variations.append(variant)
        except Exception:
            logger.debug(f"Failed to parse {aug_type} variation")

    def augment_scenarios(self, scenarios: List[Dict], target_count: int = None) -> List[Dict]:    
        if target_count is None:
            target_count = len(scenarios) // 2  
            
        scenarios_to_augment = min(len(scenarios), target_count)
        
        logger.info(f"Starting augmentation: Target {scenarios_to_augment} scenarios")
        
        augmented_scenarios = []
        for idx, scenario in enumerate(scenarios[:scenarios_to_augment], 1):
            if idx % 50 == 0:
                logger.info(f"Progress: {idx}/{scenarios_to_augment} | Generated: {len(augmented_scenarios)}")
            
            try:
                variations = self.augment_single_scenario(scenario)
                augmented_scenarios.extend(variations)
                time.sleep(0.3) 
            except Exception as e:
                logger.error(f"Error augmenting scenario {idx}: {e}")
                continue
        
        return augmented_scenarios
    
    def augment_scenarios_file(self, input_file: str, output_file: str = None, target_total: int = 3000) -> str:
        input_path = self.data_dir / input_file
        logger.info(f"Loading data from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_scenarios = data.get('scenarios', data if isinstance(data, list) else [])
        current_count = len(original_scenarios)
        
        if current_count >= target_total:
            logger.info(f"Target already met ({current_count}/{target_total}). Skipping.")
            return str(input_path)
        
        needed = target_total - current_count
        scenarios_to_augment = min(len(original_scenarios), needed // 2)
        
        augmented = self.augment_scenarios(original_scenarios, target_count=scenarios_to_augment)
        combined = original_scenarios + augmented
        final_scenarios = deduplicate_scenarios_simple(combined)
        
        output_path = self.data_dir / (output_file or input_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'final_count': len(final_scenarios),
                'scenarios': final_scenarios
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Augmentation complete. Final count: {len(final_scenarios)}. Saved to {output_path}")
        return str(output_path)

def deduplicate_scenarios_simple(scenarios: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for scenario in scenarios:
        sig = f"{scenario.get('title', '')}:{scenario.get('severity', '')}".lower()
        if sig not in seen:
            seen.add(sig)
            unique.append(scenario)
    return unique

if __name__ == "__main__":
    augmentor = ScenarioAugmentation()
    try:
        augmentor.augment_scenarios_file("all_authoritative_scenarios.json")
    except Exception as e:
        logger.critical(f"Process crashed: {e}")