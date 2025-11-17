"""
Scenario Augmentation Module
Creates medically valid variations of existing scenarios

Usage from master_pipeline.py:
    from collectors.augmentation import ScenarioAugmentor
    augmentor = ScenarioAugmentor(data_dir="./data")
    augmented = augmentor.augment_scenarios_file("all_authoritative_scenarios.json")
"""

import json
import requests
import time
from typing import List, Dict
from pathlib import Path


class ScenarioAugmentor:
    """Smart medical scenario augmentation"""
    
    def __init__(self, data_dir: str = "./data", ollama_model: str = "llama3.2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False,
                    'temperature': 0.3
                },
                timeout=90
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            return ""
        except Exception as e:
            print(f"      Ollama error: {str(e)[:60]}")
            return ""
    
    def augment_single_scenario(self, scenario: Dict) -> List[Dict]:
        """
        Create medically valid variations of a single scenario
        
        Returns:
            List of augmented scenario variations
        """
        variations = []
        
        # Skip if already critical
        if scenario.get('severity') == 'critical':
            return []
        
        # Variation 1: Age-specific (elderly)
        age_prompt = f"""You are a medical professional. Adapt this scenario for ELDERLY patients (65+).

IMPORTANT: Elderly have fragile bones, take medications, different symptoms.

Original:
{json.dumps(scenario, indent=2)}

Create elderly-specific variation. Modify symptoms, actions, warnings for elderly.
Return ONLY JSON object, no extra text."""

        response = self._call_ollama(age_prompt)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                variant = json.loads(response[start:end])
                variant['augmentation_type'] = 'age_specific_elderly'
                variant['base_scenario_id'] = scenario.get('emergency_type', scenario.get('title', 'unknown'))
                variations.append(variant)
        except:
            pass
        
        # Variation 2: Severity escalation
        severity_prompt = f"""You are a medical professional. Create a MORE SEVERE version.

Original (moderate):
{json.dumps(scenario, indent=2)}

Create severe/critical version with:
- More intense symptoms
- More urgent actions
- Clear 911 call needed

Return ONLY JSON object."""

        response = self._call_ollama(severity_prompt)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                variant = json.loads(response[start:end])
                variant['augmentation_type'] = 'severity_increased'
                variant['severity'] = 'severe'
                variant['base_scenario_id'] = scenario.get('emergency_type', scenario.get('title', 'unknown'))
                variations.append(variant)
        except:
            pass
        
        return variations
    
    def augment_scenarios(self, scenarios: List[Dict], target_count: int = None) -> List[Dict]:
        """
        Augment a list of scenarios
        
        Args:
            scenarios: List of original scenarios
            target_count: Target number of augmented scenarios (if None, augments all)
        
        Returns:
            List of augmented scenarios
        """
        print("\n" + "="*70)
        print("🔬 SCENARIO AUGMENTATION")
        print("="*70)
        
        if target_count is None:
            target_count = len(scenarios) // 2  # Augment 50% by default
        
        scenarios_to_augment = min(len(scenarios), target_count)
        
        print(f"\nOriginal scenarios: {len(scenarios)}")
        print(f"Will augment: {scenarios_to_augment} scenarios")
        print(f"Expected output: ~{scenarios_to_augment * 2} augmented scenarios")
        print(f"Estimated time: {scenarios_to_augment * 0.5 / 60:.1f} minutes")
        
        augmented_scenarios = []
        
        for idx, scenario in enumerate(scenarios[:scenarios_to_augment], 1):
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{scenarios_to_augment} | Generated: {len(augmented_scenarios)}")
            
            try:
                variations = self.augment_single_scenario(scenario)
                augmented_scenarios.extend(variations)
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                print(f"    ⚠️  Error augmenting scenario {idx}: {e}")
                continue
        
        print(f"\n✅ Generated {len(augmented_scenarios)} augmented scenarios")
        
        return augmented_scenarios
    
    def augment_scenarios_file(self, input_file: str, output_file: str = None, 
                              target_total: int = 3000) -> str:
        """
        Augment scenarios from a JSON file
        
        Args:
            input_file: Path to input JSON file (relative to data_dir)
            output_file: Path to output file (if None, overwrites input)
            target_total: Target total scenarios after augmentation
        
        Returns:
            Path to output file
        """
        print("\n" + "="*70)
        print("📂 AUGMENTING SCENARIOS FROM FILE")
        print("="*70)
        
        # Load scenarios
        input_path = self.data_dir / input_file
        print(f"\n📥 Loading: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_scenarios = data.get('scenarios', data if isinstance(data, list) else [])
        print(f"   Loaded: {len(original_scenarios)} scenarios")
        
        # Calculate how many to augment
        current_count = len(original_scenarios)
        if current_count >= target_total:
            print(f"\n✅ Already have {current_count} scenarios (target: {target_total})")
            print("   Skipping augmentation")
            return str(input_path)
        
        needed = target_total - current_count
        scenarios_to_augment = min(len(original_scenarios), needed // 2)
        
        # Augment
        augmented = self.augment_scenarios(original_scenarios, target_count=scenarios_to_augment)
        
        # Combine
        combined = original_scenarios + augmented
        
        # Deduplicate
        final_scenarios = deduplicate_scenarios_simple(combined)
        
        # Save
        if output_file is None:
            output_file = input_file
        
        output_path = self.data_dir / output_file
        
        print(f"\n💾 Saving to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_scenarios': len(final_scenarios),
                'original_count': len(original_scenarios),
                'augmented_count': len(augmented),
                'final_count': len(final_scenarios),
                'scenarios': final_scenarios
            }, f, indent=2, ensure_ascii=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print("AUGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"\n📊 Statistics:")
        print(f"   Original scenarios:  {len(original_scenarios)}")
        print(f"   Augmented scenarios: {len(augmented)}")
        print(f"   Final total:         {len(final_scenarios)}")
        print(f"   File size:           {file_size:.2f} MB")
        
        if len(final_scenarios) >= target_total:
            print(f"\n✅ TARGET ACHIEVED: {len(final_scenarios)}/{target_total} scenarios!")
        else:
            print(f"\n⚠️  Close: {len(final_scenarios)}/{target_total} scenarios")
        
        return str(output_path)


def deduplicate_scenarios_simple(scenarios: List[Dict]) -> List[Dict]:
    """Simple deduplication helper"""
    import re
    
    seen = set()
    unique = []
    
    for scenario in scenarios:
        # Create signature
        emergency = scenario.get('emergency_type', scenario.get('title', '')).lower().strip()
        desc = scenario.get('description', '')[:150].lower().strip()
        
        # Normalize
        emergency = re.sub(r'\s+', ' ', emergency)
        desc = re.sub(r'\s+', ' ', desc)
        
        sig = f"{emergency}:{desc}"
        
        if sig not in seen:
            seen.add(sig)
            unique.append(scenario)
    
    return unique


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🚀 SCENARIO AUGMENTATION - STANDALONE MODE")
    print("="*70)
    
    # Check if input file provided
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print("   python collectors/augmentation.py <input_file> [target_total]")
        print("\nExample:")
        print("   python collectors/augmentation.py all_authoritative_scenarios.json 3000")
        print("\nDefault:")
        print("   If no arguments, will look for 'all_authoritative_scenarios.json'")
        
        # Try default file
        default_file = "all_authoritative_scenarios.json"
        data_dir = Path("./data")
        default_path = data_dir / default_file
        
        if default_path.exists():
            print(f"\n✅ Found default file: {default_path}")
            input_file = default_file
            target_total = 3000
        else:
            print(f"\n❌ Default file not found: {default_path}")
            print("\nAvailable files in data directory:")
            if data_dir.exists():
                json_files = list(data_dir.glob("*.json"))
                if json_files:
                    for f in json_files:
                        size = f.stat().st_size / 1024
                        print(f"   - {f.name} ({size:.1f} KB)")
                else:
                    print("   (no JSON files found)")
            else:
                print(f"   (data directory not found: {data_dir})")
            sys.exit(1)
    else:
        input_file = sys.argv[1]
        target_total = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    
    # Run augmentation
    try:
        augmentor = ScenarioAugmentor(data_dir="./data")
        output_file = augmentor.augment_scenarios_file(
            input_file=input_file,
            target_total=target_total
        )
        
        print(f"\n✅ SUCCESS! Output saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("\nMake sure the input file exists in the ./data directory")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)