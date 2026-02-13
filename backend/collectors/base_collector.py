import os
import json
import re
import logging
import requests
import time
from typing import List, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    
    def __init__(self, data_dir: str = "../data", ollama_model: str = "llama3.2"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.logger = logger  
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                self.logger.info(f"Ollama connected. Available models: {', '.join(model_names)}")
                
                if not any(ollama_model in name for name in model_names):
                    self.logger.warning(f"Model '{ollama_model}' not found. Installing...")
                    self.install_model(ollama_model)
            else:
                self.logger.error("Ollama not responding. Run: ollama serve")
        except requests.exceptions.RequestException:
            self.logger.error("Ollama not available. Ensure it is installed and running.")

    def install_model(self, model_name: str):

        self.logger.info(f"Pulling model {model_name}... (this may take a few minutes)")
        pull_url = "http://localhost:11434/api/pull"
        try:
            response = requests.post(pull_url, json={"name": model_name}, stream=True, timeout=300)
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        self.logger.debug(f"Pull status: {data['status']}")
            self.logger.info(f"Model {model_name} ready!")
        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            raise

    @abstractmethod
    def collect(self) -> List[Dict]:
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        pass

    def _extract_with_ollama(self, text: str, source: str) -> List[Dict]:

        if len(text) < 200:
            return []
        
        prompt = f"""You are a medical first aid expert. Extract MULTIPLE first aid scenarios from this text.

CRITICAL RULES:
1. Extract 3-5 scenarios if the text covers multiple topics
2. Each scenario MUST have at least 3 immediate steps
3. Return ONLY a JSON array.

Text to analyze:
{text[:4500]}

JSON Structure:
{{
  "title": "Specific clear title",
  "category": "Wounds|Burns|Musculoskeletal|Cardiac|Respiratory|Poisoning|Environmental|Bites/Stings|Allergic|General|Neurological|Shock",
  "subcategory": "Specific type",
  "severity": "minor|moderate|severe|critical",
  "age_group": "all|child|adult|elderly",
  "symptoms": ["symptom1", "symptom2"],
  "immediate_steps": ["step1", "step2", "step3"],
  "when_to_seek_help": ["condition1"],
  "do_not": ["action1"],
  "additional_info": "Notes"
}}"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "temperature": 0.3
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return self.clean_response(result.get('response', ''), source)
            return []
                
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            return []

    def clean_response(self, response_text: str, source: str) -> List[Dict]:

        try:

            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r'^```(?:json)?|```$', '', cleaned_text, flags=re.MULTILINE).strip()

            json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                if isinstance(data, dict):
                    data = [data]
                
                scenarios = []
                for item in data:
                    if isinstance(item, dict) and 'title' in item:
                        item.update({
                            'source': source,
                            'source_type': 'authoritative',
                            'confidence': 0.85,
                            'extracted_at': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        scenarios.append(item)
                return scenarios
            
            self.logger.warning(f"No valid JSON array found in response for {source}")
            
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON from LLM response for {source}")
        except Exception as e:
            self.logger.error(f"Error during response cleaning for {source}: {e}")
            
        return []

    def save_checkpoint(self, scenarios: List[Dict], filename: str):
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(scenarios, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved: {filename} ({len(scenarios)} scenarios)")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {filename}: {e}")