import os
import json
import re
from typing import List, Dict
from abc import ABC, abstractmethod
import requests


class BaseCollector(ABC):
    """Abstract base class for data collectors - Using Ollama"""
    
    def __init__(self, data_dir: str = "../data", ollama_model: str = "llama3.2"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Test Ollama connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✅ Ollama connected. Available models: {', '.join(model_names)}")
                
                # Check if requested model exists
                if not any(ollama_model in name for name in model_names):
                    print(f"⚠️  Model '{ollama_model}' not found. Installing...")
                    self._pull_model(ollama_model)
            else:
                print("❌ Ollama not responding. Make sure Ollama is running!")
                print("   Install: https://ollama.ai")
                print("   Then run: ollama serve")
                raise ConnectionError("Ollama not available")
        except requests.exceptions.RequestException:
            print("❌ Cannot connect to Ollama!")
            print("   1. Install Ollama: https://ollama.ai")
            print("   2. Start Ollama: ollama serve")
            print("   3. Pull a model: ollama pull llama3.2")
            raise ConnectionError("Ollama not available")
    
    def _pull_model(self, model_name: str):
        """Pull an Ollama model if not available"""
        print(f"📥 Pulling model {model_name}... (this may take a few minutes)")
        pull_url = "http://localhost:11434/api/pull"
        try:
            response = requests.post(
                pull_url,
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        print(f"  {data['status']}")
            print(f"✅ Model {model_name} ready!")
        except Exception as e:
            print(f"❌ Failed to pull model: {e}")
            raise
    
    @abstractmethod
    def collect(self) -> List[Dict]:
        """Collect scenarios from the source"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this data source"""
        pass
    
    def _extract_with_ollama(self, text: str, source: str) -> List[Dict]:
        """Extract scenarios using Ollama"""
        
        if len(text) < 200:
            return []
        
        # Build detailed prompt
        prompt = f"""You are a medical first aid expert. Extract MULTIPLE first aid scenarios from this text.

CRITICAL RULES:
1. Extract 3-5 scenarios if the text covers multiple topics
2. Create separate scenarios for different injury types
3. Create separate scenarios for different severity levels
4. Each scenario MUST have at least 3 immediate steps
5. Be specific with titles (not generic)

Text to analyze:
{text[:4500]}

For each scenario, provide this EXACT JSON structure:
{{
  "title": "Specific clear title",
  "category": "Wounds|Burns|Musculoskeletal|Cardiac|Respiratory|Poisoning|Environmental|Bites/Stings|Allergic|General|Neurological|Shock",
  "subcategory": "Specific type",
  "severity": "minor|moderate|severe|critical",
  "age_group": "all|child|adult|elderly",
  "symptoms": ["symptom1", "symptom2", "symptom3"],
  "immediate_steps": ["step1", "step2", "step3", "step4", "step5"],
  "when_to_seek_help": ["condition1", "condition2"],
  "do_not": ["action1", "action2"],
  "additional_info": "Important notes"
}}

Return ONLY a JSON array of scenarios. No other text or explanation."""

        try:
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "top_p": 0.9
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Parse the response
                scenarios = self._parse_llm_response(response_text, source)
                return scenarios
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            print("⚠️  Ollama timeout - text may be too long")
            return []
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return []
    
    def _extract_rule_based(self, text: str, source: str) -> List[Dict]:
        """Enhanced rule-based extraction (fallback method)"""
        
        scenarios = []
        
        # Pattern matching for common first aid topics
        patterns = {
            'Burns': r'(burn|scald|thermal injury)',
            'Wounds': r'(cut|wound|bleeding|laceration|gash)',
            'Cardiac': r'(heart attack|cardiac arrest|chest pain|CPR)',
            'Respiratory': r'(choking|breathing|airway|asthma|suffocation)',
            'Musculoskeletal': r'(fracture|broken bone|sprain|dislocation|strain)',
            'Poisoning': r'(poison|toxic|overdose|ingestion)',
            'Bites/Stings': r'(bite|sting|snake|insect|animal)',
            'Environmental': r'(heatstroke|hypothermia|frostbite|drowning)',
            'Shock': r'(shock|unconscious|faint)',
            'Allergic': r'(allergy|allergic|anaphyla)',
        }
        
        # Split into sections (more aggressive)
        sections = self._split_into_sections(text)
        
        for section in sections:
            if len(section) < 150:
                continue
                
            # Detect category
            category = 'General'
            for cat, pattern in patterns.items():
                if re.search(pattern, section, re.IGNORECASE):
                    category = cat
                    break
            
            # Extract components
            title = self._extract_title(section)
            steps = self._extract_steps(section)
            symptoms = self._extract_symptoms(section)
            warnings = self._extract_warnings(section)
            donts = self._extract_donts(section)
            
            # Only include if we have meaningful content (at least 3 steps)
            if title and len(steps) >= 3:
                scenario = {
                    'title': title,
                    'category': category,
                    'subcategory': self._infer_subcategory(title, category),
                    'severity': self._infer_severity(section),
                    'age_group': 'all',
                    'symptoms': symptoms,
                    'immediate_steps': steps,
                    'when_to_seek_help': warnings,
                    'do_not': donts,
                    'additional_info': '',
                    'source': source,
                    'source_type': 'authoritative',
                    'confidence': 0.7
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _gpt_extract_scenarios(self, text: str, source: str, source_type: str = "authoritative") -> List[Dict]:
        """Main extraction method - uses Ollama"""
        return self._extract_with_ollama(text, source)
    
    # Helper methods for rule-based extraction
    
    def _split_text(self, text: str, max_chars: int = 3000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= max_chars:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections - more aggressive"""
        # Try multiple splitting strategies
        sections = []
        
        # Strategy 1: Split by headings (all caps, numbers, etc.)
        parts = re.split(r'\n\s*(?:[A-Z][A-Z\s]{10,}|[0-9]+\.|[IVX]+\.|\[.*?\])\s*\n', text)
        sections.extend([s.strip() for s in parts if len(s.strip()) > 150])
        
        # Strategy 2: Split by double newlines
        parts = re.split(r'\n\n+', text)
        sections.extend([s.strip() for s in parts if len(s.strip()) > 150])
        
        # Strategy 3: Split by topic sentences
        parts = re.split(r'(?<=\.)\s+(?=[A-Z][a-z]+\s+(?:is|are|can|may|should))', text)
        sections.extend([s.strip() for s in parts if len(s.strip()) > 150])
        
        return list(set(sections))  # Remove duplicates
    
    def _extract_title(self, text: str) -> str:
        """Extract title from section"""
        lines = text.split('\n')
        
        # Look for short lines (likely headings)
        for line in lines[:3]:
            clean_line = line.strip()
            if 10 < len(clean_line) < 100 and not clean_line.endswith(':'):
                return clean_line
        
        # Look for numbered headings
        match = re.match(r'^(?:\d+\.|[A-Z]+\.)\s+(.+?)(?:\.|$)', text, re.MULTILINE)
        if match:
            return match.group(1).strip()[:100]
        
        # Fall back to first sentence
        sentences = re.split(r'[.!?]', text)
        if sentences:
            return sentences[0].strip()[:100]
        
        return "First Aid Procedure"
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract numbered steps or procedures"""
        steps = []
        
        # Pattern 1: Numbered lists (1., 2., etc.)
        numbered = re.findall(r'(?:^|\n)\s*(\d+[\.)]\s*[^\n]+)', text, re.MULTILINE)
        if numbered:
            steps.extend([re.sub(r'^\d+[\.)]\s*', '', s).strip() for s in numbered])
        
        # Pattern 2: Bullet points
        bullets = re.findall(r'(?:^|\n)\s*[•\-*]\s*([^\n]+)', text)
        if bullets:
            steps.extend([s.strip() for s in bullets])
        
        # Pattern 3: Step keywords
        step_matches = re.findall(
            r'(?:Step \d+:|First,|Then,|Next,|Finally,|After that,)\s*([^.!?]+[.!?])',
            text,
            re.IGNORECASE
        )
        if step_matches:
            steps.extend([s.strip() for s in step_matches])
        
        # Pattern 4: Action verbs at start of sentences
        action_verbs = r'^(?:Check|Call|Apply|Remove|Place|Hold|Press|Cover|Clean|Wash|Elevate|Immobilize|Monitor|Assess|Ensure|Move|Position|Give|Administer)'
        action_steps = re.findall(f'{action_verbs}[^.!?]+[.!?]', text, re.MULTILINE | re.IGNORECASE)
        if action_steps:
            steps.extend([s.strip() for s in action_steps])
        
        # Clean and deduplicate
        unique_steps = []
        seen = set()
        for step in steps:
            clean = step.strip()
            if 15 < len(clean) < 500 and clean.lower() not in seen:
                seen.add(clean.lower())
                unique_steps.append(clean)
        
        return unique_steps[:10]
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms"""
        symptom_indicators = [
            'symptom', 'sign', 'look for', 'may include', 'may experience',
            'characterized by', 'experience', 'feel', 'appear', 'present with'
        ]
        
        symptoms = []
        for indicator in symptom_indicators:
            pattern = f'{indicator}[^.!?]*[.!?]'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                items = re.split(r',\s*(?:and\s+)?|;\s*', match)
                symptoms.extend([s.strip() for s in items if 10 < len(s.strip()) < 150])
        
        return list(set(symptoms))[:8]
    
    def _extract_warnings(self, text: str) -> List[str]:
        """Extract 'when to seek help' warnings"""
        warning_patterns = [
            r'call (?:911|999|emergency|ambulance)',
            r'seek (?:immediate |emergency )?medical (?:help|attention|care)',
            r'go to (?:the )?emergency (?:room|department)',
            r'if.*(?:worsens|severe|unconscious|breathing|bleeding)'
        ]
        
        warnings = []
        for pattern in warning_patterns:
            matches = re.findall(f'[^.!?]*{pattern}[^.!?]*[.!?]', text, re.IGNORECASE)
            warnings.extend([m.strip() for m in matches])
        
        return list(set(warnings))[:5]
    
    def _extract_donts(self, text: str) -> List[str]:
        """Extract 'do not' warnings"""
        dont_pattern = r"(?:do not|don't|never|avoid|must not)[^.!?]+[.!?]"
        matches = re.findall(dont_pattern, text, re.IGNORECASE)
        return list(set([m.strip() for m in matches]))[:5]
    
    def _infer_subcategory(self, title: str, category: str) -> str:
        """Infer subcategory from title"""
        title_lower = title.lower()
        
        subcategories = {
            'Burns': ['first-degree', 'second-degree', 'third-degree', 'chemical', 'electrical', 'thermal'],
            'Wounds': ['minor cut', 'deep cut', 'puncture', 'abrasion', 'laceration', 'amputation'],
            'Cardiac': ['heart attack', 'cardiac arrest', 'angina', 'CPR'],
            'Respiratory': ['choking', 'asthma', 'hyperventilation', 'drowning'],
            'Musculoskeletal': ['fracture', 'sprain', 'strain', 'dislocation'],
        }
        
        if category in subcategories:
            for subcat in subcategories[category]:
                if subcat in title_lower:
                    return subcat.title()
        
        return category
    
    def _infer_severity(self, text: str) -> str:
        """Infer severity from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['critical', 'life-threatening', '911', 'emergency', 'immediately']):
            return 'critical'
        elif any(word in text_lower for word in ['severe', 'serious', 'deep', 'major', 'heavy']):
            return 'severe'
        elif any(word in text_lower for word in ['moderate', 'significant']):
            return 'moderate'
        else:
            return 'minor'
    
    def _parse_llm_response(self, response: str, source: str) -> List[Dict]:
        """Parse LLM response into structured scenarios"""
        scenarios = []
        
        try:
            # Clean response - extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'title' in item and len(item.get('immediate_steps', [])) >= 3:
                            item['source'] = source
                            item['source_type'] = 'authoritative'
                            item['confidence'] = 0.85
                            scenarios.append(item)
            
        except json.JSONDecodeError:
            # Try to find individual JSON objects
            objects = re.findall(r'\{[^{}]*"title"[^{}]*\}', response, re.DOTALL)
            for obj_str in objects:
                try:
                    item = json.loads(obj_str)
                    if 'title' in item and len(item.get('immediate_steps', [])) >= 3:
                        item['source'] = source
                        item['source_type'] = 'authoritative'
                        item['confidence'] = 0.85
                        scenarios.append(item)
                except:
                    continue
        
        return scenarios
    
    def save_checkpoint(self, scenarios: List[Dict], filename: str):
        """Save intermediate results"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved: {filename} ({len(scenarios)} scenarios)")