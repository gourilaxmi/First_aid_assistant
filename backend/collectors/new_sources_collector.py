"""
New Sources Collector - 10 Authoritative Medical Sources
Collects from: AHA, WHO, MedlinePlus, Red Cross Online, Poison Control,
               KidsHealth, FamilyDoctor, EmergencyCareForYou, AAP, Johns Hopkins

Usage from master_pipeline.py:
    from collectors.new_sources_collector import NewSourcesCollector
    collector = NewSourcesCollector(data_dir="./data")
    collector.collect_all_new_sources()
"""

import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
from pathlib import Path
import urllib3
from .base_collector import BaseCollector

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class NewSourcesCollector(BaseCollector):
    """Collects scenarios from 10 new authoritative sources"""
    
    def __init__(self, data_dir: str = "./data", ollama_model: str = "llama3.2"):
        super().__init__(data_dir=data_dir, ollama_model=ollama_model)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.verify = False
    
    def get_source_name(self) -> str:
        return "New Sources Collector"
    
    def _fetch_page_safe(self, url: str, timeout: int = 15) -> BeautifulSoup:
        """Fetch page content safely"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"    ⚠️  {str(e)[:80]}")
            return None
    
    def _extract_text_from_soup(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text from selectors"""
        if not soup:
            return ""
        
        text_parts = []
        for selector in selectors:
            try:
                elements = soup.select(selector)
                for elem in elements:
                    for tag in elem(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    text = elem.get_text(strip=True, separator='\n')
                    if len(text) > 100:
                        text_parts.append(text)
            except:
                pass
        return "\n\n".join(text_parts)
    
    def discover_and_collect(self, base_url: str, discovery_pages: List[str], 
                            keywords: List[str], source_name: str, 
                            max_pages: int = 50) -> List[Dict]:
        """Generic discover and collect method"""
        
        print(f"\n{'='*70}")
        print(f"COLLECTING: {source_name.upper()}")
        print(f"{'='*70}")
        
        all_links = set()
        
        # Discover pages
        print(f"🔍 Discovering {source_name} pages...")
        for page in discovery_pages:
            try:
                url = page if page.startswith('http') else base_url + page
                soup = self._fetch_page_safe(url)
                if soup:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(kw in href.lower() for kw in keywords):
                            if href.startswith('/'):
                                all_links.add(base_url + href)
                            elif base_url.replace('https://', '').replace('www.', '') in href:
                                all_links.add(href)
                time.sleep(1)
            except:
                pass
        
        print(f"📄 Found {len(all_links)} pages")
        
        # Collect scenarios
        scenarios = []
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                if idx % 10 == 0:
                    print(f"  Progress: {idx}/{min(len(all_links), max_pages)} | Scenarios: {len(scenarios)}")
                
                soup = self._fetch_page_safe(url)
                if soup:
                    content = self._extract_text_from_soup(
                        soup, 
                        ['article', 'main', '.content', '[role="main"]', '.page-content']
                    )
                    if len(content) > 300:
                        extracted = self._gpt_extract_scenarios(content, source_name)
                        scenarios.extend(extracted)
                
                time.sleep(1.5)
            except:
                continue
        
        print(f"✅ {source_name}: {len(scenarios)} scenarios collected")
        return scenarios
    
    # ========================================================================
    # 10 SOURCE COLLECTORS
    # ========================================================================
    
    def collect_american_heart_association(self) -> List[Dict]:
        """American Heart Association - CPR & Cardiovascular"""
        return self.discover_and_collect(
            base_url="https://cpr.heart.org",
            discovery_pages=[
                "/en/resources/cpr-facts-and-stats",
                "/en/cpr-courses-and-kits/hands-only-cpr",
                "/en/emergency-cardiovascular-care"
            ],
            keywords=['cpr', 'emergency', 'cardiac', 'heart', 'stroke'],
            source_name="American Heart Association",
            max_pages=50
        )
    
    def collect_who_emergency(self) -> List[Dict]:
        """WHO Emergency Care"""
        return self.discover_and_collect(
            base_url="https://www.who.int",
            discovery_pages=[
                "/health-topics/emergency-care",
                "/health-topics/injuries"
            ],
            keywords=['emergency', 'injury', 'trauma', 'first-aid'],
            source_name="WHO Emergency Care",
            max_pages=40
        )
    
    def collect_medlineplus(self) -> List[Dict]:
        """MedlinePlus (NIH)"""
        return self.discover_and_collect(
            base_url="https://medlineplus.gov",
            discovery_pages=[
                "/firstaid.html",
                "/emergencies.html"
            ],
            keywords=['first', 'emergency', 'injury', 'ency'],
            source_name="MedlinePlus NIH",
            max_pages=80
        )
    
    def collect_redcross_online(self) -> List[Dict]:
        """Red Cross Online (supplement to PDF)"""
        return self.discover_and_collect(
            base_url="https://www.redcross.org",
            discovery_pages=[
                "/get-help/how-to-prepare-for-emergencies",
                "/take-a-class/first-aid"
            ],
            keywords=['first-aid', 'emergency', 'prepare'],
            source_name="Red Cross Online",
            max_pages=60
        )
    
    def collect_poison_control(self) -> List[Dict]:
        """National Poison Control"""
        return self.discover_and_collect(
            base_url="https://www.poison.org",
            discovery_pages=["/articles"],
            keywords=['poison', 'articles', 'swallow', 'toxic'],
            source_name="Poison Control",
            max_pages=50
        )
    
    def collect_kidshealth(self) -> List[Dict]:
        """KidsHealth - Pediatric Emergencies"""
        return self.discover_and_collect(
            base_url="https://kidshealth.org",
            discovery_pages=[
                "/en/parents/firstaid-safe.html",
                "/en/teens/safety.html"
            ],
            keywords=['first', 'safety', 'emergency'],
            source_name="KidsHealth",
            max_pages=40
        )
    
    def collect_familydoctor(self) -> List[Dict]:
        """FamilyDoctor.org"""
        return self.discover_and_collect(
            base_url="https://familydoctor.org",
            discovery_pages=[
                "/prevention-wellness/staying-healthy/first-aid/"
            ],
            keywords=['first-aid', 'emergency', 'injury'],
            source_name="FamilyDoctor",
            max_pages=40
        )
    
    def collect_emergencycarefor_you(self) -> List[Dict]:
        """EmergencyCareForYou.org"""
        return self.discover_and_collect(
            base_url="https://www.emergencycareforyou.org",
            discovery_pages=["/emergency-101/"],
            keywords=['emergency', 'when-to-go', 'health'],
            source_name="EmergencyCareForYou",
            max_pages=30
        )
    
    def collect_aap_publications(self) -> List[Dict]:
        """American Academy of Pediatrics"""
        return self.discover_and_collect(
            base_url="https://www.healthychildren.org",
            discovery_pages=[
                "/English/health-issues/injuries-emergencies"
            ],
            keywords=['emergency', 'injury', 'safety'],
            source_name="AAP HealthyChildren",
            max_pages=40
        )
    
    def collect_johns_hopkins_medicine(self) -> List[Dict]:
        """Johns Hopkins Medicine"""
        return self.discover_and_collect(
            base_url="https://www.hopkinsmedicine.org",
            discovery_pages=[
                "/health/treatment-tests-and-therapies"
            ],
            keywords=['emergency', 'first-aid', 'treatment'],
            source_name="Johns Hopkins Medicine",
            max_pages=40
        )
    
    # ========================================================================
    # MAIN COLLECTION METHOD
    # ========================================================================
    
    def collect_all_new_sources(self) -> Dict[str, List[Dict]]:
        """
        Collect from all 10 new sources
        
        Returns:
            Dictionary mapping source name to scenarios list
        """
        print("\n" + "="*70)
        print("🚀 NEW SOURCES COLLECTION - 10 AUTHORITATIVE SOURCES")
        print("="*70)
        print("Estimated time: 4-6 hours")
        print("="*70)
        
        sources = [
            ("American Heart Association", self.collect_american_heart_association),
            ("WHO Emergency Care", self.collect_who_emergency),
            ("MedlinePlus NIH", self.collect_medlineplus),
            ("Red Cross Online", self.collect_redcross_online),
            ("Poison Control", self.collect_poison_control),
            ("KidsHealth", self.collect_kidshealth),
            ("FamilyDoctor", self.collect_familydoctor),
            ("EmergencyCareForYou", self.collect_emergencycarefor_you),
            ("AAP HealthyChildren", self.collect_aap_publications),
            ("Johns Hopkins Medicine", self.collect_johns_hopkins_medicine),
        ]
        
        all_results = {}
        
        for idx, (name, collect_func) in enumerate(sources, 1):
            print(f"\n[{idx}/{len(sources)}] Starting: {name}")
            try:
                scenarios = collect_func()
                all_results[name] = scenarios
                
                # Save checkpoint
                checkpoint_file = Path(self.data_dir) / f"checkpoint_{name.lower().replace(' ', '_')}.json"
                self.save_checkpoint(scenarios, checkpoint_file.name)
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                all_results[name] = []
        
        # Summary
        print(f"\n{'='*70}")
        print("COLLECTION COMPLETE")
        print(f"{'='*70}")
        
        total_scenarios = sum(len(s) for s in all_results.values())
        print(f"\n📊 Total scenarios collected: {total_scenarios}")
        print(f"\n📚 Source breakdown:")
        for source, scenarios in sorted(all_results.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   {source:30s}: {len(scenarios):4d} scenarios")
        
        return all_results
    
    def collect(self) -> List[Dict]:
        """Override base collect method"""
        results = self.collect_all_new_sources()
        # Flatten all scenarios
        all_scenarios = []
        for scenarios in results.values():
            all_scenarios.extend(scenarios)
        return all_scenarios