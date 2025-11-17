"""
Enhanced Web Scraping with Ollama for Extraction
Target: 600+ unique scenarios from Mayo Clinic, NHS.UK, St John Ambulance, 
        WebMD, Cleveland Clinic, Healthline, CDC Emergency Preparedness
Excludes: Johns Hopkins (links not working)
Deduplicates: Against Red Cross PDF scenarios
"""

import time
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Set
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .base_collector import BaseCollector


import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings (optional, for cleaner output)
urllib3.disable_warnings(InsecureRequestWarning)


class EnhancedWebCollector(BaseCollector):
    """Base class with BeautifulSoup, Selenium, and Ollama support"""
    
    def __init__(
        self, 
        data_dir: str = "../data", 
        use_selenium: bool = False,
        ollama_model: str = "llama3.2",
        existing_scenarios_file: str = None,
        verify_ssl: bool = False  # NEW: Disable SSL verification by default
    ):
        super().__init__(data_dir, ollama_model)
        
        self.use_selenium = use_selenium
        self.verify_ssl = verify_ssl  # NEW
        self.existing_scenarios: Set[str] = set()
        
        # Load existing scenarios for deduplication
        if existing_scenarios_file:
            self._load_existing_scenarios(existing_scenarios_file)
        
        # Setup requests session with SSL configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.verify = verify_ssl  # NEW: Apply SSL setting
        
        # Setup Selenium if needed
        if self.use_selenium:
            self._setup_selenium()
    
    def _load_existing_scenarios(self, filepath: str):
        """Load existing scenarios from fast_scenarios.json to avoid duplicates"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                scenarios = data if isinstance(data, list) else data.get('scenarios', [])
                
                for scenario in scenarios:
                    # Create normalized signature for deduplication
                    sig = self._normalize_scenario_signature(scenario)
                    self.existing_scenarios.add(sig)
                
                print(f"✅ Loaded {len(self.existing_scenarios)} existing scenarios for deduplication")
        except Exception as e:
            print(f"⚠️  Could not load existing scenarios: {e}")
    
    def _normalize_scenario_signature(self, scenario: Dict) -> str:
        """Create a normalized signature for scenario comparison"""
        emergency = scenario.get('emergency_type', '').lower().strip()
        desc = scenario.get('description', '')[:100].lower().strip()
        return f"{emergency}:{desc}"
    
    def _is_duplicate(self, scenario: Dict) -> bool:
        """Check if scenario is duplicate of existing Red Cross scenarios"""
        sig = self._normalize_scenario_signature(scenario)
        return sig in self.existing_scenarios
    
    def _setup_selenium(self):
        """Initialize Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✅ Selenium WebDriver initialized")
        except Exception as e:
            print(f"⚠️  Selenium initialization failed: {e}")
            print("   Falling back to requests only")
            self.use_selenium = False
            self.driver = None

    def _fetch_page(self, url: str, timeout: int = 15, wait_for: str = None) -> BeautifulSoup:
        """Fetch page with requests or Selenium"""
        try:
            if self.use_selenium and self.driver:
                self.driver.get(url)
                
                if wait_for:
                    WebDriverWait(self.driver, timeout).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                    )
                else:
                    time.sleep(2)
                
                page_source = self.driver.page_source
                return BeautifulSoup(page_source, 'html.parser')
            else:
                # NEW: Pass verify parameter explicitly
                response = self.session.get(
                    url, 
                    timeout=timeout,
                    verify=self.verify_ssl  # Use instance setting
                )
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"  ⚠️  Error fetching {url}: {e}")
            return None
    
    def _extract_text(self, soup: BeautifulSoup, selectors: List[str] = None) -> str:
        """Extract text from specific selectors or entire page"""
        if soup is None:
            return ""
        
        if selectors:
            text_parts = []
            for selector in selectors:
                elements = soup.select(selector)
                for elem in elements:
                    # Remove script and style tags
                    for script in elem(["script", "style", "nav", "footer", "header", "aside"]):
                        script.decompose()
                    text = elem.get_text(strip=True, separator='\n')
                    if len(text) > 100:  # Only add substantial content
                        text_parts.append(text)
            return "\n\n".join(text_parts)
        else:
            # Remove unwanted tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            return soup.get_text(separator='\n')[:10000]  # Increased limit
    
    def _polite_wait(self, seconds: float = 1.5):
        """Wait between requests to be polite"""
        time.sleep(seconds)
    
    def __del__(self):
        """Cleanup Selenium driver"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except:
                pass


class MayoClinicCollector(EnhancedWebCollector):
    """Mayo Clinic scraper with Ollama extraction - Enhanced for 150+ scenarios"""
    
    BASE_URL = "https://www.mayoclinic.org"
    
    # Comprehensive first aid URLs
    FIRST_AID_URLS = [
        # Core emergencies
        "/first-aid/first-aid-bleeding/basics/art-20056661",
        "/first-aid/first-aid-burns/basics/art-20056649",
        "/first-aid/first-aid-cpr/basics/art-20056600",
        "/first-aid/first-aid-choking/basics/art-20056637",
        "/first-aid/first-aid-fractures/basics/art-20056641",
        "/first-aid/first-aid-heart-attack/basics/art-20056679",
        "/first-aid/first-aid-shock/basics/art-20056620",
        "/first-aid/first-aid-sprain/basics/art-20056622",
        "/first-aid/first-aid-stroke/basics/art-20056654",
        "/first-aid/first-aid-poisoning/basics/art-20056657",
        "/first-aid/first-aid-nosebleeds/basics/art-20056683",
        "/first-aid/first-aid-frostbite/basics/art-20056653",
        "/first-aid/first-aid-heat-exhaustion/basics/art-20056651",
        "/first-aid/first-aid-hypothermia/basics/art-20056624",
        "/first-aid/first-aid-seizures/basics/art-20056638",
        # Additional emergencies
        "/first-aid/first-aid-cuts/basics/art-20056711",
        "/first-aid/first-aid-bee-stings/basics/art-20056743",
        "/first-aid/first-aid-snake-bites/basics/art-20056681",
        "/first-aid/first-aid-animal-bites/basics/art-20056591",
        "/first-aid/first-aid-insect-bites/basics/art-20056593",
        "/first-aid/first-aid-tick-bites/basics/art-20056641",
        "/first-aid/first-aid-dental-emergencies/basics/art-20056597",
        "/first-aid/first-aid-eye-emergencies/basics/art-20056645",
        "/first-aid/first-aid-drowning/basics/art-20056649",
        "/first-aid/first-aid-electric-shock/basics/art-20056695",
        "/first-aid/first-aid-concussion/basics/art-20056626",
        "/first-aid/first-aid-head-trauma/basics/art-20056626",
        "/first-aid/first-aid-severe-bleeding/basics/art-20056661",
        "/first-aid/first-aid-puncture-wounds/basics/art-20056665",
    ]
    
    def get_source_name(self) -> str:
        return "Mayo Clinic"
    
    def collect(self, max_pages: int = 150) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: MAYO CLINIC (Target: 150+ scenarios)")
        print("="*60)
        
        scenarios = []
        processed = 0
        duplicates_filtered = 0
        
        print(f"📄 Processing {len(self.FIRST_AID_URLS)} Mayo Clinic pages...")
        
        for url_path in self.FIRST_AID_URLS[:max_pages]:
            try:
                full_url = self.BASE_URL + url_path
                topic = url_path.split('/')[-2].replace('first-aid-', '').replace('-', ' ').title()
                print(f"  → {topic}")
                
                soup = self._fetch_page(full_url)
                
                if soup:
                    # Mayo Clinic specific content selectors
                    content = self._extract_text(
                        soup,
                        selectors=[
                            'article .content',
                            '.main-content',
                            '[itemprop="articleBody"]',
                            '.content-wrapper',
                            '.article-body',
                            'main article'
                        ]
                    )
                    
                    if len(content) > 300:
                        # Use Ollama to extract MORE scenarios per page
                        extracted = self._gpt_extract_scenarios(
                            content,
                            f"Mayo Clinic: {topic}",
                        )
                        
                        # Filter duplicates
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        
                        scenarios.extend(unique_scenarios)
                        processed += 1
                        
                        if len(unique_scenarios) > 0:
                            print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios ({len(extracted) - len(unique_scenarios)} filtered)")
                        else:
                            print(f"    ⚠️  No unique scenarios extracted")
                
                self._polite_wait(1.5)
                
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                continue
        
        print(f"\n✅ Mayo Clinic: {len(scenarios)} scenarios from {processed} pages")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "mayo_scenarios.json")
        return scenarios


class ClevelandClinicCollector(EnhancedWebCollector):
    """Cleveland Clinic scraper - Target 100+ scenarios"""
    
    BASE_URL = "https://my.clevelandclinic.org"
    
    DISCOVERY_URLS = [
        "/health/treatments/emergencies",
        "/health/treatments/first-aid",
        "/health/articles/emergency-care",
        "/health/treatments/injuries",
        "/health/diseases/emergency-conditions",
    ]
    
    def get_source_name(self) -> str:
        return "Cleveland Clinic"
    
    def collect(self, max_pages: int = 120) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: CLEVELAND CLINIC (Target: 100+ scenarios)")
        print("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        # Discover all emergency/first aid links
        print("🔍 Discovering Cleveland Clinic pages...")
        for base_path in self.DISCOVERY_URLS:
            try:
                soup = self._fetch_page(self.BASE_URL + base_path)
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if any(term in href.lower() for term in 
                              ['first-aid', 'emergency', 'treatment', 'injury', 'accident', 
                               'trauma', 'bleeding', 'burn', 'fracture', 'poisoning']):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                print(f"  ⚠️  Discovery error: {e}")
        
        print(f"📄 Found {len(all_links)} Cleveland Clinic pages")
        
        # Scrape discovered pages
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                print(f"  → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=[
                            'article',
                            '.article-body',
                            '.content-body',
                            'main',
                            '[class*="content"]'
                        ]
                    )
                    
                    if len(content) > 300:
                        extracted = self._gpt_extract_scenarios(
                            content,
                            f"Cleveland Clinic: {url.split('/')[-1]}",
                        )
                        
                        # Filter duplicates
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        scenarios.extend(unique_scenarios)
                        
                        if len(unique_scenarios) > 0:
                            print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios")
                
                if idx % 10 == 0:
                    print(f"  📊 Progress: {idx} pages | {len(scenarios)} scenarios")
                
                self._polite_wait(1.5)
                
            except Exception as e:
                continue
        
        print(f"\n✅ Cleveland Clinic: {len(scenarios)} scenarios")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "cleveland_scenarios.json")
        return scenarios


class HealthlineCollector(EnhancedWebCollector):
    """Healthline scraper - Target 100+ scenarios"""
    
    BASE_URL = "https://www.healthline.com"
    
    DISCOVERY_URLS = [
        "/health/first-aid",
        "/health/emergency-medicine",
        "/health-news/emergency-care",
        "/health/injuries",
        "/health/treatments",
    ]
    
    def get_source_name(self) -> str:
        return "Healthline"
    
    def collect(self, max_pages: int = 120) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: HEALTHLINE (Target: 100+ scenarios)")
        print("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        print("🔍 Discovering Healthline pages...")
        for base_path in self.DISCOVERY_URLS:
            try:
                soup = self._fetch_page(self.BASE_URL + base_path)
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if any(term in href.lower() for term in 
                              ['first-aid', 'emergency', 'treatment', 'injury', 'how-to-treat',
                               'bleeding', 'burn', 'fracture', 'bite', 'sting']):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            if 'healthline.com' in full_url:
                                all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                print(f"  ⚠️  Discovery error: {e}")
        
        print(f"📄 Found {len(all_links)} Healthline pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                print(f"  → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=[
                            'article',
                            '[class*="article-body"]',
                            '.content-wrapper',
                            'main',
                            '[id*="content"]'
                        ]
                    )
                    
                    if len(content) > 300:
                        extracted = self._gpt_extract_scenarios(
                            content,
                            f"Healthline: {url.split('/')[-1]}",
                        )
                        
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        scenarios.extend(unique_scenarios)
                        
                        if len(unique_scenarios) > 0:
                            print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios")
                
                if idx % 10 == 0:
                    print(f"  📊 Progress: {idx} pages | {len(scenarios)} scenarios")
                
                self._polite_wait(1.5)
                
            except Exception as e:
                continue
        
        print(f"\n✅ Healthline: {len(scenarios)} scenarios")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "healthline_scenarios.json")
        return scenarios


class CDCEmergencyCollector(EnhancedWebCollector):
    """CDC Emergency Preparedness - Target 80+ scenarios"""
    
    BASE_URL = "https://www.cdc.gov"
    
    DISCOVERY_URLS = [
        "/disasters/index.html",
        "/cpr/index.htm",
        "/disasters/injury/facts.html",
        "/disasters/extremeheat/warning.html",
        "/disasters/winter/beforestorm/supplylists.html",
        "/disasters/extremeheat/heat_guide.html",
        "/disasters/winter/during.html",
        "/disasters/floods/index.html",
    ]
    
    def get_source_name(self) -> str:
        return "CDC Emergency Preparedness"
    
    def collect(self, max_pages: int = 100) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: CDC EMERGENCY (Target: 80+ scenarios)")
        print("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        print("🔍 Discovering CDC pages...")
        for base_path in self.DISCOVERY_URLS:
            try:
                soup = self._fetch_page(self.BASE_URL + base_path)
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if any(term in href.lower() for term in 
                              ['emergency', 'disaster', 'injury', 'first-aid', 'cpr', 
                               'treatment', 'safety', 'preparedness']):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            if 'cdc.gov' in full_url:
                                all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                print(f"  ⚠️  Discovery error: {e}")
        
        print(f"📄 Found {len(all_links)} CDC pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                print(f"  → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=[
                            'article',
                            '.content',
                            'main',
                            '[id="content"]',
                            '.page-content',
                            '[role="main"]'
                        ]
                    )
                    
                    if len(content) > 300:
                        extracted = self._gpt_extract_scenarios(
                            content,
                            f"CDC: {url.split('/')[-1].replace('.html', '')}",
                        )
                        
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        scenarios.extend(unique_scenarios)
                        
                        if len(unique_scenarios) > 0:
                            print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios")
                
                if idx % 10 == 0:
                    print(f"  📊 Progress: {idx} pages | {len(scenarios)} scenarios")
                
                self._polite_wait(2.0)
                
            except Exception as e:
                continue
        
        print(f"\n✅ CDC: {len(scenarios)} scenarios")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "cdc_scenarios.json")
        return scenarios


class NHSCollector(EnhancedWebCollector):
    """NHS UK scraper - Target 100+ scenarios"""
    
    BASE_URL = "https://www.nhs.uk"
    
    NHS_FIRST_AID_PAGES = [
        "/conditions/accidents-and-first-aid/",
        "/conditions/first-aid/",
        "/common-health-questions/accidents-first-aid-and-treatments/",
    ]
    
    def get_source_name(self) -> str:
        return "NHS UK"
    
    def collect(self, max_pages: int = 120) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: NHS UK (Target: 100+ scenarios)")
        print("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        print("🔍 Discovering NHS first aid pages...")
        for base_path in self.NHS_FIRST_AID_PAGES:
            try:
                soup = self._fetch_page(self.BASE_URL + base_path)
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if any(term in href.lower() for term in 
                              ['first-aid', 'accident', 'emergency', 'injury', 'treatment', 
                               'conditions', 'bleeding', 'burn', 'fracture', 'bite']):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                print(f"  ⚠️  Discovery error: {e}")
        
        print(f"📄 Found {len(all_links)} NHS pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                print(f"  → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=[
                            'article',
                            '.nhsuk-main-wrapper',
                            '[class*="article"]',
                            '.nhsuk-body-m',
                            'main'
                        ]
                    )
                    
                    if len(content) > 300:
                        extracted = self._gpt_extract_scenarios(
                            content,
                            f"NHS UK: {url.split('/')[-1] or 'general'}",
                        )
                        
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        scenarios.extend(unique_scenarios)
                        
                        if len(unique_scenarios) > 0:
                            print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios")
                
                if idx % 10 == 0:
                    print(f"  📊 Progress: {idx} pages | {len(scenarios)} scenarios")
                
                self._polite_wait(1.5)
                
            except Exception as e:
                continue
        
        print(f"\n✅ NHS UK: {len(scenarios)} scenarios")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "nhs_scenarios.json")
        return scenarios


class StJohnCollector(EnhancedWebCollector):
    """St John Ambulance - Target 80+ scenarios"""
    
    BASE_URL = "https://www.sja.org.uk"
    ADVICE_BASE = f"{BASE_URL}/get-advice/first-aid-advice/"
    
    def get_source_name(self) -> str:
        return "St John Ambulance"
    
    def collect(self, max_pages: int = 100) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: ST JOHN AMBULANCE (Target: 80+ scenarios)")
        print("="*60)
        
        scenarios = []
        duplicates_filtered = 0
        
        print("🔍 Discovering advice pages...")
        try:
            soup = self._fetch_page(self.ADVICE_BASE)
            if not soup:
                print("❌ Could not access St John Ambulance website")
                return scenarios
            
            advice_links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/get-advice/' in href or '/first-aid/' in href:
                    full_url = href if href.startswith('http') else self.BASE_URL + href
                    advice_links.add(full_url)
            
            print(f"📄 Found {len(advice_links)} advice pages")
            
            for idx, url in enumerate(list(advice_links)[:max_pages], 1):
                try:
                    print(f"  → Page {idx}/{min(len(advice_links), max_pages)}")
                    soup = self._fetch_page(url)
                    
                    if soup:
                        content = self._extract_text(
                            soup,
                            selectors=[
                                'article',
                                '.advice-content',
                                '[class*="content"]',
                                'main',
                                '[role="main"]'
                            ]
                        )
                        
                        if len(content) > 300:
                            extracted = self._gpt_extract_scenarios(
                                content,
                                f"St John Ambulance: {url.split('/')[-1]}",
                            )
                            
                            unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                            duplicates_filtered += len(extracted) - len(unique_scenarios)
                            scenarios.extend(unique_scenarios)
                            
                            if len(unique_scenarios) > 0:
                                print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios")
                    
                    if idx % 10 == 0:
                        print(f"  📊 Progress: {idx} pages | {len(scenarios)} scenarios")
                    
                    self._polite_wait(1.5)
                    
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print(f"\n✅ St John: {len(scenarios)} scenarios")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "stjohn_scenarios.json")
        return scenarios


class WebMDCollector(EnhancedWebCollector):
    """WebMD First Aid - Target 90+ scenarios"""
    
    BASE_URL = "https://www.webmd.com"
    
    FIRST_AID_SECTIONS = [
        "/first-aid/default.htm",
        "/first-aid/emergencies-injuries-directory",
        "/first-aid/understanding-basics-first-aid",
    ]
    
    def get_source_name(self) -> str:
        return "WebMD"
    
    def collect(self, max_pages: int = 110) -> List[Dict]:
        print("\n" + "="*60)
        print("COLLECTING: WEBMD (Target: 90+ scenarios)")
        print("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        print("🔍 Discovering WebMD pages...")
        for section in self.FIRST_AID_SECTIONS:
            try:
                soup = self._fetch_page(self.BASE_URL + section)
                if soup:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if '/first-aid/' in href or '/emergency/' in href:
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                print(f"  ⚠️  Discovery error: {e}")
        
        print(f"📄 Found {len(all_links)} WebMD pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                print(f"  → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=[
                            'article',
                            '[class*="article-body"]',
                            '.content-wrapper',
                            'main',
                            '[id*="article"]'
                        ]
                    )
                    
                    if len(content) > 300:
                        extracted = self._gpt_extract_scenarios(
                            content,
                            f"WebMD: {url.split('/')[-1]}",
                        )
                        
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        scenarios.extend(unique_scenarios)
                        
                        if len(unique_scenarios) > 0:
                            print(f"    ✓ Extracted {len(unique_scenarios)} unique scenarios")
                
                if idx % 10 == 0:
                    print(f"  📊 Progress: {idx} pages | {len(scenarios)} scenarios")
                
                self._polite_wait(2.0)
                
            except Exception as e:
                continue
        
        print(f"\n✅ WebMD: {len(scenarios)} scenarios")
        print(f"   🔍 Filtered {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "webmd_scenarios.json")
        return scenarios


def run_enhanced_collection(
    data_dir: str = "../data",
    existing_scenarios_file: str = None,
    use_selenium: bool = False,
    ollama_model: str = "llama3.2",
    verify_ssl: bool = False  # NEW: Add parameter
) -> Dict:
    """
    Run complete enhanced collection targeting 600+ scenarios
    """
    
    print("\n" + "="*70)
    print("ENHANCED WEB SCRAPING - TARGET: 600+ UNIQUE SCENARIOS")
    print("="*70)
    print(f"Ollama Model: {ollama_model}")
    print(f"Selenium: {'Enabled' if use_selenium else 'Disabled'}")
    print(f"SSL Verification: {'Enabled' if verify_ssl else 'Disabled (Dev Mode)'}")  # NEW
    if existing_scenarios_file:
        print(f"Deduplicating against: {existing_scenarios_file}")
    print("="*70)
    
    all_scenarios = []
    source_breakdown = {}
    
    # Collection targets per source (total 600+)
    collectors = [
        (MayoClinicCollector, 150, "Mayo Clinic"),
        (ClevelandClinicCollector, 120, "Cleveland Clinic"),
        (HealthlineCollector, 120, "Healthline"),
        (NHSCollector, 120, "NHS UK"),
        (StJohnCollector, 100, "St John Ambulance"),
        (WebMDCollector, 110, "WebMD"),
        (CDCEmergencyCollector, 100, "CDC Emergency"),
    ]
    
    for CollectorClass, max_pages, source_name in collectors:
        try:
            collector = CollectorClass(
                data_dir=data_dir,
                use_selenium=use_selenium,
                ollama_model=ollama_model,
                existing_scenarios_file=existing_scenarios_file,
                verify_ssl=verify_ssl  # NEW: Pass SSL setting
            )
            
            scenarios = collector.collect(max_pages=max_pages)
            all_scenarios.extend(scenarios)
            source_breakdown[source_name] = len(scenarios)
            
        except Exception as e:
            print(f"\n❌ Error collecting from {source_name}: {e}")
            source_breakdown[source_name] = 0
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("COLLECTION COMPLETE - FINAL SUMMARY")
    print("="*70)
    
    print("\n📊 Breakdown by Source:")
    for source, count in source_breakdown.items():
        print(f"   {source:25s}: {count:4d} scenarios")
    
    print(f"\n{'='*70}")
    print(f"TOTAL UNIQUE SCENARIOS: {len(all_scenarios)}")
    print(f"{'='*70}")
    
    if len(all_scenarios) >= 600:
        print(f"✅ SUCCESS: Target of 600+ scenarios achieved!")
    else:
        print(f"⚠️  WARNING: Only {len(all_scenarios)} scenarios collected (target: 600+)")
    
    # Save final combined dataset
    try:
        import json
        import os
        
        output_file = os.path.join(data_dir, "web_scenarios_enhanced.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_scenarios': len(all_scenarios),
                'source_breakdown': source_breakdown,
                'scenarios': all_scenarios
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n⚠️  Error saving final dataset: {e}")
    
    return {
        'total': len(all_scenarios),
        'breakdown': source_breakdown,
        'scenarios': all_scenarios
    }


# Example usage
if __name__ == "__main__":
    results = run_enhanced_collection(
        data_dir="../data",
        existing_scenarios_file="../data/fast_scenarios.json",
        use_selenium=False,
        ollama_model="llama3.2",
        verify_ssl=False  # NEW: Disable SSL verification for development
    )
    
    print(f"\n✅ Collection complete: {results['total']} scenarios")
    print(f"   Target achieved: {results['total'] >= 600}")