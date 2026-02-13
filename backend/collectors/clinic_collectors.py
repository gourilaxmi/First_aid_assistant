import sys
import logging
import time
import urllib3
from typing import List, Dict
from urllib3.exceptions import InsecureRequestWarning

from .base_web_collector import WebData

urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("clinic_collectors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MayoClinicCollector(WebData):
    
    BASE_URL = "https://www.mayoclinic.org"
    
    FIRST_AID_URLS = [
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
        logger.info("="*60)
        logger.info(f"COLLECTING: {self.get_source_name().upper()}")
        logger.info("="*60)
        
        scenarios = []
        processed = 0
        duplicates_filtered = 0
        
        logger.info(f"Processing {len(self.FIRST_AID_URLS)} Mayo Clinic pages...")
        
        for url_path in self.FIRST_AID_URLS[:max_pages]:
            try:
                full_url = self.BASE_URL + url_path
                topic = url_path.split('/')[-2].replace('first-aid-', '').replace('-', ' ').title()
                logger.info(f" → {topic}")
                
                soup = self._fetch_page(full_url)
                if soup:
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
                        extracted = self._extract_with_ollama(content, f"Mayo Clinic: {topic}")
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        
                        for scenario in unique_scenarios:
                            scenarios.append(scenario)
                            self._add_to_existing(scenario)
                        
                        processed += 1
                        
                        if len(unique_scenarios) > 0:
                            logger.info(f"   Extracted {len(unique_scenarios)} unique scenarios")
                        else:
                            logger.debug("   No unique scenarios extracted")
                            
                self._polite_wait(1.5)
            except Exception as e:
                logger.error(f" Error processing {url_path}: {e}")
                continue
                
        logger.info("="*60)
        logger.info(f"Mayo Clinic: {len(scenarios)} scenarios from {processed} pages")
        logger.info(f"Filtered {duplicates_filtered} duplicates")
        logger.info("="*60)
        
        self.save_checkpoint(scenarios, "mayo_scenarios.json")
        return scenarios


class ClevelandClinicCollector(WebData):
    
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
        logger.info("="*60)
        logger.info(f"COLLECTING: {self.get_source_name().upper()}")
        logger.info("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        for base_path in self.DISCOVERY_URLS:
            try:
                soup = self._fetch_page(self.BASE_URL + base_path)
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        terms = ['first-aid', 'emergency', 'treatment', 'injury', 'accident', 
                                 'trauma', 'bleeding', 'burn', 'fracture', 'poisoning']
                        if any(term in href.lower() for term in terms):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                logger.error(f"Error: {e}")
                
        logger.info(f"Found {len(all_links)} Cleveland Clinic pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                logger.debug(f" → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=['article', '.article-body', '.content-body', 'main', '[class*="content"]']
                    )
                    
                    if len(content) > 300:
                        extracted = self._extract_with_ollama(content, f"Cleveland Clinic: {url.split('/')[-1]}")
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        
                        for scenario in unique_scenarios:
                            scenarios.append(scenario)
                            self._add_to_existing(scenario)
                        
                        if len(unique_scenarios) > 0:
                            logger.info(f"   Extracted {len(unique_scenarios)} unique scenarios")
                            
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx} pages | {len(scenarios)} scenarios")
                self._polite_wait(1.5)
            except Exception as e:
                logger.debug(f"Error on page {idx}: {e}")
                continue
                
        logger.info(f"Cleveland Clinic: {len(scenarios)} scenarios")
        logger.info(f"{duplicates_filtered} duplicates")
        
        self.save_checkpoint(scenarios, "cleveland_scenarios.json")
        return scenarios


class HealthlineCollector(WebData):
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
        logger.info("="*60)
        logger.info(f"COLLECTING: {self.get_source_name().upper()}")
        logger.info("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        for base_path in self.DISCOVERY_URLS:
            try:
                soup = self._fetch_page(self.BASE_URL + base_path)
                if soup:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        terms = ['first-aid', 'emergency', 'treatment', 'injury', 'how-to-treat',
                                 'bleeding', 'burn', 'fracture', 'bite', 'sting']
                        if any(term in href.lower() for term in terms):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            if 'healthline.com' in full_url:
                                all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                logger.error(f"Discovery error: {e}")
                
        logger.info(f"Found {len(all_links)} Healthline pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                logger.debug(f" → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=['article', '[class*="article-body"]', '.content-wrapper', 'main', '[id*="content"]']
                    )
                    
                    if len(content) > 300:
                        extracted = self._extract_with_ollama(content, f"Healthline: {url.split('/')[-1]}")
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        
                        for scenario in unique_scenarios:
                            scenarios.append(scenario)
                            self._add_to_existing(scenario)
                        
                        if len(unique_scenarios) > 0:
                            logger.info(f"   Extracted {len(unique_scenarios)} unique scenarios")
                            
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx} pages | {len(scenarios)} scenarios")
                self._polite_wait(1.5)
            except Exception as e:
                logger.debug(f"Error on page {idx}: {e}")
                continue
                
        logger.info(f"Healthline: {len(scenarios)} scenarios")
        logger.info(f" {duplicates_filtered} duplicates")
        self.save_checkpoint(scenarios, "healthline_scenarios.json")
        return scenarios