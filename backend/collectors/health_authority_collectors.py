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
        logging.FileHandler("health_authority_collectors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CDCEmergencyCollector(WebData):
    
    BASE_URL = "https://www.cdc.gov"
    
    DISCOVERY_URLS = [
        "/disasters/index.html",
        "/cpr/index.htm",
        "/disasters/injury/facts.html",
        "/disasters/extremeheat/warning.html"
    ]
    
    def get_source_name(self) -> str:
        return "CDC Emergency"
    
    def collect(self, max_pages: int = 100) -> List[Dict]:
        logger.info(f" {self.get_source_name().upper()}")
        
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
                        if any(term in href.lower() for term in ['disaster', 'emergency', 'injury', 'cpr', 'first']):
                            full_url = href if href.startswith('http') else self.BASE_URL + href
                            all_links.add(full_url)
                self._polite_wait(1.0)
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info(f"Found {len(all_links)} CDC pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                logger.debug(f" → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=['article', '.content', 'main', '[role="main"]']
                    )
                    
                    if len(content) > 300:
                        extracted = self._extract_with_ollama(content, f"CDC: {url.split('/')[-1]}")
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
        
        logger.info(f"CDC Emergency: {len(scenarios)} scenarios")
        logger.info(f"{duplicates_filtered} duplicates")
        
        self.save_checkpoint(scenarios, "cdc_scenarios.json")
        return scenarios


class NHSCollector(WebData):
    
    BASE_URL = "https://www.nhs.uk"
    
    NHS_FIRST_AID_PAGES = [
        "/conditions/accidents-and-first-aid/",
        "/conditions/first-aid/"
    ]
    
    def get_source_name(self) -> str:
        return "NHS UK"
    
    def collect(self, max_pages: int = 120) -> List[Dict]:
        logger.info(f"{self.get_source_name().upper()}")
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
        for path in self.NHS_FIRST_AID_PAGES:
            try:
                soup = self._fetch_page(self.BASE_URL + path)
                if soup:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(t in href.lower() for t in ['first-aid', 'accident', 'injury', 'bleeding']):
                            all_links.add(href if href.startswith('http') else self.BASE_URL + href)
                self._polite_wait(1.0)
            except Exception as e:
                logger.error(f"Error: {e}")

        logger.info(f"Found {len(all_links)} NHS pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                logger.debug(f" → Page {idx}/{min(len(all_links), max_pages)}")
                soup = self._fetch_page(url)
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=['article', '.nhsuk-main-wrapper', 'main']
                    )
                    
                    if len(content) > 300:
                        extracted = self._extract_with_ollama(content, f"NHS UK: {url.split('/')[-1]}")
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
                
        logger.info(f"NHS UK: {len(scenarios)} scenarios")
        logger.info(f"{duplicates_filtered} duplicates")
        
        self.save_checkpoint(scenarios, "nhs_scenarios.json")
        return scenarios


class StJohnCollector(WebData):
    
    BASE_URL = "https://www.sja.org.uk"
    ADVICE_BASE = f"{BASE_URL}/get-advice/first-aid-advice/"
    
    def get_source_name(self) -> str:
        return "St John Ambulance"
    
    def collect(self, max_pages: int = 100) -> List[Dict]:
       
        logger.info(f"{self.get_source_name().upper()}")
        scenarios = []
        duplicates_filtered = 0
        
        soup = self._fetch_page(self.ADVICE_BASE)
        if not soup:
            logger.error("Failed to fetch St John Ambulance base page")
            return []
        
        advice_links = {
            (l['href'] if l['href'].startswith('http') else self.BASE_URL + l['href'])
            for l in soup.find_all('a', href=True) 
            if '/get-advice/' in l['href']
        }

        logger.info(f"Found {len(advice_links)} St John advice pages")
        
        for idx, url in enumerate(list(advice_links)[:max_pages], 1):
            try:
                logger.debug(f" → Page {idx}/{min(len(advice_links), max_pages)}")
                soup = self._fetch_page(url)
                if soup:
                    content = self._extract_text(
                        soup,
                        selectors=['article', '.advice-content', 'main']
                    )
                    
                    if len(content) > 300:
                        extracted = self._extract_with_ollama(content, f"St John Ambulance: {url.split('/')[-1]}")
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
                
        logger.info(f"St John Ambulance: {len(scenarios)} scenarios")
        logger.info(f"{duplicates_filtered} duplicates")
        
        self.save_checkpoint(scenarios, "stjohn_scenarios.json")
        return scenarios


class WebMDCollector(WebData):
    
    BASE_URL = "https://www.webmd.com"
    
    FIRST_AID_SECTIONS = [
        "/first-aid/default.htm",
        "/first-aid/emergencies-injuries-directory"
    ]
    
    def get_source_name(self) -> str:
        return "WebMD"
    
    def collect(self, max_pages: int = 110) -> List[Dict]:
        logger.info("="*60)
        logger.info(f"{self.get_source_name().upper()}")
        logger.info("="*60)
        
        scenarios = []
        all_links = set()
        duplicates_filtered = 0
        
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
                logger.error(f"Error: {e}")
        
        logger.info(f"Found {len(all_links)} WebMD pages")
        
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                logger.debug(f" → Page {idx}/{min(len(all_links), max_pages)}")
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
                        extracted = self._extract_with_ollama(
                            content,
                            f"WebMD: {url.split('/')[-1]}"
                        )
                        
                        unique_scenarios = [s for s in extracted if not self._is_duplicate(s)]
                        duplicates_filtered += len(extracted) - len(unique_scenarios)
                        
                        for scenario in unique_scenarios:
                            scenarios.append(scenario)
                            self._add_to_existing(scenario)
                        
                        if len(unique_scenarios) > 0:
                            logger.info(f"   Extracted {len(unique_scenarios)} unique scenarios")
                
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx} pages | {len(scenarios)} scenarios")
                
                self._polite_wait(2.0)
                
            except Exception as e:
                logger.debug(f"Error on page {idx}: {e}")
                continue
        
        logger.info(f"WebMD: {len(scenarios)} scenarios")
        logger.info(f"{duplicates_filtered} duplicates")
        
        self.save_checkpoint(scenarios, "webmd_scenarios.json")
        return scenarios