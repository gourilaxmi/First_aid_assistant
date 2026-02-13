import sys
import json
import time
import logging
import requests
import urllib3
from bs4 import BeautifulSoup
from typing import List, Dict
from pathlib import Path

from .base_collector import BaseCollector

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("new_sources_collector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NewSourcesCollector(BaseCollector):

    def __init__(self, data_dir: str = "./data", ollama_model: str = "llama3.2"):
        super().__init__(data_dir=data_dir, ollama_model=ollama_model)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.session.verify = False

    def get_source_name(self) -> str:
        return "New Sources Collector"

    def _fetch_page_safe(self, url: str, timeout: int = 15) -> BeautifulSoup:
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)[:80]}")
            return None

    def _extract_text_from_soup(self, soup: BeautifulSoup, selectors: List[str]) -> str:
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
            except Exception as e:
                logger.debug(f"Error extracting with selector {selector}: {e}")
                pass
        return "\n\n".join(text_parts)

    def discover_and_collect(
        self, 
        base_url: str, 
        discovery_pages: List[str], 
        keywords: List[str], 
        source_name: str, 
        max_pages: int = 50
    ) -> List[Dict]:
        
        logger.info(f" {source_name.upper()}")
        
        all_links = set()

        logger.info(f"{source_name} ")
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
            except Exception as e:
                logger.error(f"Discovery error for {page}: {e}")

        logger.info(f"Found {len(all_links)} candidate pages")

        scenarios = []
        for idx, url in enumerate(list(all_links)[:max_pages], 1):
            try:
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{min(len(all_links), max_pages)} | Scenarios: {len(scenarios)}")

                soup = self._fetch_page_safe(url)
                if soup:
                    content = self._extract_text_from_soup(
                        soup, 
                        ['article', 'main', '.content', '[role="main"]', '.page-content']
                    )
                    if len(content) > 300:
                        extracted = self._extract_with_ollama(content, source_name)
                        scenarios.extend(extracted)

                time.sleep(1.5)
            except Exception as e:
                logger.debug(f"Error processing {url}: {e}")
                continue

        logger.info(f"{source_name} Complete: {len(scenarios)} scenarios collected")
        return scenarios

    def _american_heart_association(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://cpr.heart.org",
            discovery_pages=[
                "/en/resources/cpr-facts-and-stats", 
                "/en/cpr-courses-and-kits/hands-only-cpr", 
                "/en/emergency-cardiovascular-care"
            ],
            keywords=['cpr', 'emergency', 'cardiac', 'heart', 'stroke'],
            source_name="American Heart Association"
        )

    def _who_emergency(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://www.who.int",
            discovery_pages=["/health-topics/emergency-care", "/health-topics/injuries"],
            keywords=['emergency', 'injury', 'trauma', 'first-aid'],
            source_name="WHO Emergency Care"
        )

    def _medlineplus(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://medlineplus.gov",
            discovery_pages=["/firstaid.html", "/emergencies.html"],
            keywords=['first', 'emergency', 'injury', 'ency'],
            source_name="MedlinePlus NIH",
            max_pages=80
        )

    def _redcross_online(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://www.redcross.org",
            discovery_pages=[
                "/get-help/how-to-prepare-for-emergencies", 
                "/take-a-class/first-aid"
            ],
            keywords=['first-aid', 'emergency', 'prepare'],
            source_name="Red Cross Online"
        )

    def _poison_control(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://www.poison.org",
            discovery_pages=["/articles"],
            keywords=['poison', 'articles', 'swallow', 'toxic'],
            source_name="Poison Control"
        )

    def _kidshealth(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://kidshealth.org",
            discovery_pages=[
                "/en/parents/firstaid-safe.html", 
                "/en/teens/safety.html"
            ],
            keywords=['first', 'safety', 'emergency'],
            source_name="KidsHealth"
        )

    def _familydoctor(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://familydoctor.org",
            discovery_pages=["/prevention-wellness/staying-healthy/first-aid/"],
            keywords=['first-aid', 'emergency', 'injury'],
            source_name="FamilyDoctor"
        )

    def _emergencycarefor_you(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://www.emergencycareforyou.org",
            discovery_pages=["/emergency-101/"],
            keywords=['emergency', 'when-to-go', 'health'],
            source_name="EmergencyCareForYou"
        )

    def _aap_publications(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://www.healthychildren.org",
            discovery_pages=["/English/health-issues/injuries-emergencies"],
            keywords=['emergency', 'injury', 'safety'],
            source_name="AAP HealthyChildren"
        )

    def _johns_hopkins_medicine(self) -> List[Dict]:
        return self.discover_and_collect(
            base_url="https://www.hopkinsmedicine.org",
            discovery_pages=["/health/treatment-tests-and-therapies"],
            keywords=['emergency', 'first-aid', 'treatment'],
            source_name="Johns Hopkins Medicine"
        )

    def collect_all_new_sources(self) -> Dict[str, List[Dict]]:
        
        sources = [
            ("American Heart Association", self._american_heart_association),
            ("WHO Emergency Care", self._who_emergency),
            ("MedlinePlus NIH", self._medlineplus),
            ("Red Cross Online", self._redcross_online),
            ("Poison Control", self._poison_control),
            ("KidsHealth", self._kidshealth),
            ("FamilyDoctor", self._familydoctor),
            ("EmergencyCareForYou", self._emergencycarefor_you),
            ("AAP HealthyChildren", self._aap_publications),
            ("Johns Hopkins Medicine", self._johns_hopkins_medicine),
        ]

        all_results = {}
        for idx, (name, collect_func) in enumerate(sources, 1):
            logger.info(f"\n[{idx}/{len(sources)}] Starting: {name}")
            try:
                scenarios = collect_func()
                all_results[name] = scenarios
                
                checkpoint_name = f"checkpoint_{name.lower().replace(' ', '_')}.json"
                self.save_checkpoint(scenarios, checkpoint_name)
            except Exception as e:
                logger.error(f"Failed {name}: {e}", exc_info=True)
                all_results[name] = []

        total_scenarios = sum(len(s) for s in all_results.values())
        logger.info(f"Total scenarios collected: {total_scenarios}")
        logger.info("="*60)

        for source, scenarios in sorted(all_results.items(), key=lambda x: len(x[1]), reverse=True):
            logger.info(f" {source:30s}: {len(scenarios):4d} scenarios")

        return all_results

    def collect(self) -> List[Dict]:
        results = self.collect_all_new_sources()
        all_scenarios = []
        for scenarios in results.values():
            all_scenarios.extend(scenarios)
        return all_scenarios