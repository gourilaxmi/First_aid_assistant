import time
import json
import requests
import urllib3
import logging
import sys
from bs4 import BeautifulSoup
from typing import List, Dict, Set
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib3.exceptions import InsecureRequestWarning
from collectors.base_collector import BaseCollector

urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("web_scraper_backend.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WebData(BaseCollector):
   

    def __init__(
        self,
        data_dir: str = "../data",
        use_selenium: bool = False,
        ollama_model: str = "llama3.2",
        existing_scenarios_file: str = None,
        verify_ssl: bool = False
    ):
        super().__init__(data_dir, ollama_model)

        self.use_selenium = use_selenium
        self.verify_ssl = verify_ssl
        self.existing_scenarios: Set[str] = set()
        self.driver = None

        if existing_scenarios_file:
            self._load_existing_scenarios(existing_scenarios_file)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.session.verify = verify_ssl

        if self.use_selenium:
            self._setup_selenium()

    def _load_existing_scenarios(self, filepath: str):
        try:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"Scenario file not found: {filepath}")
                return

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                scenarios = data if isinstance(data, list) else data.get('scenarios', [])

                for scenario in scenarios:
                    sig = self._normalize_scenario_signature(scenario)
                    self.existing_scenarios.add(sig)

            logger.info(f"Deduplication engine active: {len(self.existing_scenarios)} signatures loaded.")
        except Exception as e:
            logger.error(f"Failed to load existing scenarios: {e}")

    def _normalize_scenario_signature(self, scenario: Dict) -> str:
        emergency = scenario.get('emergency_type', scenario.get('title', '')).lower().strip()
        desc = scenario.get('description', scenario.get('additional_info', ''))[:100].lower().strip()
        return f"{emergency}:{desc}"

    def _is_duplicate(self, scenario: Dict) -> bool:
        return self._normalize_scenario_signature(scenario) in self.existing_scenarios

    def _add_to_existing(self, scenario: Dict):
        self.existing_scenarios.add(self._normalize_scenario_signature(scenario))

    def _setup_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully.")
        except Exception as e:
            logger.error(f"Selenium setup failed: {e}. Falling back to Requests.")
            self.use_selenium = False

    def _fetch_page(self, url: str, timeout: int = 15, wait_for: str = None) -> BeautifulSoup:
        try:
            if self.use_selenium and self.driver:
                self.driver.get(url)
                if wait_for:
                    WebDriverWait(self.driver, timeout).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                    )
                else:
                    time.sleep(2) # Allow JS to execute
                return BeautifulSoup(self.driver.page_source, 'html.parser')

            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Fetch failure at {url}: {e}")
            return None

    def _extract_text(self, soup: BeautifulSoup, selectors: List[str] = None) -> str:
        if not soup:
            return ""

        unwanted = ["script", "style", "nav", "footer", "header", "aside", "form"]
        
        if selectors:
            text_parts = []
            for selector in selectors:
                for elem in soup.select(selector):
                    for tag in elem(unwanted):
                        tag.decompose()
                    text = elem.get_text(strip=True, separator='\n')
                    if len(text) > 50:
                        text_parts.append(text)
            return "\n\n".join(text_parts)

        for tag in soup(unwanted):
            tag.decompose()
        return soup.get_text(separator='\n')[:12000]

    def _polite_wait(self, seconds: float = 1.5):
        time.sleep(seconds)

    def __del__(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass