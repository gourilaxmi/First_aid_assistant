import os
import sys
import logging
import pdfplumber
from typing import List, Dict
from .base_collector import BaseCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("red_cross_collector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RedCrossCollector(BaseCollector):

    def __init__(self, data_dir: str = None, ollama_model: str = "llama3.2"):
       
        super().__init__(
            data_dir=data_dir if data_dir else "../data",
            ollama_model=ollama_model
        )

        if data_dir:
            self.data_dir = data_dir

    def get_source_name(self) -> str:
        return "Red Cross"

    def collect(self, pdf_path: str = None) -> List[Dict]:

        if pdf_path is None:
            if self.data_dir:
                pdf_path = os.path.join(
                    self.data_dir,
                    "Comprehensive_Guide_for_FirstAidCPR_en.pdf"
                )
            else:
                pdf_path = (
                    r"C:\Users\HP\first_aid_assistant\backend\data"
                    r"\Comprehensive_Guide_for_FirstAidCPR_en.pdf"
                )

        logger.info(" RED CROSS PDF")

        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return []

        scenarios = []
        total_pages = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages from Red Cross PDF")
                logger.info(f"Using Ollama model: {self.ollama_model}")

                for page_num in range(total_pages):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()

                    if not page_text or len(page_text) < 200:
                        logger.debug(f"Page {page_num + 1}: Skipping ")
                        continue

                    page_scenarios = self._extract_with_ollama(
                        page_text,
                        f"Red Cross Manual, Page {page_num + 1}"
                    )

                    scenarios.extend(page_scenarios)

                    if (page_num + 1) % 10 == 0:
                        avg_per_page = len(scenarios) / (page_num + 1)
                        logger.info(
                            f"Progress: Page {page_num + 1}/{total_pages} | "
                            f"Scenarios: {len(scenarios)} | "
                            f"Avg: {avg_per_page:.1f}/page"
                        )

                    if (page_num + 1) % 50 == 0:
                        self.save_checkpoint(
                            scenarios,
                            f"redcross_checkpoint_{page_num + 1}.json"
                        )

        except Exception as e:
            logger.error(f"Error processing Red Cross PDF: {e}", exc_info=True)

        if total_pages > 0:
            logger.info(f"Red Cross Total: {len(scenarios)} scenarios")
        else:
            logger.warning("No pages processed from Red Cross PDF")

        self.save_checkpoint(scenarios, "redcross_scenarios.json")

        return scenarios