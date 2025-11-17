"""
Red Cross PDF Collector - Using Ollama (Local LLM)
"""

import os
import pdfplumber
from typing import List, Dict
from .base_collector import BaseCollector


class RedCrossCollector(BaseCollector):
    """Collect scenarios from Red Cross First Aid Manual PDF"""
    
    def __init__(self, data_dir: str = None, ollama_model: str = "llama3.2"):
        # Initialize base collector with Ollama
        super().__init__(
            data_dir=data_dir if data_dir else "../data",
            ollama_model=ollama_model
        )
        # Override data_dir if provided
        if data_dir:
            self.data_dir = data_dir
        
    def get_source_name(self) -> str:
        return "Red Cross"
    
    def collect(self, pdf_path: str = None) -> List[Dict]:
        """Extract scenarios from Red Cross PDF"""
        
        if pdf_path is None:
            if self.data_dir:
                pdf_path = os.path.join(self.data_dir, "Comprehensive_Guide_for_FirstAidCPR_en.pdf")
            else:
                pdf_path = r"C:\Users\HP\first_aid_assistant\backend\data\Comprehensive_Guide_for_FirstAidCPR_en.pdf"
        
        print("\n" + "="*60)
        print("COLLECTING: RED CROSS PDF")
        print("="*60)
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            return []

        scenarios = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"📄 Processing {total_pages} pages...")
                print(f"🎯 Target: ~600 scenarios (3 per page minimum)")
                print(f"🤖 Using Ollama model: {self.ollama_model}")
                
                # Process pages individually for better extraction
                # This ensures we get 3+ scenarios per page
                
                for page_num in range(total_pages):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    
                    if not page_text or len(page_text) < 200:
                        continue
                    
                    # Extract scenarios from this page
                    page_scenarios = self._gpt_extract_scenarios(
                        page_text,
                        f"Red Cross Manual, Page {page_num + 1}",
                        "authoritative"
                    )
                    
                    # If Ollama didn't extract enough, use rule-based as backup
                    if len(page_scenarios) < 2:
                        backup_scenarios = self._extract_rule_based(
                            page_text,
                            f"Red Cross Manual, Page {page_num + 1}"
                        )
                        page_scenarios.extend(backup_scenarios)
                    
                    scenarios.extend(page_scenarios)
                    
                    # Progress update every 10 pages
                    if (page_num + 1) % 10 == 0:
                        avg_per_page = len(scenarios) / (page_num + 1)
                        print(f"  ✓ Page {page_num + 1}/{total_pages} | Scenarios: {len(scenarios)} | Avg: {avg_per_page:.1f}/page")
                    
                    # Save checkpoint every 50 pages
                    if (page_num + 1) % 50 == 0:
                        self.save_checkpoint(scenarios, f"redcross_checkpoint_{page_num + 1}.json")
        
        except Exception as e:
            print(f"❌ Error processing Red Cross PDF: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Red Cross Total: {len(scenarios)} scenarios")
        print(f"   Average: {len(scenarios)/total_pages:.1f} scenarios per page")
        
        # Save final checkpoint
        self.save_checkpoint(scenarios, "redcross_scenarios.json")
        
        return scenarios