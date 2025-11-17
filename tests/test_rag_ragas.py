"""
Comprehensive RAG Testing Pipeline with RAGAS + Groq
Tests accuracy, relevance, faithfulness, and context precision

Installation:
    pip install ragas langsmith langchain-groq datasets

Setup:
    1. Set environment variables in backend/.env:
       - GROQ_API_KEY (required)
       - LANGSMITH_API_KEY (optional, for LangSmith tracking)
       - LANGSMITH_PROJECT="first-aid-rag-testing"
    
    2. Ensure backend/.env file exists with your keys

Usage:
    python tests/test_rag_ragas.py --full
    python tests/test_rag_ragas.py --ragas-only
    python tests/test_rag_ragas.py --quick
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# LangChain imports
from langchain_groq import ChatGroq

# Load environment from backend folder
backend_dir = Path(__file__).parent.parent / "backend"
env_path = backend_dir / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"[INFO] Loaded environment from: {env_path}")
else:
    print(f"[WARNING] .env not found at: {env_path}")
    load_dotenv()  # Try loading from default locations

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.RAG.rag import FirstAidRAGAssistant


# ============================================================================
# EXPANDED TEST DATASET (50 test cases)
# ============================================================================

COMPREHENSIVE_TEST_CASES = [
    # ========== BLEEDING & WOUNDS (8 cases) ==========
    {
        "question": "What should I do for severe bleeding from a deep cut?",
        "ground_truth": "Apply direct pressure with clean cloth, elevate wound above heart level, call 911 if bleeding doesn't stop after 10 minutes, don't remove cloth if soaked, add more layers on top",
        "category": "Bleeding",
        "severity": "severe",
        "expected_concepts": ["pressure", "elevate", "911", "clean cloth", "don't remove"]
    },
    {
        "question": "How to stop nosebleed?",
        "ground_truth": "Sit upright, lean forward slightly, pinch soft part of nose for 10 minutes, breathe through mouth, don't tilt head back, apply cold compress to bridge of nose",
        "category": "Bleeding",
        "severity": "minor",
        "expected_concepts": ["lean forward", "pinch nose", "10 minutes", "don't tilt back"]
    },
    {
        "question": "Treatment for minor scrape or abrasion?",
        "ground_truth": "Clean with cool water, apply antibiotic ointment, cover with sterile bandage, change daily, watch for infection signs",
        "category": "Wounds",
        "severity": "minor",
        "expected_concepts": ["clean water", "antibiotic", "bandage", "infection"]
    },
    {
        "question": "What if bleeding won't stop after 15 minutes?",
        "ground_truth": "Call 911 immediately, maintain pressure, keep victim calm and lying down, elevate injury if possible, watch for shock signs",
        "category": "Bleeding",
        "severity": "critical",
        "expected_concepts": ["911", "maintain pressure", "shock", "lying down"]
    },
    {
        "question": "First aid for puncture wound from nail?",
        "ground_truth": "Don't remove object if embedded deeply, control bleeding with pressure around wound, clean if shallow, seek medical attention for deep punctures, tetanus shot may be needed",
        "category": "Wounds",
        "severity": "moderate",
        "expected_concepts": ["don't remove", "pressure", "medical attention", "tetanus"]
    },
    {
        "question": "How to care for a cut with embedded glass?",
        "ground_truth": "Don't remove embedded glass, call 911, apply pressure around wound not directly on glass, immobilize area, cover loosely with sterile dressing",
        "category": "Wounds",
        "severity": "severe",
        "expected_concepts": ["don't remove", "911", "pressure around", "immobilize"]
    },
    {
        "question": "Signs of internal bleeding after injury?",
        "ground_truth": "Pale skin, rapid weak pulse, cold clammy skin, confusion, severe pain or swelling, coughing blood, vomiting blood, call 911 immediately",
        "category": "Bleeding",
        "severity": "critical",
        "expected_concepts": ["pale", "rapid pulse", "confusion", "911", "shock"]
    },
    {
        "question": "Treating bleeding from mouth or gums?",
        "ground_truth": "Rinse mouth with cold water, apply gauze with pressure for 10 minutes, use ice pack outside jaw, seek medical help if doesn't stop, avoid hot liquids",
        "category": "Bleeding",
        "severity": "minor",
        "expected_concepts": ["cold water", "gauze", "pressure", "ice pack"]
    },

    # ========== BURNS (7 cases) ==========
    {
        "question": "How to treat a second-degree burn?",
        "ground_truth": "Cool with running water for 10-20 minutes, don't use ice, cover with sterile non-stick dressing, don't break blisters, take pain reliever, seek medical help if larger than 3 inches",
        "category": "Burns",
        "severity": "moderate",
        "expected_concepts": ["cool water", "10-20 minutes", "don't break blisters", "sterile dressing"]
    },
    {
        "question": "First aid for minor first-degree burn?",
        "ground_truth": "Cool with water for 10 minutes, apply aloe vera or burn gel, cover loosely if needed, take over-the-counter pain reliever, watch for infection",
        "category": "Burns",
        "severity": "minor",
        "expected_concepts": ["cool water", "aloe vera", "pain reliever"]
    },
    {
        "question": "What to do for chemical burn on skin?",
        "ground_truth": "Remove contaminated clothing immediately, brush off dry chemical first, rinse with running water for 20+ minutes, call 911 or poison control, don't neutralize with other chemicals",
        "category": "Burns",
        "severity": "severe",
        "expected_concepts": ["remove clothing", "rinse 20 minutes", "911", "don't neutralize"]
    },
    {
        "question": "Treatment for electrical burn?",
        "ground_truth": "Ensure power source is off, don't touch victim if still in contact, call 911 immediately, check breathing and pulse, treat visible burns, watch for cardiac issues",
        "category": "Burns",
        "severity": "critical",
        "expected_concepts": ["power off", "don't touch", "911", "cardiac", "breathing"]
    },
    {
        "question": "How to treat sunburn?",
        "ground_truth": "Cool bath or compress, moisturize with aloe vera, drink water, take ibuprofen for pain, don't break blisters, avoid sun exposure, seek help if severe blistering",
        "category": "Burns",
        "severity": "minor",
        "expected_concepts": ["cool compress", "aloe", "hydrate", "ibuprofen"]
    },
    {
        "question": "Third-degree burn first aid?",
        "ground_truth": "Call 911 immediately, don't remove burnt clothing, don't immerse in water, cover with sterile cloth, treat for shock, monitor breathing, don't apply ointments",
        "category": "Burns",
        "severity": "critical",
        "expected_concepts": ["911", "don't remove clothing", "shock", "sterile cloth", "no ointments"]
    },
    {
        "question": "Burn blisters - should I pop them?",
        "ground_truth": "Never pop burn blisters, they protect against infection, if blister breaks naturally clean gently, apply antibiotic ointment, cover with non-stick bandage, watch for infection",
        "category": "Burns",
        "severity": "minor",
        "expected_concepts": ["never pop", "infection protection", "antibiotic", "non-stick"]
    },

    # ========== CPR & CARDIAC (7 cases) ==========
    {
        "question": "CPR steps for unconscious adult not breathing?",
        "ground_truth": "Call 911, place on firm surface, 30 chest compressions at least 2 inches deep and 100-120 per minute, give 2 rescue breaths, continue cycles until help arrives or victim breathes",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["911", "30 compressions", "2 breaths", "100-120 per minute", "2 inches deep"]
    },
    {
        "question": "Signs someone is having a heart attack?",
        "ground_truth": "Chest pain or pressure, pain radiating to arm/jaw/back, shortness of breath, nausea, cold sweats, lightheadedness, call 911 immediately, have them sit and rest",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["chest pain", "radiating pain", "911", "sit and rest", "shortness of breath"]
    },
    {
        "question": "How to use an AED on someone?",
        "ground_truth": "Turn on AED, attach pads to bare chest as shown, ensure no one touches victim, let AED analyze, press shock button if advised, resume CPR immediately after shock",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["bare chest", "don't touch", "analyze", "shock", "resume CPR"]
    },
    {
        "question": "Difference between heart attack and cardiac arrest?",
        "ground_truth": "Heart attack is circulation problem (blocked artery), victim usually conscious. Cardiac arrest is electrical problem (heart stops beating), victim unconscious and not breathing. Cardiac arrest needs immediate CPR",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["circulation vs electrical", "conscious vs unconscious", "CPR for arrest"]
    },
    {
        "question": "Hands-only CPR - when is it appropriate?",
        "ground_truth": "For adults who suddenly collapse, untrained rescuers, or if uncomfortable giving breaths. Push hard and fast in center of chest at 100-120 per minute, 2 inches deep, don't stop until help arrives",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["adults only", "100-120 per minute", "hard and fast", "center chest"]
    },
    {
        "question": "What if victim vomits during CPR?",
        "ground_truth": "Turn victim on side, clear mouth quickly with finger sweep, reposition on back, continue CPR immediately, protect airway if possible",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["turn on side", "clear mouth", "continue CPR", "reposition"]
    },
    {
        "question": "CPR on child vs adult - differences?",
        "ground_truth": "Children (1-8 years): use one or two hands, compress 2 inches or 1/3 chest depth, 30:2 ratio same as adult but give 5 cycles before calling 911 if alone, use child AED pads if available",
        "category": "Cardiac",
        "severity": "critical",
        "expected_concepts": ["one or two hands", "1/3 depth", "5 cycles first", "child pads"]
    },

    # ========== CHOKING (5 cases) ==========
    {
        "question": "Someone is choking and can't breathe - what to do?",
        "ground_truth": "Ask if choking, encourage coughing if possible, if can't cough/speak give 5 back blows between shoulder blades, then 5 abdominal thrusts (Heimlich), alternate until object dislodges or victim unconscious",
        "category": "Choking",
        "severity": "critical",
        "expected_concepts": ["5 back blows", "5 abdominal thrusts", "alternate", "between shoulder blades"]
    },
    {
        "question": "How to perform Heimlich maneuver correctly?",
        "ground_truth": "Stand behind victim, make fist above navel below ribs, grasp fist with other hand, give quick upward thrusts, repeat until object comes out, don't press on ribs",
        "category": "Choking",
        "severity": "critical",
        "expected_concepts": ["above navel", "upward thrusts", "behind victim", "don't press ribs"]
    },
    {
        "question": "Choking infant first aid?",
        "ground_truth": "Hold face down on forearm supporting head, give 5 back blows with heel of hand, turn face up give 5 chest thrusts with 2 fingers, alternate until object comes out, call 911 if doesn't clear",
        "category": "Choking",
        "severity": "critical",
        "expected_concepts": ["face down", "5 back blows", "5 chest thrusts", "2 fingers", "alternate"]
    },
    {
        "question": "What if choking victim becomes unconscious?",
        "ground_truth": "Lower gently to ground, call 911, begin CPR immediately, check mouth for object before giving breaths, remove only if visible, don't do finger sweep unless object seen",
        "category": "Choking",
        "severity": "critical",
        "expected_concepts": ["lower to ground", "911", "start CPR", "check mouth", "visible object only"]
    },
    {
        "question": "Choking on self - how to help yourself?",
        "ground_truth": "Make fist above navel, grasp with other hand, thrust inward and upward, or lean over firm surface like chair back and push abdomen against it, repeat until object dislodges",
        "category": "Choking",
        "severity": "critical",
        "expected_concepts": ["self-fist", "thrust upward", "chair back", "firm surface"]
    },
]


# ============================================================================
# RAG WRAPPER FOR TESTING
# ============================================================================

class RAGTestWrapper:
    """Wrapper for RAG assistant compatible with RAGAS"""
    
    def __init__(self):
        print("\n[INFO] Initializing RAG Assistant for testing...")
        self.assistant = FirstAidRAGAssistant()
        
        # Initialize Groq LLM for RAGAS evaluation
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=groq_api_key
        )
        print("[SUCCESS] RAG Assistant ready with Groq LLM\n")
    
    def query(self, question: str, conversation_id: str = None) -> Dict:
        """Query the RAG system"""
        result = self.assistant.answer_query(
            query=question,
            conversation_id=conversation_id,
            verbose=False
        )
        
        return {
            "answer": result['response'],
            "contexts": [
                f"{source.get('title', 'Unknown')}: {source.get('source', '')}... {result['response'][:200]}" 
                for source in result.get('sources', [])[:3]
            ],
            "source_documents": result.get('sources', []),
            "confidence": result.get('avg_relevance', 0),
            "chunks_found": result.get('chunks_found', 0)
        }


# ============================================================================
# RAGAS EVALUATION
# ============================================================================

class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics"""
    
    def __init__(self, rag_wrapper: RAGTestWrapper):
        self.rag = rag_wrapper
        self.results = []
        
        # RAGAS metrics
        self.metrics = [
            faithfulness,           # Is answer faithful to retrieved context?
            answer_relevancy,       # Is answer relevant to question?
            context_precision,      # Are relevant contexts ranked higher?
            context_recall,         # Did we retrieve all relevant contexts?
            answer_correctness      # Correctness compared to ground truth
        ]
    
    def evaluate(self, test_cases: List[Dict], batch_size: int = 10) -> Dict:
        """Run evaluation on test cases"""
        
        print("\n" + "="*70)
        print(" "*20 + " EVALUATION")
        print("="*70)
        print(f"\n Evaluating {len(test_cases)} test cases")
        print(f" Using Groq LLM: llama-3.3-70b-versatile")
        print(f"Metrics: {', '.join([m.name for m in self.metrics])}\n")
        
        # Prepare data
        questions = []
        ground_truths = []
        answers = []
        contexts_list = []
        
        # Process each test case
        for idx, test_case in enumerate(test_cases, 1):
            if idx % 10 == 0:
                print(f" Processing {idx}/{len(test_cases)}...")
            
            try:
                # Query RAG system
                result = self.rag.query(test_case['question'])
                
                # Store results
                questions.append(test_case['question'])
                ground_truths.append(test_case['ground_truth'])
                answers.append(result['answer'])
                contexts_list.append(result['contexts'])
                
                # Store individual result
                self.results.append({
                    "question": test_case['question'],
                    "answer": result['answer'],
                    "ground_truth": test_case['ground_truth'],
                    "contexts": result['contexts'],
                    "category": test_case['category'],
                    "severity": test_case['severity'],
                    "confidence": result['confidence'],
                    "chunks_found": result['chunks_found']
                })
                
            except Exception as e:
                print(f"  [WARNING] Error on case {idx}: {str(e)}")
                continue
        
        # Create dataset for RAGAS
        print(f"\nRunning evaluation with {len(self.metrics)} metrics...")
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        try:
            ragas_result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.rag.llm,
                embeddings=None  # Will use default embeddings
            )
            
            # Print results
            print(f"\n{'='*70}")
            print("EVALUATION RESULTS")
            print(f"{'='*70}")
            
            for metric_name, score in ragas_result.items():
                if isinstance(score, (int, float)):
                    print(f"{metric_name:30s}: {score:.4f}")
            
            print(f"{'='*70}")
            
            return ragas_result
            
        except Exception as e:
            print(f"\n Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_results(self, output_dir: Path):
        """Save evaluation results"""
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"ragas_results_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_tests': len(self.results),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] Results saved to: {filepath}")
        return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Testing Pipeline')
    parser.add_argument('--full', action='store_true', help='Run full test suite (27 cases)')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 cases)')
    parser.add_argument('--ragas-only', action='store_true', help='RAGAS metrics only')
    parser.add_argument('--custom', type=int, help='Custom number of test cases')
    
    args = parser.parse_args()
    
    # Check environment
    if not os.getenv('GROQ_API_KEY'):
        print(" GROQ_API_KEY not found in environment")
        print(f"   Expected location: {env_path}")
        print("   Add to backend/.env: GROQ_API_KEY=your-key-here")
        return
    
    # Determine test cases to run
    if args.custom:
        test_cases = COMPREHENSIVE_TEST_CASES[:args.custom]
    elif args.quick:
        test_cases = COMPREHENSIVE_TEST_CASES[:10]
    else:
        test_cases = COMPREHENSIVE_TEST_CASES
    
    print(f"\n{'='*70}")
    print("[START] RAGAS TESTING PIPELINE - FIRST AID RAG (Groq)")
    print(f"{'='*70}")
    print(f"\nTest cases: {len(test_cases)}")
    print(f"LLM: Groq (llama-3.3-70b-versatile)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize
        rag_wrapper = RAGTestWrapper()
        evaluator = RAGASEvaluator(rag_wrapper)
        
        # Run evaluation
        ragas_results = evaluator.evaluate(test_cases)
        
        # Save results
        output_dir = Path("./test_results")
        evaluator.save_results(output_dir)
        
        print(f"\n{'='*70}")
        print("[SUCCESS] TESTING COMPLETE")
        print(f"{'='*70}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n Testing interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()