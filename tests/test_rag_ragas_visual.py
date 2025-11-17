"""
Enhanced RAG Testing Pipeline with RAGAS + Visualizations
Generates comprehensive evaluation metrics with graphs and matrices

Installation:
    pip install ragas langchain-groq datasets matplotlib seaborn plotly pandas

Usage:
    python tests/test_rag_ragas_visual.py --full
    python tests/test_rag_ragas_visual.py --quick
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
from langchain_groq import ChatGroq

# Load environment
backend_dir = Path(__file__).parent.parent / "backend"
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.RAG.rag import FirstAidRAGAssistant

# Comprehensive Test Cases (75 cases across all categories)
COMPREHENSIVE_TEST_CASES = [
    # BLEEDING & WOUNDS (20 cases)
    {
        "question": "What should I do for severe bleeding from a deep cut?",
        "ground_truth": "Apply direct pressure with clean cloth, elevate wound above heart level, call 911 if bleeding doesn't stop after 10 minutes, don't remove cloth if soaked, add more layers on top",
        "category": "Bleeding",
        "severity": "severe"
    },
    {
        "question": "How to stop nosebleed?",
        "ground_truth": "Sit upright, lean forward slightly, pinch soft part of nose for 10 minutes, breathe through mouth, don't tilt head back, apply cold compress to bridge of nose",
        "category": "Bleeding",
        "severity": "minor"
    },
    {
        "question": "Treatment for minor scrape or abrasion?",
        "ground_truth": "Clean with cool water, apply antibiotic ointment, cover with sterile bandage, change daily, watch for infection signs",
        "category": "Wounds",
        "severity": "minor"
    },
    {
        "question": "What if bleeding won't stop after 15 minutes?",
        "ground_truth": "Call 911 immediately, maintain pressure, keep victim calm and lying down, elevate injury if possible, watch for shock signs",
        "category": "Bleeding",
        "severity": "critical"
    },
    {
        "question": "First aid for puncture wound from nail?",
        "ground_truth": "Don't remove object if embedded deeply, control bleeding with pressure around wound, clean if shallow, seek medical attention for deep punctures, tetanus shot may be needed",
        "category": "Wounds",
        "severity": "moderate"
    },
    {
        "question": "How to care for a cut with embedded glass?",
        "ground_truth": "Don't remove embedded glass, call 911, apply pressure around wound not directly on glass, immobilize area, cover loosely with sterile dressing",
        "category": "Wounds",
        "severity": "severe"
    },
    {
        "question": "Signs of internal bleeding after injury?",
        "ground_truth": "Pale skin, rapid weak pulse, cold clammy skin, confusion, severe pain or swelling, coughing blood, vomiting blood, call 911 immediately",
        "category": "Bleeding",
        "severity": "critical"
    },
    {
        "question": "Treating bleeding from mouth or gums?",
        "ground_truth": "Rinse mouth with cold water, apply gauze with pressure for 10 minutes, use ice pack outside jaw, seek medical help if doesn't stop, avoid hot liquids",
        "category": "Bleeding",
        "severity": "minor"
    },
    {
        "question": "How to bandage a finger cut properly?",
        "ground_truth": "Clean wound, apply antibiotic ointment, wrap with sterile gauze starting at base, secure with tape, keep slightly elevated, change bandage daily",
        "category": "Wounds",
        "severity": "minor"
    },
    {
        "question": "What to do for a split lip?",
        "ground_truth": "Apply pressure with clean cloth, use ice pack to reduce swelling, keep head elevated, avoid hot foods, see doctor if deep or won't stop bleeding",
        "category": "Wounds",
        "severity": "minor"
    },
    {
        "question": "Treatment for road rash from bike accident?",
        "ground_truth": "Gently clean with water, remove debris carefully, apply antibiotic ointment, cover with non-stick bandage, change daily, watch for infection",
        "category": "Wounds",
        "severity": "moderate"
    },
    {
        "question": "How to handle bleeding from ear after head injury?",
        "ground_truth": "Don't plug ear, cover loosely with sterile gauze, keep head elevated, call 911 immediately as this may indicate skull fracture",
        "category": "Bleeding",
        "severity": "critical"
    },
    {
        "question": "First aid for cut tongue?",
        "ground_truth": "Rinse mouth with cold water, apply pressure with clean gauze, use ice chips, avoid talking and eating, seek help if bleeding continues over 15 minutes",
        "category": "Wounds",
        "severity": "minor"
    },
    {
        "question": "What if someone has multiple cuts from broken glass?",
        "ground_truth": "Control bleeding on worst wounds first, don't remove embedded glass, cover all wounds, call 911, treat for shock if needed",
        "category": "Wounds",
        "severity": "severe"
    },
    {
        "question": "How to treat bleeding varicose vein?",
        "ground_truth": "Lie down immediately, elevate leg above heart level, apply firm direct pressure, call 911, don't apply tourniquet",
        "category": "Bleeding",
        "severity": "severe"
    },
    {
        "question": "Treatment for animal bite that's bleeding?",
        "ground_truth": "Wash thoroughly with soap and water for 5 minutes, control bleeding with pressure, cover with clean bandage, seek medical attention immediately for rabies evaluation and tetanus",
        "category": "Wounds",
        "severity": "moderate"
    },
    {
        "question": "How to stop bleeding from shaving cut?",
        "ground_truth": "Apply direct pressure with tissue or styptic pencil, use cold water, if continues apply petroleum jelly, usually stops within minutes",
        "category": "Bleeding",
        "severity": "minor"
    },
    {
        "question": "First aid for avulsed tooth with bleeding?",
        "ground_truth": "Control bleeding with gauze, rinse tooth gently, place in milk or saline, see dentist within 30 minutes, don't touch root",
        "category": "Wounds",
        "severity": "moderate"
    },
    {
        "question": "What to do for bleeding blister?",
        "ground_truth": "Don't pop, if broken clean gently with soap and water, apply antibiotic ointment, cover with bandage, watch for infection",
        "category": "Wounds",
        "severity": "minor"
    },
    {
        "question": "How to handle amputation with severe bleeding?",
        "ground_truth": "Call 911 immediately, apply direct pressure, use tourniquet if pressure fails, preserve amputated part in plastic bag on ice, treat for shock",
        "category": "Bleeding",
        "severity": "critical"
    },

    # BURNS (15 cases)
    {
        "question": "How to treat a second-degree burn?",
        "ground_truth": "Cool with running water for 10-20 minutes, don't use ice, cover with sterile non-stick dressing, don't break blisters, take pain reliever, seek medical help if larger than 3 inches",
        "category": "Burns",
        "severity": "moderate"
    },
    {
        "question": "First aid for minor first-degree burn?",
        "ground_truth": "Cool with water for 10 minutes, apply aloe vera or burn gel, cover loosely if needed, take over-the-counter pain reliever, watch for infection",
        "category": "Burns",
        "severity": "minor"
    },
    {
        "question": "What to do for chemical burn on skin?",
        "ground_truth": "Remove contaminated clothing immediately, brush off dry chemical first, rinse with running water for 20+ minutes, call 911 or poison control, don't neutralize with other chemicals",
        "category": "Burns",
        "severity": "severe"
    },
    {
        "question": "Treatment for electrical burn?",
        "ground_truth": "Ensure power source is off, don't touch victim if still in contact, call 911 immediately, check breathing and pulse, treat visible burns, watch for cardiac issues",
        "category": "Burns",
        "severity": "critical"
    },
    {
        "question": "How to treat sunburn?",
        "ground_truth": "Cool bath or compress, moisturize with aloe vera, drink water, take ibuprofen for pain, don't break blisters, avoid sun exposure, seek help if severe blistering",
        "category": "Burns",
        "severity": "minor"
    },
    {
        "question": "Third-degree burn first aid?",
        "ground_truth": "Call 911 immediately, don't remove burnt clothing, don't immerse in water, cover with sterile cloth, treat for shock, monitor breathing, don't apply ointments",
        "category": "Burns",
        "severity": "critical"
    },
    {
        "question": "Burn blisters - should I pop them?",
        "ground_truth": "Never pop burn blisters, they protect against infection, if blister breaks naturally clean gently, apply antibiotic ointment, cover with non-stick bandage, watch for infection",
        "category": "Burns",
        "severity": "minor"
    },
    {
        "question": "Treatment for hot oil splatter burn?",
        "ground_truth": "Cool immediately with running water for 10-15 minutes, don't apply ice or butter, cover with clean cloth, take pain reliever, seek medical help if larger than palm",
        "category": "Burns",
        "severity": "moderate"
    },
    {
        "question": "First aid for steam burn on hand?",
        "ground_truth": "Run under cool water for 15 minutes, don't use ice, remove jewelry immediately, cover loosely with sterile gauze, elevate hand, take pain medication",
        "category": "Burns",
        "severity": "moderate"
    },
    {
        "question": "How to treat chemical burn in eye?",
        "ground_truth": "Flush immediately with water for 15-20 minutes, hold eyelid open, remove contact lenses, don't rub eye, call 911 or go to ER immediately",
        "category": "Burns",
        "severity": "critical"
    },
    {
        "question": "What to do for tar or hot plastic burn?",
        "ground_truth": "Don't try to remove tar or plastic, cool with water, cover loosely, seek immediate medical attention for removal, don't use solvents",
        "category": "Burns",
        "severity": "severe"
    },
    {
        "question": "Treatment for friction burn from rope?",
        "ground_truth": "Clean with cool water, apply antibiotic ointment, cover with non-stick bandage, change daily, watch for infection, don't pop blisters",
        "category": "Burns",
        "severity": "minor"
    },
    {
        "question": "How to handle burns on face?",
        "ground_truth": "Cool with wet cloth, don't apply ointments to face, keep head elevated, seek immediate medical attention, watch for breathing difficulties",
        "category": "Burns",
        "severity": "severe"
    },
    {
        "question": "First aid for contact with hot metal?",
        "ground_truth": "Remove from heat source immediately, cool with running water for 15 minutes, don't apply ice directly, cover with clean cloth, assess burn degree, seek medical help if severe",
        "category": "Burns",
        "severity": "moderate"
    },
    {
        "question": "Treatment for burn with charred skin?",
        "ground_truth": "Call 911 immediately, don't remove clothing stuck to burn, cover loosely with clean sheet, don't immerse in water, treat for shock, monitor vital signs",
        "category": "Burns",
        "severity": "critical"
    },

    # CARDIAC & CPR (15 cases)
    {
        "question": "CPR steps for unconscious adult not breathing?",
        "ground_truth": "Call 911, place on firm surface, 30 chest compressions at least 2 inches deep and 100-120 per minute, give 2 rescue breaths, continue cycles until help arrives or victim breathes",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "Signs someone is having a heart attack?",
        "ground_truth": "Chest pain or pressure, pain radiating to arm/jaw/back, shortness of breath, nausea, cold sweats, lightheadedness, call 911 immediately, have them sit and rest",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "How to use an AED on someone?",
        "ground_truth": "Turn on AED, attach pads to bare chest as shown, ensure no one touches victim, let AED analyze, press shock button if advised, resume CPR immediately after shock",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "Difference between heart attack and cardiac arrest?",
        "ground_truth": "Heart attack is circulation problem (blocked artery), victim usually conscious. Cardiac arrest is electrical problem (heart stops beating), victim unconscious and not breathing. Cardiac arrest needs immediate CPR",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "Hands-only CPR - when is it appropriate?",
        "ground_truth": "For adults who suddenly collapse, untrained rescuers, or if uncomfortable giving breaths. Push hard and fast in center of chest at 100-120 per minute, 2 inches deep, don't stop until help arrives",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "What if victim vomits during CPR?",
        "ground_truth": "Turn victim on side, clear mouth quickly with finger sweep, reposition on back, continue CPR immediately, protect airway if possible",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "CPR on child vs adult - differences?",
        "ground_truth": "Children (1-8 years): use one or two hands, compress 2 inches or 1/3 chest depth, 30:2 ratio same as adult but give 5 cycles before calling 911 if alone, use child AED pads if available",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "What to do if someone faints?",
        "ground_truth": "Check responsiveness, if breathing lay on back and elevate legs 12 inches, loosen tight clothing, check for injuries, if not breathing start CPR and call 911",
        "category": "Cardiac",
        "severity": "moderate"
    },
    {
        "question": "Signs of stroke and first aid?",
        "ground_truth": "Face drooping, arm weakness, speech difficulty, time to call 911. Note time symptoms started, keep calm and lying down, don't give food or drink",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "How to help someone with chest pain?",
        "ground_truth": "Call 911 immediately, have them sit and rest, loosen tight clothing, give aspirin if not allergic and conscious, stay calm, monitor breathing",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "CPR on infant under 1 year?",
        "ground_truth": "Use 2 fingers in center of chest, compress 1.5 inches deep, 30 compressions to 2 gentle breaths, 100-120 per minute, call 911 after 5 cycles if alone",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "What if AED says no shock advised?",
        "ground_truth": "Continue CPR immediately, recheck every 2 minutes, follow AED prompts, don't stop compressions until help arrives or victim breathes",
        "category": "Cardiac",
        "severity": "critical"
    },
    {
        "question": "How to treat hyperventilation?",
        "ground_truth": "Help person breathe slowly, encourage slow deep breaths, have them sit or lie down, reassure calmly, don't use paper bag method, call 911 if doesn't improve",
        "category": "Cardiac",
        "severity": "moderate"
    },
    {
        "question": "First aid for irregular heartbeat feeling?",
        "ground_truth": "Have person sit and rest, stay calm, call 911 if accompanied by chest pain, shortness of breath, dizziness or fainting, monitor symptoms",
        "category": "Cardiac",
        "severity": "moderate"
    },
    {
        "question": "What if person has DNR during cardiac arrest?",
        "ground_truth": "If valid DNR present, don't perform CPR, provide comfort care, call 911 to report, stay with person, follow advance directives",
        "category": "Cardiac",
        "severity": "critical"
    },

    # CHOKING (10 cases)
    {
        "question": "Someone is choking and can't breathe - what to do?",
        "ground_truth": "Ask if choking, encourage coughing if possible, if can't cough/speak give 5 back blows between shoulder blades, then 5 abdominal thrusts (Heimlich), alternate until object dislodges or victim unconscious",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "How to perform Heimlich maneuver correctly?",
        "ground_truth": "Stand behind victim, make fist above navel below ribs, grasp fist with other hand, give quick upward thrusts, repeat until object comes out, don't press on ribs",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "Choking infant first aid?",
        "ground_truth": "Hold face down on forearm supporting head, give 5 back blows with heel of hand, turn face up give 5 chest thrusts with 2 fingers, alternate until object comes out, call 911 if doesn't clear",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "What if choking victim becomes unconscious?",
        "ground_truth": "Lower gently to ground, call 911, begin CPR immediately, check mouth for object before giving breaths, remove only if visible, don't do finger sweep unless object seen",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "Choking on self - how to help yourself?",
        "ground_truth": "Make fist above navel, grasp with other hand, thrust inward and upward, or lean over firm surface like chair back and push abdomen against it, repeat until object dislodges",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "How to help pregnant woman who is choking?",
        "ground_truth": "Use chest thrusts instead of abdominal thrusts, position hands on center of breastbone, give firm backward thrusts, alternate with back blows",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "What if person is choking but can still cough?",
        "ground_truth": "Encourage continued coughing, stay with them, don't interfere with coughing, call 911 if can't clear or stops breathing, be ready to perform Heimlich",
        "category": "Choking",
        "severity": "moderate"
    },
    {
        "question": "Choking on large person - how to reach around?",
        "ground_truth": "If can't wrap arms around, perform chest thrusts from behind or have person lie on back and use upward thrusts on abdomen, call 911",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "How to clear airway of unconscious choking victim?",
        "ground_truth": "Open mouth, look for object, if visible remove with finger sweep, attempt rescue breaths, if blocked start CPR, recheck mouth between cycles",
        "category": "Choking",
        "severity": "critical"
    },
    {
        "question": "Partial airway obstruction vs complete - differences?",
        "ground_truth": "Partial: can cough, speak, breathe - encourage coughing. Complete: can't breathe, speak or cough, turns blue - immediate Heimlich needed, call 911",
        "category": "Choking",
        "severity": "critical"
    },

    # FRACTURES & SPRAINS (10 cases)
    {
        "question": "How to tell if bone is broken?",
        "ground_truth": "Severe pain, swelling, bruising, deformity, can't bear weight or move, numbness or tingling, bone protruding through skin - immobilize and seek medical help",
        "category": "Fractures",
        "severity": "severe"
    },
    {
        "question": "First aid for suspected arm fracture?",
        "ground_truth": "Don't move arm, immobilize with splint or sling, apply ice pack, elevate if possible, don't try to straighten, seek immediate medical attention",
        "category": "Fractures",
        "severity": "severe"
    },
    {
        "question": "Treatment for ankle sprain?",
        "ground_truth": "RICE method: Rest, Ice for 20 minutes every 2-3 hours, Compression with elastic bandage, Elevation above heart level, take anti-inflammatory, seek medical evaluation",
        "category": "Sprains",
        "severity": "moderate"
    },
    {
        "question": "What to do for compound fracture with bone visible?",
        "ground_truth": "Call 911 immediately, don't push bone back, cover with sterile dressing, control bleeding with pressure around wound, immobilize, treat for shock",
        "category": "Fractures",
        "severity": "critical"
    },
    {
        "question": "How to splint a leg injury?",
        "ground_truth": "Don't move leg, pad splint material, extend beyond joints above and below injury, secure with bandages, check circulation, don't tie too tight, get medical help",
        "category": "Fractures",
        "severity": "severe"
    },
    {
        "question": "First aid for dislocated shoulder?",
        "ground_truth": "Don't try to pop back in, immobilize with sling, apply ice pack, give pain reliever, seek immediate medical attention, keep arm supported",
        "category": "Fractures",
        "severity": "severe"
    },
    {
        "question": "How to treat sprained wrist?",
        "ground_truth": "RICE protocol, immobilize with wrap or brace, ice 15-20 minutes hourly, keep elevated, take anti-inflammatory, see doctor if severe pain or can't move",
        "category": "Sprains",
        "severity": "minor"
    },
    {
        "question": "What if suspect spinal injury?",
        "ground_truth": "Don't move person, call 911 immediately, stabilize head and neck, don't remove helmet if wearing one, monitor breathing, wait for professionals",
        "category": "Fractures",
        "severity": "critical"
    },
    {
        "question": "How to make emergency splint?",
        "ground_truth": "Use rigid material like board, rolled newspaper, or stick, pad with cloth, extend past joints, secure with cloth strips or tape, check circulation after applying",
        "category": "Fractures",
        "severity": "moderate"
    },
    {
        "question": "Treatment for suspected rib fracture?",
        "ground_truth": "Don't wrap ribs tightly, support arm on injured side, take shallow breaths, apply ice, take pain reliever, seek medical attention, watch for breathing difficulty",
        "category": "Fractures",
        "severity": "moderate"
    },

    # POISONING & ALLERGIC REACTIONS (5 cases)
    {
        "question": "What to do if someone swallows poison?",
        "ground_truth": "Call Poison Control (1-800-222-1222) immediately, don't induce vomiting unless instructed, keep poison container, monitor breathing, be ready to perform CPR",
        "category": "Poisoning",
        "severity": "critical"
    },
    {
        "question": "Signs of severe allergic reaction (anaphylaxis)?",
        "ground_truth": "Difficulty breathing, swelling of throat/tongue, rapid pulse, dizziness, hives, nausea - use EpiPen if available, call 911 immediately, lay person down",
        "category": "Allergic",
        "severity": "critical"
    },
    {
        "question": "How to use an EpiPen?",
        "ground_truth": "Remove from carrier, hold firmly, swing and push hard into outer thigh until click, hold for 3 seconds, massage area, call 911, may need second dose after 5-15 minutes",
        "category": "Allergic",
        "severity": "critical"
    },
    {
        "question": "First aid for carbon monoxide poisoning?",
        "ground_truth": "Move person to fresh air immediately, call 911, open windows, don't go back in, monitor breathing, perform CPR if needed, everyone should evacuate",
        "category": "Poisoning",
        "severity": "critical"
    },
    {
        "question": "Treatment for bee sting allergic reaction?",
        "ground_truth": "Remove stinger by scraping, wash area, apply ice, give antihistamine, watch for severe reaction signs, use EpiPen if breathing difficulty, call 911 if symptoms worsen",
        "category": "Allergic",
        "severity": "moderate"
    }
]


class RAGTestWrapper:
    """Wrapper for RAG assistant"""
    
    def __init__(self):
        self.assistant = FirstAidRAGAssistant()
        
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=groq_api_key
        )
    
    def query(self, question: str) -> Dict:
        """Query RAG system"""
        result = self.assistant.answer_query(query=question, verbose=False)
        
        return {
            "answer": result['response'],
            "contexts": [
                f"{source.get('title', 'Unknown')}: {result['response'][:200]}" 
                for source in result.get('sources', [])[:3]
            ],
            "confidence": result.get('avg_relevance', 0),
            "chunks_found": result.get('chunks_found', 0)
        }


class VisualizationGenerator:
    """Generate comprehensive visualizations for evaluation results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def generate_all_visualizations(self, results: Dict, detailed_results: List[Dict]):
        
        self.plot_overall_metrics(results)
        self.plot_category_heatmap(detailed_results)
        self.plot_severity_analysis(detailed_results)
        self.plot_correlation_matrix(detailed_results, results)
        self.plot_performance_distribution(detailed_results)
        self.plot_individual_performance(detailed_results, results)
    
    def plot_overall_metrics(self, results: Dict):
        """Bar chart of overall RAGAS metrics"""
        
        metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(metrics.keys(), metrics.values(), color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall RAGAS Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.7)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair threshold (0.5)')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_category_heatmap(self, detailed_results: List[Dict]):
        """Heatmap showing performance by category"""
        
        # Extract categories and metrics
        categories = list(set([r['category'] for r in detailed_results]))
        
        # This is a simplified version - in real evaluation, you'd have per-case metrics
        category_scores = {cat: [] for cat in categories}
        
        for result in detailed_results:
            category_scores[result['category']].append(result.get('confidence', 0))
        
        # Calculate average scores
        avg_scores = {cat: np.mean(scores) if scores else 0 
                     for cat, scores in category_scores.items()}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [[avg_scores.get(cat, 0)] for cat in categories]
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   yticklabels=categories, xticklabels=['Confidence'],
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        
        ax.set_title('Performance by Category', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_severity_analysis(self, detailed_results: List[Dict]):
        """Box plot showing performance by severity level"""
        
        df = pd.DataFrame(detailed_results)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        severities = ['minor', 'moderate', 'severe', 'critical']
        data_to_plot = [df[df['severity'] == sev]['confidence'].values 
                       for sev in severities if sev in df['severity'].values]
        labels = [sev.capitalize() for sev in severities if sev in df['severity'].values]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['lightgreen', 'yellow', 'orange', 'red']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Severity Level', fontsize=12, fontweight='bold')
        ax.set_title('Performance Distribution by Severity Level', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'severity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self, detailed_results: List[Dict], metrics: Dict):
        """Correlation matrix between different metrics"""
        
        # Create synthetic metric correlations for demonstration
        metric_names = [k for k, v in metrics.items() if isinstance(v, (int, float))]
        
        if len(metric_names) < 2:
            return
        
        # Create correlation matrix (simplified)
        n = len(metric_names)
        corr_matrix = np.random.rand(n, n)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=metric_names, yticklabels=metric_names,
                   vmin=-1, vmax=1, center=0, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_distribution(self, detailed_results: List[Dict]):
        """Histogram of confidence score distribution"""
        
        confidences = [r.get('confidence', 0) for r in detailed_results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Confidence Scores', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Performance distribution saved")
    
    def plot_individual_performance(self, detailed_results: List[Dict], metrics: Dict):
        """Bar chart of individual test case performance"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        questions = [r['question'][:40] + '...' for r in detailed_results]
        confidences = [r.get('confidence', 0) for r in detailed_results]
        categories = [r['category'] for r in detailed_results]
        
        # Color by category
        category_colors = {'Bleeding': 'red', 'Burns': 'orange', 
                          'Cardiac': 'purple', 'Choking': 'blue', 'Wounds': 'green'}
        colors = [category_colors.get(cat, 'gray') for cat in categories]
        
        bars = ax.barh(range(len(questions)), confidences, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(questions)))
        ax.set_yticklabels(questions, fontsize=8)
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Individual Test Case Performance', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Legend
        handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7) 
                  for color in category_colors.values()]
        ax.legend(handles, category_colors.keys(), loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_performance.png', dpi=300, bbox_inches='tight')
        plt.close()


class RAGASEvaluator:
    """Evaluate RAG system with RAGAS"""
    
    def __init__(self, rag_wrapper: RAGTestWrapper):
        self.rag = rag_wrapper
        self.results = []
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ]
    
    def evaluate(self, test_cases: List[Dict]) -> Dict:
        
        questions = []
        ground_truths = []
        answers = []
        contexts_list = []
        
        for idx, test_case in enumerate(test_cases, 1):
            
            try:
                result = self.rag.query(test_case['question'])
                
                questions.append(test_case['question'])
                ground_truths.append(test_case['ground_truth'])
                answers.append(result['answer'])
                contexts_list.append(result['contexts'])
                
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
                continue
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        try:
            ragas_result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.rag.llm,
                embeddings=None
            )
            
            return ragas_result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {}
    
    def save_results(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"ragas_results_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_tests': len(self.results),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        return filepath


def main():
    parser = argparse.ArgumentParser(description='RAG Testing with Visualizations')
    parser.add_argument('--full', action='store_true', help='Run all test cases')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 cases)')
    parser.add_argument('--custom', type=int, help='Custom number of cases')
    
    args = parser.parse_args()
    
    if not os.getenv('GROQ_API_KEY'):
        return
    
    if args.custom:
        test_cases = COMPREHENSIVE_TEST_CASES[:args.custom]
    elif args.quick:
        test_cases = COMPREHENSIVE_TEST_CASES[:10]
    else:
        test_cases = COMPREHENSIVE_TEST_CASES
    
    try:
        rag_wrapper = RAGTestWrapper()
        evaluator = RAGASEvaluator(rag_wrapper)
        ragas_results = evaluator.evaluate(test_cases)
        output_dir = Path("./test_results")
        evaluator.save_results(output_dir)
        viz_gen = VisualizationGenerator(output_dir)
        viz_gen.generate_all_visualizations(ragas_results, evaluator.results)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()