"""
Query preprocessing and expansion for RAG system
"""
import logging
import re
from typing import List, Dict

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


# Medical terminology mapping for query expansion
MEDICAL_SYNONYMS = {
    'choking': ['airway obstruction', 'blocked airway', 'cannot breathe', 'foreign object throat'],
    'bleeding': ['hemorrhage', 'blood loss', 'cut', 'wound', 'laceration'],
    'burn': ['scald', 'thermal injury', 'fire injury', 'heat damage'],
    'headache': ['head pain', 'migraine', 'cephalgia'],
    'poisoning': ['toxic ingestion', 'overdose', 'toxic exposure'],
    'heart attack': ['cardiac arrest', 'myocardial infarction', 'chest pain'],
    'fracture': ['broken bone', 'bone break', 'bone fracture'],
    'seizure': ['convulsion', 'fit', 'epileptic episode'],
    'allergic': ['anaphylaxis', 'allergic reaction', 'hypersensitivity'],
    'unconscious': ['unresponsive', 'loss of consciousness', 'passed out'],
    'breathing': ['respiration', 'respiratory', 'airway'],
    'snake': ['serpent', 'venomous bite', 'reptile bite'],
    'alcohol': ['intoxication', 'ethanol', 'drunk'],
    'heat': ['hyperthermia', 'heat stroke', 'overheating'],
    'cold': ['hypothermia', 'freezing', 'frostbite'],
}


class QueryProcessor:
    """Process and expand user queries for better retrieval"""
    
    def __init__(self, medical_synonyms: Dict[str, List[str]] = None):
        """
        Initialize query processor
        
        Args:
            medical_synonyms: Dictionary mapping medical terms to synonyms
        """
        self.medical_synonyms = medical_synonyms or MEDICAL_SYNONYMS
        logger.debug(f"QueryProcessor initialized with {len(self.medical_synonyms)} synonym mappings")
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and normalize query
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove trailing question marks and exclamation points
        query = re.sub(r'[?!]+$', '', query).strip()
        
        return query
    
    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand query with medical synonyms and related terms
        
        Args:
            query: Preprocessed query
            max_expansions: Maximum number of expansions to generate
            
        Returns:
            List of expanded query variations
        """
        expanded_queries = [query]
        query_lower = query.lower()
        
        # Add synonym expansions
        for term, synonyms in self.medical_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:max_expansions]:
                    expanded = query_lower.replace(term, syn)
                    if expanded not in expanded_queries:
                        expanded_queries.append(expanded)
        
        return expanded_queries[:max_expansions]
    
    def extract_keywords(self, query: str) -> List[str]:
       
        # Common stop words to ignore
        stop_words = {
            'is', 'am', 'are', 'what', 'how', 'do', 'to', 'the', 'a', 'an', 
            'my', 'i', 'on', 'for', 'with', 'from', 'and', 'or', 'should',
            'can', 'will', 'having', 'have', 'has'
        }
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def detect_emergency(self, query: str) -> bool:
        """
        Detect if query indicates an emergency situation
        
        Args:
            query: User query
            
        Returns:
            True if emergency detected, False otherwise
        """
        emergency_keywords = [
            'unconscious', 'not breathing', 'no pulse', 'severe bleeding',
            'chest pain', 'heart attack', 'stroke', 'seizure', 'anaphylaxis',
            'choking', 'poisoning', 'overdose', 'severe burn', 'head injury',
            'spinal injury', 'can\'t breathe', 'blue', 'unresponsive'
        ]
        
        query_lower = query.lower()
        for keyword in emergency_keywords:
            if keyword in query_lower:
                logger.warning(f"Emergency keyword detected in query: {keyword}")
                return True
        
        return False