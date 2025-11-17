"""
RAG Assistant with Improved Accuracy
Key improvements:
1. Query expansion and preprocessing
2. Hybrid search (semantic + keyword)
3. Lower similarity threshold with smart filtering
4. Multiple retrieval strategies
5. Better context building
"""

import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv
from collections import Counter

load_dotenv()


class FirstAidRAGAssistant:
    
    def __init__(
        self,
        pinecone_api_key: str = None,
        mongodb_uri: str = None,
        groq_api_key: str = None,
        biobert_model: str = "dmis-lab/biobert-v1.1",
        index_name: str = "first-aid-assistant",
        groq_model: str = "llama-3.3-70b-versatile"
    ):
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.groq_model = groq_model
        
        if not all([self.pinecone_api_key, self.groq_api_key, self.mongodb_uri]):
            raise ValueError("Missing API keys")
        
        print("="*70)
        print(" "*10 + "FIRST AID RAG ASSISTANT")
        print("="*70)
        
        # Initialize components
        print(f"\nLoading BioBERT: {biobert_model}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(biobert_model)
        self.embedding_model = AutoModel.from_pretrained(biobert_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        print(f"BioBERT loaded on {self.device}")
        
        print("\nConnecting to Pinecone...")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(index_name)
        stats = self.index.describe_index_stats()
        print(f"Pinecone: {stats.total_vector_count} vectors")
        
        print("\nConnecting to MongoDB...")
        self.mongo_client = MongoClient(self.mongodb_uri)
        self.db = self.mongo_client['first_aid_db']
        self.scenarios_collection = self.db['scenarios']
        self.chunks_collection = self.db['chunks']
        self.conversations_collection = self.db['conversations']
        self.chat_history_collection = self.db['chat_history']
        
        scenario_count = self.scenarios_collection.count_documents({})
        chunk_count = self.chunks_collection.count_documents({})
        print(f"MongoDB: {scenario_count} scenarios, {chunk_count} chunks")
        
        print(f"\nConnecting to Groq API ({groq_model})...")
        self.groq_client = Groq(api_key=self.groq_api_key)
        print("Groq API connected")
        
        # Medical terminology mapping for query expansion
        self.medical_synonyms = {
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
        
        # System prompt
        self.system_prompt = """You are an expert first aid assistant with access to authoritative medical sources. 

**Response Format**:

WARNING: Immediate Action
[Critical first steps - include "CALL 911/999 IMMEDIATELY" for life-threatening situations]

Step-by-Step Instructions
1. [First action]
2. [Second action]
3. [Continue with clear steps]

When to Seek Medical Help
- [Warning sign 1]
- [Warning sign 2]

What NOT to Do
- [Avoid 1]
- [Avoid 2]

Additional Notes
[Important context, warnings, or tips]


**Guidelines**:
- Use simple, clear language
- Be specific and actionable
- Always prioritize safety
- Cite sources used
- If information is limited, say so clearly
- For emergencies, emphasize calling 911/999"""
        
        print("\n" + "="*70)
        print(f"First Aid Assistant Ready!")
        print("="*70)
    
    def clean_response_format(self, text: str) -> str:
        """
        Remove common markdown artifacts (headings, bold/asterisk, inline code),
        normalize bullets and collapse excessive blank lines so the UI receives
        plain, clean text.
        """
        if not text:
            return text

        # Remove markdown headings like ##, ### etc.
        text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)

        # Remove bold/italic markup **bold**, *italic*
        text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text, flags=re.DOTALL)

        # Remove inline code `code`
        text = re.sub(r'`(.+?)`', r'\1', text, flags=re.DOTALL)

        # Normalize bullet characters (leading -, •, *, etc.)
        text = re.sub(r'(?m)^[\s]*[-\*\u2022]\s+', '• ', text)

        # Collapse 3+ newlines to exactly 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Trim
        return text.strip()
    
    def preprocess_query(self, query: str) -> str:
        """Clean and normalize query"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove question marks and exclamation points at the end
        query = re.sub(r'[?!]+$', '', query).strip()
        
        return query
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with medical synonyms and related terms"""
        expanded_queries = [query]
        query_lower = query.lower()
        
        # Add synonym expansions
        for term, synonyms in self.medical_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:5]:
                    expanded = query_lower.replace(term, syn)
                    if expanded not in expanded_queries:
                        expanded_queries.append(expanded)
        
        return expanded_queries[:5]
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important medical keywords from query"""
        # Common stop words to ignore
        stop_words = {'is', 'am', 'are', 'what', 'how', 'do', 'to', 'the', 'a', 'an', 
                     'my', 'i', 'on', 'for', 'with', 'from', 'and', 'or', 'should',
                     'can', 'will', 'having', 'have', 'has'}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate BioBERT embedding"""
        inputs = self.embedding_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.60
    ) -> List[Dict]:
        """
        Hybrid search combining:
        1. Semantic search (embeddings)
        2. Query expansion
        3. Keyword matching
        """
        all_chunks = []
        seen_chunk_ids = set()
        
        # Preprocess query
        clean_query = self.preprocess_query(query)
        
        # 1. Semantic search with original query
        query_embedding = self.generate_embedding(clean_query)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        for match in results.matches:
            if match.score >= min_score and match.id not in seen_chunk_ids:
                chunk = self._build_chunk_from_match(match)
                if chunk:
                    chunk['search_method'] = 'semantic_original'
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(match.id)
        
        # 2. Search with expanded queries
        expanded_queries = self.expand_query(clean_query)
        for exp_query in expanded_queries[1:]:
            exp_embedding = self.generate_embedding(exp_query)
            results = self.index.query(
                vector=exp_embedding.tolist(),
                top_k=5,
                include_metadata=True
            )
            
            for match in results.matches:
                if match.score >= min_score and match.id not in seen_chunk_ids:
                    chunk = self._build_chunk_from_match(match)
                    if chunk:
                        chunk['search_method'] = 'semantic_expanded'
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(match.id)
        
        # 3. Keyword-based MongoDB search (fallback)
        if len(all_chunks) < 3:
            keywords = self.extract_keywords(query)
            keyword_chunks = self._keyword_search_mongodb(keywords, limit=5)
            
            for chunk in keyword_chunks:
                if chunk['chunk_id'] not in seen_chunk_ids:
                    chunk['search_method'] = 'keyword_fallback'
                    chunk['score'] = 0.65
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk['chunk_id'])
        
        # Sort by score and return top results
        all_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return all_chunks[:top_k]
    
    def _build_chunk_from_match(self, match) -> Optional[Dict]:
        """Build chunk dict from Pinecone match"""
        try:
            chunk_id = match.id
            full_chunk = self.chunks_collection.find_one({'chunk_id': chunk_id})
            
            return {
                'chunk_id': chunk_id,
                'score': match.score,
                'text': full_chunk['text'] if full_chunk else match.metadata.get('text', ''),
                'title': match.metadata.get('title', ''),
                'category': match.metadata.get('category', ''),
                'severity': match.metadata.get('severity', ''),
                'source': match.metadata.get('source', ''),
                'scenario_id': match.metadata.get('scenario_id', '')
            }
        except Exception as e:
            print(f"Error building chunk: {e}")
            return None
    
    def _keyword_search_mongodb(self, keywords: List[str], limit: int = 5) -> List[Dict]:
        """Search MongoDB using keywords as fallback"""
        try:
            # Create text search query
            search_terms = " ".join(keywords)
            
            # Search in chunks collection
            results = self.chunks_collection.find(
                {"$text": {"$search": search_terms}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            chunks = []
            for doc in results:
                chunks.append({
                    'chunk_id': doc.get('chunk_id', ''),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'category': doc.get('category', ''),
                    'severity': doc.get('severity', ''),
                    'source': doc.get('source', ''),
                    'scenario_id': doc.get('scenario_id', ''),
                    'score': 0.60  
                })
            
            return chunks
            
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []
    
    def rerank_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Rerank chunks based on relevance to query"""
        if not chunks:
            return chunks
        
        query_terms = set(self.extract_keywords(query))
        
        for chunk in chunks:
            # Calculate keyword overlap
            chunk_terms = set(self.extract_keywords(chunk['text']))
            overlap = len(query_terms & chunk_terms)
            
            # Boost score based on keyword overlap
            chunk['keyword_overlap'] = overlap
            chunk['boosted_score'] = chunk['score'] + (overlap * 0.02)
        
        # Sort by boosted score
        chunks.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        return chunks
    
    def build_rich_context(self, chunks: List[Dict], max_chunks: int = 6) -> str:
        """Build comprehensive context from multiple chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            source_name = chunk['source'].split(':')[0] if chunk['source'] else 'Unknown'
            
            context = f"""
SOURCE {i}: {source_name}
Title: {chunk['title']}
Category: {chunk['category']} | Severity: {chunk['severity']}
Relevance: {chunk['score']:.3f} | Method: {chunk.get('search_method', 'N/A')}

Content:
{chunk['text']}
"""
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    def generate_response_with_groq(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: List[Dict] = None
    ) -> str:
        """Generate response using Groq with rich context and return cleaned text."""
        # Build comprehensive context
        context_text = self.build_rich_context(context_chunks)

        user_prompt = f"""Using the authoritative medical information below, provide a comprehensive first aid response.

Question: {query}

Medical Knowledge Base:
{context_text}

Instructions:
1. Follow the structured response format
2. Be specific and actionable
3. Cite which sources you used
4. If information is limited, acknowledge it
5. For life-threatening situations, emphasize calling emergency services
6. Use simple, clear language"""

        try:
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # Add chat history (last 3 exchanges)
            if chat_history:
                for msg in chat_history[-6:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": user_prompt})

            # Call Groq
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2500,
                top_p=0.9
            )

            raw_text = response.choices[0].message.content
            # Clean markdown/asterisk artifacts before returning
            return self.clean_response_format(raw_text)

        except Exception as e:
            # Return a safe, helpful fallback but cleaned too
            fallback = (f"Error generating response: {str(e)}\n\n"
                        "For any medical emergency, call 911/999 immediately.\n\n"
                        "If this is an emergency:\n"
                        "1. Call emergency services (911/999)\n"
                        "2. Stay calm and follow dispatcher instructions\n"
                        "3. If trained, provide appropriate first aid until help arrives")
            return self.clean_response_format(fallback)
    
    def get_chat_history(self, conversation_id: str, limit: int = 6) -> List[Dict]:
        """Retrieve chat history"""
        try:
            history = list(self.chat_history_collection.find(
                {"conversation_id": conversation_id}
            ).sort("timestamp", 1))
            
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]
        except Exception as e:
            print(f"Could not retrieve chat history: {e}")
            return []
    
    def save_to_chat_history(self, conversation_id: str, role: str, content: str):
        """Save message to chat history"""
        try:
            self.chat_history_collection.insert_one({
                'conversation_id': conversation_id,
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            print(f"Could not save to chat history: {e}")
    
    def answer_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 10,
        verbose: bool = False
    ) -> Dict:
        """Main method to answer queries with retrieval"""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}\n")
        
        # Get chat history
        chat_history = []
        if conversation_id:
            chat_history = self.get_chat_history(conversation_id)
            if verbose and chat_history:
                print(f"Loaded {len(chat_history)} previous messages\n")
        
        # Enhanced hybrid search
        if verbose:
            print("Performing hybrid search (semantic + keyword)...")
        
        chunks = self.hybrid_search(query, top_k=top_k)
        
        if verbose:
            print(f"Found {len(chunks)} relevant chunks")
            for i, chunk in enumerate(chunks[:5], 1):
                method_icon = {"semantic_original": "[O]", "semantic_expanded": "[E]", "keyword_fallback": "[K]"}.get(chunk.get('search_method', ''), '[?]')
                print(f"  {method_icon} {i}. {chunk['title'][:60]} (score: {chunk['score']:.3f})")
        
        # Rerank chunks
        chunks = self.rerank_chunks(chunks, query)
        
        # After reranking
        if not chunks:
            expanded = self.expand_query(self.preprocess_query(query))
            extra_chunks = []
            for alt in expanded[1:]:
                try:
                    emb = self.generate_embedding(alt)
                    res = self.index.query(vector=emb.tolist(), top_k=3, include_metadata=True)
                    for m in res.matches:
                        if m.id not in [c.get('chunk_id') for c in extra_chunks]:
                            c = self._build_chunk_from_match(m)
                            if c:
                                c['search_method'] = 'semantic_expanded_final'
                                extra_chunks.append(c)
                except Exception:
                    continue

            if extra_chunks:
                chunks = extra_chunks
                chunks = self.rerank_chunks(chunks, query)
            else:
                fallback_response = self._generate_fallback_response(query)
                cleaned = self.clean_response_format(fallback_response)
                return {
                    'query': query,
                    'response': cleaned,
                    'sources': [],
                    'confidence': 'LOW',
                    'chunks_found': 0,
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        # Generate response
        if verbose:
            print("\nGenerating response with Groq...\n")
        
        response_text = self.generate_response_with_groq(query, chunks, chat_history)
        
        # Calculate confidence
        avg_score = sum(c['boosted_score'] for c in chunks) / len(chunks)
        if avg_score > 0.80:
            confidence = 'HIGH'
        elif avg_score > 0.65:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Prepare response
        response_obj = {
            'query': query,
            'response': response_text,
            'sources': [
                {
                    'title': c['title'],
                    'source': c['source'].split(':')[0] if c['source'] else 'Unknown',
                    'category': c['category'],
                    'severity': c['severity'],
                    'relevance_score': round(c['score'], 3),
                    'search_method': c.get('search_method', 'N/A')
                }
                for c in chunks[:6]
            ],
            'confidence': confidence,
            'chunks_found': len(chunks),
            'avg_relevance': round(avg_score, 3),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Save to MongoDB
        if conversation_id:
            self.save_to_chat_history(conversation_id, "user", query)
            self.save_to_chat_history(conversation_id, "assistant", response_text)
            self._save_conversation(conversation_id, response_obj)
        
        return response_obj
    
    def _generate_fallback_response(self, query: str) -> str:
        """Advanced context-aware fallback with both injury & symptom detection."""
        
        query_lower = query.lower()
        
        # Context keyword maps
        injury_map = {
            "sprain": "sprain",
            "ankle": "sprain",
            "wrist": "sprain",
            "strain": "sprain",
            "cut": "bleeding",
            "bleed": "bleeding",
            "blood": "bleeding",
            "burn": "burn",
            "fire": "burn",
            "scald": "burn",
            "acid": "acid_burn",
            "chemical": "acid_burn",
            "fracture": "fracture",
            "broken": "fracture",
            "bone": "fracture",
            "choke": "choking",
            "airway": "choking",
            "poison": "poisoning",
            "overdose": "poisoning",
            "snake": "snake_bite",
            "bite": "snake_bite",
            "sting": "snake_bite",
            "allergy": "allergic_reaction",
            "anaphylaxis": "allergic_reaction",
            "shock": "shock",
            "seizure": "seizure",
            "fit": "seizure",
            "faint": "fainting",
            "unconscious": "fainting",
            "breath": "breathing",
            "asthma": "breathing",
            "chest pain": "heart_attack",
            "heart": "heart_attack",
        }
        
        symptom_map = {
            "vomit": "nausea_vomiting",
            "nausea": "nausea_vomiting",
            "headache": "headache",
            "migraine": "headache",
            "dizzy": "dizziness",
            "lightheaded": "dizziness",
            "sweat": "heat_exhaustion",
            "hot": "heat_exhaustion",
            "heat": "heat_exhaustion",
            "fever": "fever",
            "muscle ache": "flu_like_illness",
            "body ache": "flu_like_illness",
            "chills": "flu_like_illness",
            "diarrhea": "food_poisoning",
            "stomach pain": "food_poisoning",
            "cramps": "food_poisoning",
        }
        
        # Match context
        detected_context = None
        for k, v in {**injury_map, **symptom_map}.items():
            if k in query_lower:
                detected_context = v
                break
        
        # Tailored advice for each case
        responses = {
            # Injury cases
            "sprain": (
                "⚠️ **Immediate Action**\n"
                "• Stop any activity that causes pain.\n"
                "• Apply an ice pack wrapped in a towel for 15-20 minutes every 2 hours.\n"
                "• Compress with a soft bandage (not too tight).\n"
                "• Elevate the injured limb above heart level.\n\n"
                "**When to Seek Medical Help**\n"
                "• You can't move or bear weight.\n"
                "• Pain or swelling worsens after 48 hours."
            ),
            "bleeding": (
                "⚠️ **Immediate Action**\n"
                "• Apply firm pressure with a clean cloth or bandage.\n"
                "• Keep pressure for at least 10 minutes.\n"
                "• Elevate the wounded area if possible.\n\n"
                "**When to Seek Medical Help**\n"
                "• Bleeding won't stop or blood spurts.\n"
                "• The wound is deep or contaminated."
            ),
            "burn": (
                "⚠️ **Immediate Action**\n"
                "• Cool the burn under cool running water for at least 10 minutes.\n"
                "• Remove jewelry or tight clothing near the burn.\n"
                "• Cover with sterile gauze (no creams or butter).\n\n"
                "**When to Seek Medical Help**\n"
                "• The burn is larger than your palm.\n"
                "• It affects the face, joints, or genitals."
            ),
            "acid_burn": (
                "⚠️ **Immediate Action**\n"
                "Since the provided sources do not directly address acid burns, it's crucial to act quickly and carefully. "
                "If the acid burn is severe or covers a large area, CALL 911/999 IMMEDIATELY.\n\n"
                "**Step-by-Step Instructions**\n"
                "1. Remove any contaminated clothing or jewelry from the affected area to prevent further damage.\n"
                "2. Rinse the affected area with cool or lukewarm water for at least 20 minutes to help neutralize the acid. "
                "Avoid using hot water, as it can activate the acid and cause further damage.\n"
                "3. After rinsing, cover the affected area with a sterile, non-stick dressing or a clean cloth to protect it from further irritation.\n\n"
                "**When to Seek Medical Help**\n"
                "• If the burn is severe, large, or deep\n"
                "• If the burn is on the face, hands, or feet\n"
                "• If you experience difficulty breathing, as some acids can release toxic fumes\n\n"
                "**What NOT to Do**\n"
                "• Do not apply ice or ice water to the burn, as this can cause further damage\n"
                "• Do not use harsh or abrasive cleansers, as they can irritate the skin and worsen the burn\n\n"
                "**Additional Notes**\n"
                "The provided sources do not offer specific guidance on treating acid burns. However, general first aid principles "
                "suggest rinsing the affected area with cool water and seeking medical attention if the burn is severe or covers a large area. "
                "It's essential to consult a medical professional or a reliable first aid resource for specific guidance on treating acid burns. "
                "The American Red Cross and NHS UK websites may have more comprehensive information on treating acid burns."
            ),
            "fracture": (
                "⚠️ **Immediate Action**\n"
                "• Keep the injured limb still and supported.\n"
                "• Apply ice wrapped in a cloth to reduce swelling.\n"
                "• Do not move or straighten the limb.\n\n"
                "**When to Seek Medical Help**\n"
                "• Bone is visible or limb looks deformed.\n"
                "• Pain or swelling is severe.\n"
                "CALL 911/999 immediately for severe fractures."
            ),
            "choking": (
                "⚠️ **Immediate Action**\n"
                "• Ask them to cough strongly.\n"
                "• If unable to breathe or talk, perform 5 back blows, then 5 abdominal thrusts.\n"
                "• Alternate until the obstruction clears.\n\n"
                "**When to Seek Medical Help**\n"
                "• They remain unable to breathe or lose consciousness.\n"
                "CALL 911/999 immediately."
            ),
            "poisoning": (
                "⚠️ **Immediate Action**\n"
                "• Call a poison helpline or emergency services immediately.\n"
                "• Do not induce vomiting.\n"
                "• Keep the substance container for reference.\n\n"
                "CALL 911/999 IMMEDIATELY."
            ),
            "snake_bite": (
                "⚠️ **Immediate Action**\n"
                "• Keep the person calm and still.\n"
                "• Keep bite area below heart level.\n"
                "• Remove tight items and call emergency help immediately.\n\n"
                "**What NOT to Do**\n"
                "• Do not cut, suck, or apply a tourniquet.\n\n"
                "CALL 911/999 IMMEDIATELY."
            ),
            "allergic_reaction": (
                "⚠️ **Immediate Action**\n"
                "• If the person has an EpiPen, help them use it immediately.\n"
                "• Call emergency services right away.\n"
                "• Keep the person lying down with legs elevated.\n"
                "• Monitor breathing and be ready to perform CPR.\n\n"
                "CALL 911/999 IMMEDIATELY."
            ),
            "shock": (
                "⚠️ **Immediate Action**\n"
                "• Call emergency services immediately.\n"
                "• Lay the person down with legs elevated (unless spinal injury suspected).\n"
                "• Keep them warm with blankets.\n"
                "• Do not give food or water.\n\n"
                "CALL 911/999 IMMEDIATELY."
            ),
            "seizure": (
                "⚠️ **Immediate Action**\n"
                "• Protect the person from injury by clearing the area.\n"
                "• Place something soft under their head.\n"
                "• Time the seizure.\n"
                "• Turn them on their side after the seizure ends.\n\n"
                "**What NOT to Do**\n"
                "• Do not restrain them or put anything in their mouth.\n\n"
                "**When to Seek Medical Help**\n"
                "• Seizure lasts more than 5 minutes\n"
                "• Person doesn't regain consciousness\n"
                "• This is their first seizure"
            ),
            "fainting": (
                "⚠️ **Immediate Action**\n"
                "• Help the person lie down flat.\n"
                "• Raise legs above heart level.\n"
                "• Loosen tight clothing.\n"
                "• Check breathing and responsiveness.\n\n"
                "**When to Seek Medical Help**\n"
                "• If unresponsive or not breathing, start CPR and call for help.\n"
                "• If they don't recover within a minute."
            ),
            "breathing": (
                "⚠️ **Immediate Action**\n"
                "• Help them sit upright in a comfortable position.\n"
                "• Loosen tight clothing.\n"
                "• If they have an inhaler, help them use it.\n"
                "• Encourage slow, deep breaths.\n\n"
                "**When to Seek Medical Help**\n"
                "• Breathing doesn't improve\n"
                "• Lips or face turn blue\n"
                "• Person becomes unresponsive\n\n"
                "CALL 911/999 if severe."
            ),
            "heart_attack": (
                "⚠️ **Immediate Action**\n"
                "CALL 911/999 IMMEDIATELY.\n"
                "• Keep the person calm and seated.\n"
                "• Loosen tight clothing.\n"
                "• Help them chew one regular aspirin (unless allergic).\n"
                "• Be ready to perform CPR if breathing stops."
            ),
            
            # Symptom cases
            "nausea_vomiting": (
                "**Possible causes:** food poisoning, stomach infection, migraine, or dehydration.\n\n"
                "⚠️ **Immediate Care**\n"
                "• Sip small amounts of water or an electrolyte drink.\n"
                "• Avoid solid food until vomiting subsides.\n"
                "• Rest in a cool, ventilated place.\n\n"
                "**When to Seek Medical Help**\n"
                "• Vomiting lasts more than 24 hours or you can't keep fluids down.\n"
                "• There's blood in vomit or severe stomach pain."
            ),
            "headache": (
                "**Possible causes:** tension headache, dehydration, migraine, stress, or heat.\n\n"
                "⚠️ **Relief Steps**\n"
                "• Rest in a quiet, dark room.\n"
                "• Drink water - dehydration can worsen headaches.\n"
                "• Apply a cold compress to your forehead.\n\n"
                "**When to Seek Medical Help**\n"
                "• The headache is sudden and severe.\n"
                "• Vision changes, confusion, or vomiting occur."
            ),
            "dizziness": (
                "**Possible causes:** dehydration, low blood sugar, fatigue, or fainting onset.\n\n"
                "⚠️ **What to Do**\n"
                "• Sit or lie down immediately.\n"
                "• Drink water or an electrolyte solution.\n"
                "• Eat something light if you haven't eaten recently.\n\n"
                "**When to Seek Medical Help**\n"
                "• Dizziness lasts long or occurs with chest pain or shortness of breath."
            ),
            "heat_exhaustion": (
                "**Possible cause:** prolonged heat exposure or dehydration.\n\n"
                "⚠️ **Immediate Action**\n"
                "• Move to a cool, shaded area.\n"
                "• Loosen clothing and apply cool damp cloths.\n"
                "• Sip water or electrolyte solution slowly.\n\n"
                "**When to Seek Medical Help**\n"
                "• Vomiting or confusion develops.\n"
                "• Body temperature stays above 38°C (100.4°F)."
            ),
            "flu_like_illness": (
                "**Possible cause:** viral infection or seasonal flu.\n\n"
                "⚠️ **Self-care**\n"
                "• Rest and stay hydrated.\n"
                "• Take paracetamol/acetaminophen for fever if needed.\n"
                "• Drink warm fluids and eat light foods.\n\n"
                "**When to Seek Medical Help**\n"
                "• Fever persists beyond 3 days.\n"
                "• Breathing difficulty or chest pain occurs."
            ),
            "food_poisoning": (
                "**Possible cause:** contaminated food or water.\n\n"
                "⚠️ **Care Steps**\n"
                "• Drink water or ORS to prevent dehydration.\n"
                "• Avoid dairy, caffeine, and heavy foods.\n"
                "• Rest until recovery.\n\n"
                "**When to Seek Medical Help**\n"
                "• Severe abdominal pain or blood in stool.\n"
                "• Vomiting lasts more than 24 hours."
            ),
            "fever": (
                "**Possible cause:** infection, flu, or heat illness.\n\n"
                "⚠️ **Care**\n"
                "• Stay hydrated and rest.\n"
                "• Apply a cool, damp cloth to forehead.\n"
                "• Take fever medication if recommended.\n\n"
                "**When to Seek Medical Help**\n"
                "• Fever greater than 39°C (102°F) or lasts more than 3 days.\n"
                "• There's persistent vomiting or confusion."
            ),
        }
        
        # Construct contextual response
        if detected_context and detected_context in responses:
            return (
                f"Based on your symptoms, this may indicate **{detected_context.replace('_', ' ').title()}**.\n\n"
                f"{responses[detected_context]}\n\n"
                "**Additional Notes**\n"
                "This is general first aid guidance. If symptoms worsen or you're unsure, seek professional medical advice."
            )
        
        # General safe fallback
        return (
            f"I couldn't find a direct match for: \"{query}\".\n\n"
            "⚠️ **General First Aid Steps**\n"
            "1. Ensure safety and check responsiveness.\n"
            "2. Call emergency services if pain, bleeding, or confusion is severe.\n"
            "3. Provide rest, hydration, and reassurance.\n"
            "4. Monitor symptoms and avoid unnecessary movement.\n\n"
            "**Additional Notes**\n"
            "If you feel worse or the issue persists, consult a doctor immediately."
        )
    
    def _save_conversation(self, conversation_id: str, response_obj: Dict):
        """Save conversation to MongoDB"""
        try:
            self.conversations_collection.insert_one({
                'conversation_id': conversation_id,
                **response_obj
            })
        except Exception as e:
            print(f"Could not save conversation: {e}")
    
    def interactive_mode(self):
        """Interactive chat mode"""
        print("\n" + "="*70)
        print("FIRST AID ASSISTANT - INTERACTIVE MODE")
        print("="*70)
        print("\nAsk any first aid question. Type 'help' for examples.")
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit")
        print("  'help' - Show examples")
        print("  'history' - Show conversation")
        print("  'new' - Start new conversation")
        print("="*70 + "\n")
        
        conversation_id = f"conv_{int(datetime.utcnow().timestamp())}"
        print(f"Conversation ID: {conversation_id}\n")
        
        while True:
            try:
                query = input("Your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nStay safe!")
                    break
                
                if query.lower() == 'help':
                    self._show_examples()
                    continue
                
                if query.lower() == 'new':
                    conversation_id = f"conv_{int(datetime.utcnow().timestamp())}"
                    print(f"\nNew conversation: {conversation_id}\n")
                    continue
                
                # Answer query
                result = self.answer_query(
                    query,
                    conversation_id=conversation_id,
                    verbose=False
                )
                
                # Display response
                print(f"\n{'-'*70}")
                print(result['response'])
                print(f"\n{'-'*70}")
                print(f"\nConfidence: {result['confidence']} | Relevance: {result.get('avg_relevance', 0):.3f}")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    def _show_examples(self):
        """Show example questions"""
        examples = [
            "What should I do for severe bleeding?",
            "Someone is choking and turning blue",
            "How to treat second degree burn?",
            "CPR steps for unconscious adult",
            "My friend drank too much alcohol",
            "Snake bite with visible wounds",
            "Severe headache with vomiting",
        ]
        print("\nExample questions:")
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='First Aid RAG Assistant')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--query', '-q', type=str)
    
    args = parser.parse_args()
    
    try:
        assistant = FirstAidRAGAssistant()
        
        if args.interactive:
            assistant.interactive_mode()
        elif args.query:
            result = assistant.answer_query(args.query, verbose=True)
            print(f"\n{'='*70}")
            print("RESPONSE:")
            print("="*70)
            print(result['response'])
        else:
            parser.print_help()
            print("\nRun: python rag.py --interactive")
    
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()