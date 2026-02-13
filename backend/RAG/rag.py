

import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from pinecone import Pinecone
from pymongo import MongoClient
from collections import Counter

from RAG.embeddings import EmbeddingGenerator
from RAG.query_processor import QueryProcessor
from RAG.response_generator import ResponseGenerator

load_dotenv()

logger = logging.getLogger(__name__)


class FirstAidRAGAssistant:
    """Main RAG assistant combining all components"""
    
    def __init__(
        self,
        pinecone_api_key: str = None,
        mongodb_uri: str = None,
        groq_api_key: str = None,
        biobert_model: str = "dmis-lab/biobert-v1.1",
        index_name: str = "first-aid-assistant",
        groq_model: str = "llama-3.3-70b-versatile",
        log_level: int = logging.INFO
    ):
        """
        Initialize First Aid RAG Assistant
        
        Args:
            pinecone_api_key: Pinecone API key
            mongodb_uri: MongoDB connection URI
            groq_api_key: Groq API key
            biobert_model: BioBERT model identifier
            index_name: Pinecone index name
            groq_model: Groq model identifier
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Get API keys
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.groq_model = groq_model
        
        if not all([self.pinecone_api_key, self.groq_api_key, self.mongodb_uri]):
            raise ValueError("Missing required API keys")
        
        logger.info("=" * 70)
        logger.info("FIRST AID RAG ASSISTANT")
        logger.info("=" * 70)
        
        # Initialize components
        self.embedding_gen = EmbeddingGenerator(biobert_model)
        self.query_processor = QueryProcessor()
        self.response_gen = ResponseGenerator(self.groq_api_key, groq_model)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(index_name)
        stats = self.index.describe_index_stats()
        logger.info(f"Pinecone connected: {stats.total_vector_count} vectors")
        
        # Initialize MongoDB
        logger.info("Connecting to MongoDB...")
        self.mongo_client = MongoClient(self.mongodb_uri)
        self.db = self.mongo_client['first_aid_db']
        self.scenarios_collection = self.db['scenarios']
        self.chunks_collection = self.db['chunks']
        self.conversations_collection = self.db['conversations']
        self.chat_history_collection = self.db['chat_history']
        
        scenario_count = self.scenarios_collection.count_documents({})
        chunk_count = self.chunks_collection.count_documents({})
        logger.info(f"MongoDB connected: {scenario_count} scenarios, {chunk_count} chunks")
        
        logger.info("=" * 70)
        logger.info("First Aid Assistant Ready!")
        logger.info("=" * 70)
    
    def search_relevant_chunks(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using semantic search
        
        Args:
            query: User query
            top_k: Number of results to return
            min_score: Minimum relevance score
            
        Returns:
            List of relevant chunks with metadata
        """
        # Preprocess query
        processed_query = self.query_processor.preprocess_query(query)
        
        # Expand query
        expanded_queries = self.query_processor.expand_query(processed_query)
        
        # Generate embeddings for all query variations
        all_results = []
        for q in expanded_queries:
            embedding = self.embedding_gen.generate_embedding(q)
            
            # Search Pinecone
            results = self.index.query(
                vector=embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            all_results.extend(results.matches)
        
        # Deduplicate and sort by score
        seen_ids = set()
        unique_results = []
        for match in all_results:
            if match.id not in seen_ids and match.score >= min_score:
                seen_ids.add(match.id)
                unique_results.append({
                    'chunk_id': match.id,
                    'score': match.score,
                    'metadata': match.metadata,
                    'text': match.metadata.get('text', '')
                })
        
        # Sort by score and return top results
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results[:top_k]
    
    def get_full_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve full chunk content from MongoDB
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of full chunks
        """
        chunks = list(self.chunks_collection.find({'chunk_id': {'$in': chunk_ids}}))
        return chunks
    
    def answer_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.60,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a first aid query
        
        Args:
            query: User query
            conversation_id: Optional conversation ID
            top_k: Number of chunks to retrieve
            min_score: Minimum relevance score
            verbose: Enable verbose logging
            
        Returns:
            Response dictionary with answer and metadata
        """
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Query: {query}")
        
        # Search for relevant chunks
        relevant_chunks = self.search_relevant_chunks(query, top_k, min_score)
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Get conversation history if available
        conversation_history = None
        if conversation_id:
            history = list(self.chat_history_collection.find(
                {'conversation_id': conversation_id}
            ).sort('timestamp', -1).limit(4))
            
            conversation_history = [
                {'role': msg['role'], 'content': msg['content']}
                for msg in reversed(history)
            ]
        
        # Generate response
        if relevant_chunks:
            response = self.response_gen.generate_response(
                query,
                relevant_chunks,
                conversation_history
            )
            
            # Calculate confidence
            avg_score = sum(c['score'] for c in relevant_chunks) / len(relevant_chunks)
            confidence = "high" if avg_score >= 0.75 else "medium" if avg_score >= 0.60 else "low"
        else:
            logger.warning("No relevant chunks found, using fallback response")
            response = self.response_gen.generate_fallback_response(query)
            avg_score = 0.0
            confidence = "low"
        
        # Prepare sources
        sources = [
            {
                'title': chunk['metadata'].get('title', 'Unknown'),
                'category': chunk['metadata'].get('category', 'Unknown'),
                'source': chunk['metadata'].get('source', 'Unknown'),
                'relevance': chunk['score']
            }
            for chunk in relevant_chunks[:5]
        ]
        
        result = {
            'query': query,
            'response': response,
            'sources': sources,
            'confidence': confidence,
            'chunks_found': len(relevant_chunks),
            'avg_relevance': avg_score,
            'performance': {
                'chunks_retrieved': len(relevant_chunks),
                'avg_score': avg_score
            }
        }
        
        logger.info(f"Response generated (confidence: {confidence})")
        
        return result
    
    def interactive_mode(self):
        """Run interactive chat mode"""
        logger.info("=" * 70)
        logger.info("FIRST AID ASSISTANT - INTERACTIVE MODE")
        logger.info("=" * 70)
        logger.info("Ask any first aid question. Type 'help' for examples.")
        logger.info("")
        logger.info("Commands:")
        logger.info("  'quit' or 'exit' - Exit")
        logger.info("  'help' - Show examples")
        logger.info("  'new' - Start new conversation")
        logger.info("=" * 70)
        
        conversation_id = f"conv_{int(datetime.utcnow().timestamp())}"
        logger.info(f"Conversation ID: {conversation_id}")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("Stay safe!")
                    break
                
                if query.lower() == 'help':
                    self._show_examples()
                    continue
                
                if query.lower() == 'new':
                    conversation_id = f"conv_{int(datetime.utcnow().timestamp())}"
                    logger.info(f"New conversation: {conversation_id}")
                    continue
                
                # Answer query
                result = self.answer_query(
                    query,
                    conversation_id=conversation_id,
                    verbose=False
                )
                
                # Display response
                print("\n" + "-" * 70)
                print(result['response'])
                print("\n" + "-" * 70)
                print(f"Confidence: {result['confidence']} | Relevance: {result.get('avg_relevance', 0):.3f}")
                
            except KeyboardInterrupt:
                logger.info("Exiting...")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
    
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
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--query', '-q', type=str, help='Single query')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        assistant = FirstAidRAGAssistant()
        
        if args.interactive:
            assistant.interactive_mode()
        elif args.query:
            result = assistant.answer_query(args.query, verbose=args.verbose)
            print("\n" + "=" * 70)
            print("RESPONSE:")
            print("=" * 70)
            print(result['response'])
        else:
            parser.print_help()
            print("\nRun: python -m rag.rag --interactive")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
