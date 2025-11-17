"""
Phase 2: Process collected scenarios and store in Pinecone + MongoDB Atlas
"""

import os
import json
import numpy as np
from typing import List, Dict, Union
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class PineconeIntegrator:
    """
    Integrate collected scenarios with Pinecone vector database and MongoDB Atlas
    """
    
    def __init__(
        self,
        pinecone_api_key: str = None,
        mongodb_uri: str = None,
        biobert_model: str = "dmis-lab/biobert-v1.1"
    ):
        # Get API keys
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        
        # MongoDB Atlas URI (connection string from Atlas dashboard)
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment. Add your MongoDB Atlas connection string.")
        
        print("="*70)
        print(" "*15 + "PHASE 2: PINECONE + MONGODB ATLAS INTEGRATION")
        print("="*70)
        
        # Initialize BioBERT for embeddings
        print("\n🔧 Loading BioBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(biobert_model)
        self.model = AutoModel.from_pretrained(biobert_model)
        self.model.eval()
        print(f"✅ BioBERT loaded: {biobert_model}")
        
        # Initialize Pinecone
        print("\n🔧 Connecting to Pinecone...")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "first-aid-assistant"
        self._setup_pinecone_index()
        print(f"✅ Pinecone connected: {self.index_name}")
        
        # Initialize MongoDB Atlas
        print("\n🔧 Connecting to MongoDB Atlas...")
        try:
            # MongoDB Atlas connection with retry options
            self.mongo_client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            self.mongo_client.server_info()
            
            # Get database
            self.db = self.mongo_client['first_aid_db']
            self.scenarios_collection = self.db['scenarios']
            self.chunks_collection = self.db['chunks']
            
            # Create indexes for better performance
            self.scenarios_collection.create_index("scenario_id", unique=True)
            self.chunks_collection.create_index("chunk_id", unique=True)
            self.chunks_collection.create_index("scenario_id")
            
            print(f"✅ MongoDB Atlas connected: {self.db.name}")
            
        except Exception as e:
            print(f"❌ MongoDB Atlas connection failed: {e}")
            print("   Check your MONGODB_URI in .env file")
            print("   Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority")
            raise
        
        print("\n" + "="*70)
    
    def _setup_pinecone_index(self):
        """Create or connect to Pinecone index"""
        dimension = 768  # BioBERT embedding dimension
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"   Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        else:
            print(f"   Using existing index: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
    
    def chunk_scenario(self, scenario: Dict, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
        """
        Chunk a scenario into smaller pieces for embedding
        """
        # Validate scenario is a dictionary
        if not isinstance(scenario, dict):
            raise TypeError(f"Expected scenario to be dict, got {type(scenario).__name__}: {str(scenario)[:100]}")
        
        # Build comprehensive text
        parts = []
        
        if scenario.get('title'):
            parts.append(f"Title: {scenario['title']}")
        
        if scenario.get('category'):
            parts.append(f"Category: {scenario['category']}")
        
        if scenario.get('subcategory'):
            parts.append(f"Type: {scenario['subcategory']}")
        
        if scenario.get('symptoms'):
            symptoms = scenario['symptoms']
            if isinstance(symptoms, list):
                symptoms_text = "Symptoms: " + ", ".join(symptoms)
            else:
                symptoms_text = f"Symptoms: {symptoms}"
            parts.append(symptoms_text)
        
        if scenario.get('immediate_steps'):
            steps = scenario['immediate_steps']
            if isinstance(steps, list):
                steps_text = "Steps:\n" + "\n".join(
                    f"{i+1}. {step}" 
                    for i, step in enumerate(steps)
                )
            else:
                steps_text = f"Steps: {steps}"
            parts.append(steps_text)
        
        if scenario.get('when_to_seek_help'):
            warnings = scenario['when_to_seek_help']
            if isinstance(warnings, list):
                warnings_text = "Seek help when: " + ", ".join(warnings)
            else:
                warnings_text = f"Seek help when: {warnings}"
            parts.append(warnings_text)
        
        if scenario.get('do_not'):
            donts = scenario['do_not']
            if isinstance(donts, list):
                donts_text = "Do NOT: " + ", ".join(donts)
            else:
                donts_text = f"Do NOT: {donts}"
            parts.append(donts_text)
        
        if scenario.get('additional_info'):
            parts.append(f"Additional: {scenario['additional_info']}")
        
        full_text = "\n\n".join(parts)
        
        # Tokenize and chunk
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append({
                'chunk_id': f"{scenario['scenario_id']}_chunk_{chunk_idx}",
                'scenario_id': scenario['scenario_id'],
                'chunk_index': chunk_idx,
                'text': chunk_text,
                'metadata': {
                    'title': scenario.get('title', ''),
                    'category': scenario.get('category', ''),
                    'severity': scenario.get('severity', ''),
                    'source': scenario.get('source', ''),
                    'source_type': scenario.get('source_type', 'authoritative')
                }
            })
            
            chunk_idx += 1
            start += chunk_size - overlap
        
        return chunks
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate BioBERT embedding for text"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        return embedding
    
    def _load_and_validate_scenarios(self, json_filepath: str) -> List[Dict]:
        """Load and validate scenarios from JSON file"""
        print(f"\n📂 Loading scenarios from: {json_filepath}")
        
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        scenarios = []
        
        if isinstance(data, list):
            # Check if list items are dicts or strings
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    scenarios.append(item)
                elif isinstance(item, str):
                    # Try to parse string as JSON
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            scenarios.append(parsed)
                        else:
                            print(f"⚠️  Warning: Item {i} is a string that parsed to {type(parsed).__name__}, skipping")
                    except json.JSONDecodeError:
                        print(f"⚠️  Warning: Item {i} is not valid JSON, skipping: {item[:100]}")
                else:
                    print(f"⚠️  Warning: Item {i} is {type(item).__name__}, expected dict, skipping")
        
        elif isinstance(data, dict):
            # Check if it's a single scenario or wrapper object
            if 'scenarios' in data and isinstance(data['scenarios'], list):
                scenarios = data['scenarios']
            elif 'scenario_id' in data or 'title' in data:
                # Single scenario
                scenarios = [data]
            else:
                # Might be nested structure, try to extract scenarios
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"   Found list under key '{key}', checking items...")
                        for item in value:
                            if isinstance(item, dict):
                                scenarios.append(item)
                    elif isinstance(value, dict):
                        scenarios.append(value)
        
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data).__name__}")
        
        # Validate scenarios have required fields
        validated_scenarios = []
        for i, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                print(f"⚠️  Warning: Scenario {i} is {type(scenario).__name__}, skipping")
                continue
            
            # Add scenario_id if missing
            if 'scenario_id' not in scenario:
                scenario['scenario_id'] = f"scenario_{i+1}"
            
            validated_scenarios.append(scenario)
        
        print(f"📊 Found {len(validated_scenarios)} valid scenarios")
        
        if len(validated_scenarios) == 0:
            print("\n❌ No valid scenarios found!")
            print("   Please check your JSON file structure.")
            print("   Expected format: List of dictionaries or {'scenarios': [...]}")
            raise ValueError("No valid scenarios to process")
        
        # Show sample of first scenario
        if validated_scenarios:
            print(f"\n📋 Sample scenario structure:")
            sample = validated_scenarios[0]
            print(f"   Keys: {list(sample.keys())}")
            if 'title' in sample:
                print(f"   Title: {sample['title']}")
            if 'category' in sample:
                print(f"   Category: {sample['category']}")
        
        return validated_scenarios
    
    def process_scenarios_file(self, json_filepath: str, batch_size: int = 100):
        """
        Process scenarios from JSON file and store in Pinecone + MongoDB Atlas
        
        Args:
            json_filepath: Path to JSON file with scenarios (from Phase 1)
            batch_size: Number of vectors to upsert to Pinecone at once
        """
        # Load and validate scenarios
        scenarios = self._load_and_validate_scenarios(json_filepath)
        
        print(f"\n📄 Processing scenarios...")
        
        total_chunks = 0
        all_vectors = []
        errors = []
        
        for idx, scenario in enumerate(scenarios, 1):
            try:
                # Chunk the scenario
                chunks = self.chunk_scenario(scenario)
                total_chunks += len(chunks)
                
                # Generate embeddings for each chunk
                for chunk in chunks:
                    # Generate embedding
                    embedding = self.generate_embedding(chunk['text'])
                    
                    # Prepare vector for Pinecone
                    vector = {
                        'id': chunk['chunk_id'],
                        'values': embedding.tolist(),
                        'metadata': {
                            'scenario_id': chunk['scenario_id'],
                            'title': chunk['metadata']['title'],
                            'category': chunk['metadata']['category'],
                            'severity': chunk['metadata']['severity'],
                            'source': chunk['metadata']['source'],
                            'text': chunk['text'][:1000]  # Store preview in metadata
                        }
                    }
                    all_vectors.append(vector)
                    
                    # Store full chunk in MongoDB Atlas
                    self.chunks_collection.update_one(
                        {'chunk_id': chunk['chunk_id']},
                        {
                            '$set': {
                                'chunk_id': chunk['chunk_id'],
                                'scenario_id': chunk['scenario_id'],
                                'chunk_index': chunk['chunk_index'],
                                'text': chunk['text'],
                                'metadata': chunk['metadata'],
                                'created_at': datetime.utcnow()
                            }
                        },
                        upsert=True
                    )
                
                # Store full scenario in MongoDB Atlas
                self.scenarios_collection.update_one(
                    {'scenario_id': scenario['scenario_id']},
                    {
                        '$set': {
                            **scenario,
                            'num_chunks': len(chunks),
                            'processed_at': datetime.utcnow()
                        }
                    },
                    upsert=True
                )
                
                # Progress update
                if idx % 50 == 0:
                    print(f"  ✓ Processed {idx}/{len(scenarios)} scenarios | {total_chunks} chunks")
            
            except Exception as e:
                error_msg = f"Scenario {idx} ({scenario.get('scenario_id', 'unknown')}): {str(e)}"
                errors.append(error_msg)
                print(f"  ⚠️  Error processing scenario {idx}: {str(e)}")
                continue
        
        # Upsert all vectors to Pinecone in batches
        if all_vectors:
            print(f"\n📤 Uploading {len(all_vectors)} vectors to Pinecone...")
            
            for i in range(0, len(all_vectors), batch_size):
                batch = all_vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)
                
                if (i + batch_size) % 500 == 0:
                    print(f"  ✓ Uploaded {min(i+batch_size, len(all_vectors))}/{len(all_vectors)} vectors")
        
        print(f"\n✅ Processing complete!")
        print(f"   Total scenarios: {len(scenarios)}")
        print(f"   Successfully processed: {len(scenarios) - len(errors)}")
        print(f"   Total chunks: {total_chunks}")
        if len(scenarios) > 0 and total_chunks > 0:
            print(f"   Average chunks per scenario: {total_chunks/len(scenarios):.1f}")
        
        if errors:
            print(f"\n⚠️  Encountered {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more")
        
        # Get final stats
        stats = self.index.describe_index_stats()
        print(f"\n📊 Pinecone Index Stats:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
        
        mongo_scenario_count = self.scenarios_collection.count_documents({})
        mongo_chunk_count = self.chunks_collection.count_documents({})
        print(f"\n📊 MongoDB Atlas Stats:")
        print(f"   Scenarios: {mongo_scenario_count}")
        print(f"   Chunks: {mongo_chunk_count}")
    
    def test_search(self, query: str, top_k: int = 5):
        """Test semantic search with a query"""
        print(f"\n🔍 Testing search: '{query}'")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"\n📋 Top {top_k} results:")
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Score: {match.score:.4f}")
            print(f"   Title: {match.metadata.get('title', 'N/A')}")
            print(f"   Category: {match.metadata.get('category', 'N/A')}")
            print(f"   Source: {match.metadata.get('source', 'N/A')[:50]}")
            print(f"   Preview: {match.metadata.get('text', '')[:150]}...")
        
        return results


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate collected scenarios with Pinecone + MongoDB Atlas')
    parser.add_argument(
        '--input',
        type=str,
        default='./data/all_authoritative_scenarios.json',
        help='Path to scenarios JSON file from Phase 1'
    )
    parser.add_argument(
        '--test-queries',
        action='store_true',
        help='Run test queries after processing'
    )
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = PineconeIntegrator()
    
    # Process scenarios
    integrator.process_scenarios_file(args.input)
    
    # Test queries
    if args.test_queries:
        print("\n" + "="*70)
        print(" "*20 + "RUNNING TEST QUERIES")
        print("="*70)
        
        test_queries = [
            "severe bleeding wound",
            "second degree burn treatment",
            "CPR adult steps",
            "broken bone fracture",
            "heart attack symptoms"
        ]
        
        for query in test_queries:
            integrator.test_search(query, top_k=3)
            print()
    
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE")
    print("   Next: Use the RAG assistant for queries")
    print("="*70)


if __name__ == "__main__":
    main()