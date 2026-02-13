"""
Quick status check for First Aid RAG System
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print(" " * 20 + "FIRST AID RAG SYSTEM STATUS")
print("=" * 70)

# Check Phase 1 - Data Files
print("\nPhase 1 - Data Collection:")
data_dir = Path("./data")
data_files = [
    "all_authoritative_scenarios.json",
    "fast_scenarios.json", 
    "checkpoint_phase1.json"
]

phase1_complete = False
for filename in data_files:
    filepath = data_dir / filename
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both dict and list formats
                if isinstance(data, dict):
                    scenarios = data.get('scenarios', [])
                else:
                    scenarios = data
                print(f"  ✓ {filename}: {len(scenarios)} scenarios")
                phase1_complete = True
        except Exception as e:
            print(f"  ✗ {filename}: Error reading - {e}")
    else:
        print(f"  - {filename}: Not found")

if not phase1_complete:
    print("  [!] No data files found - need to run Phase 1 data collection")

# Check Phase 2 - Pinecone
print("\nPhase 2 - Vector Database:")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("first-aid-assistant")
    stats = index.describe_index_stats()
    print(f"  ✓ Pinecone vectors: {stats.total_vector_count}")
    phase2_pinecone = stats.total_vector_count > 0
except Exception as e:
    print(f"  ✗ Pinecone: Error - {str(e)[:50]}")
    phase2_pinecone = False

# Check Phase 2 - MongoDB
try:
    from pymongo import MongoClient
    client = MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['first_aid_db']
    scenario_count = db.scenarios.count_documents({})
    chunk_count = db.chunks.count_documents({})
    print(f"  ✓ MongoDB scenarios: {scenario_count}")
    print(f"  ✓ MongoDB chunks: {chunk_count}")
    phase2_mongodb = chunk_count > 0
except Exception as e:
    print(f"  ✗ MongoDB: Error - {str(e)[:50]}")
    phase2_mongodb = False

# Summary
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)

if phase1_complete and phase2_pinecone and phase2_mongodb:
    print("✓ System is fully configured and ready!")
    print("  Run your FastAPI backend to use the RAG assistant")
elif phase1_complete and phase2_pinecone and not phase2_mongodb:
    print("⚠ MongoDB is empty but Pinecone has data")
    print("  NEXT STEP: Run Phase 2 to sync MongoDB")
    print("  Command: python master_pipeline.py --phase2")
elif phase1_complete:
    print("⚠ Data collected but not uploaded to databases")
    print("  NEXT STEP: Run Phase 2")
    print("  Command: python master_pipeline.py --phase2")
else:
    print("⚠ No data collected yet")
    print("  NEXT STEP: Run Phase 1 to collect data")
    print("  Command: python master_pipeline.py --phase1")

print("=" * 70)
