import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

load_dotenv()

def check_environment():
    
    required_vars = {
       'GROQ_API_KEY': 'API key for accessing Groq LLM inference service',
        'PINECONE_API_KEY': 'Pinecone vector database',
        'MONGODB_URI': 'MongoDB connection string'
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            logger.info(f"{var:20s}: {masked} ({description})")
        else:
            logger.info(f" {var:20s}: MISSING ({description})")
            missing.append(var)
    
    if missing:
        logger.info("\n Missing environment variables!")
        logger.info("   Create a .env file with:")
        for var in missing:
            logger.info(f"   {var}=your-{var.lower().replace('_', '-')}-here")
        return False
    
    logger.info("\nAll environment variables configured")
    return True


def check_dependencies():
    #Check if required packages are installed
    logger.info(" "*20 + "DEPENDENCY CHECK")
    
    required_packages = [
        'transformers',
        'torch',
        'pinecone',
        'pymongo',
        'groq',
        'requests',
        'pdfplumber'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"{package}")
        except ImportError:
            logger.info(f"{package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        logger.info(f"\nMissing packages: {', '.join(missing)}")
        logger.info("   Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("\nAll dependencies installed")
    return True


def check_data_directory():
    #Check and create data directory"""
    data_dir = Path("./data")
    if not data_dir.exists():
        logger.info(f"\nCreating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for Red Cross PDF
    pdf_path = data_dir / "Comprehensive_Guide_for_FirstAidCPR_en.pdf"
    if not pdf_path.exists():
        logger.info(f"\nRed Cross PDF not found: {pdf_path}")
        logger.info("   Download from: https://cdn.redcross.ca/prodmedia/crc/documents/Comprehensive_Guide_for_FirstAidCPR_en.pdf")
        logger.info("   Or collection will skip Red Cross source")
    else:
        logger.info(f"\nRed Cross PDF found: {pdf_path}")
    
    return data_dir


def data_collection(sources: list = None, fast_mode: bool = False):
    #Data collection
    logger.info(" "*15 + "PHASE 1: DATA COLLECTION")
    logger.info("="*70)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    from collectors.merge_pipeline import MasterDataPipeline
    
    pipeline = MasterDataPipeline(data_dir="./data")
    
    if fast_mode and not sources:
        sources = ['redcross']
        logger.info("\nFast Mode: Collecting from Red Cross only")
    elif sources is None:
        sources = ['redcross', 'mayo', 'cleveland', 'healthline', 'cdc', 'nhs', 'webmd']
        logger.info("\nFull Mode: Collecting from 7 sources")
    else:
        logger.info(f"\n Custom Mode: Collecting from {', '.join(sources)}")
    
    scenarios = pipeline.collect_all(sources=sources)
    
    pipeline.deduplicate()
    
    output_file = "fast_scenarios.json" if fast_mode else "checkpoint_phase1.json"
    pipeline.save_merged(output_file)
    
    pipeline.print_summary()
    
    logger.info("\nPHASE 1 COMPLETE")
    logger.info(f"   Output: ./data/{output_file}")
    logger.info(f"   Scenarios: {len(pipeline.all_scenarios)}")
    
    return f"./data/{output_file}"


def new_sources():
    logger.info(" "*10 + "PHASE 1b: COLLECT FROM NEW SOURCES")
    
    from collectors.new_sources_collector import NewSourcesCollector
    
    collector = NewSourcesCollector(data_dir="./data")
    results = collector.collect_all_new_sources()
    
    logger.info(f"   Total scenarios: {sum(len(s) for s in results.values())}")


def phase2_pinecone_integration(scenarios_file: str = None):
   
    logger.info("\n" + "="*70)
    logger.info(" "*15 + "PHASE 2: PINECONE INTEGRATION")
    logger.info("="*70)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if scenarios_file is None:
        data_dir = Path("./data")
        possible_files = [
            data_dir / "all_authoritative_scenarios.json",
            data_dir / "fast_scenarios.json",
            data_dir / "checkpoint_phase1.json"
        ]
        
        for filepath in possible_files:
            if filepath.exists():
                scenarios_file = str(filepath)
                break
        
        if scenarios_file is None:
            logger.info("[ERROR] No scenarios file found!")
            logger.info("   Expected locations:")
            for f in possible_files:
                logger.info(f"   - {f}")
            logger.info("\n   Run Phase 1 first: python master_pipeline.py --phase1")
            return False
    
    logger.info(f"\n Input file: {scenarios_file}")
    
    # Import and run
    from scripts.pinecone import PineconeIntegrator
    
    integrator = PineconeIntegrator()
    integrator.process_scenarios_file(scenarios_file)
    
    # Test search
    logger.info("\n Running test searches...")
    test_queries = [
        "severe bleeding",
        "burn treatment",
        "CPR steps"
    ]
    
    for query in test_queries:
        results = integrator.test_search(query, top_k=2)
        print()
    
    logger.info("\n[SUCCESS] PHASE 2 COMPLETE")
    logger.info("   Pinecone index populated")
    logger.info("   MongoDB collections updated")
    logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True


def phase3_rag_assistant(mode: str = 'demo'):
    
    logger.info("RAG ASSISTANT")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import and initialize
    from RAG.rag import FirstAidRAGAssistant
    
    assistant = FirstAidRAGAssistant()
    
    if mode == 'interactive':
        assistant.interactive_mode()
    
    elif mode == 'test':
        test_queries = [
            "What should I do for severe bleeding?",
            "How to treat a second-degree burn?",
            "CPR steps for adults",
            "Someone is choking, what do I do?",
            "Treating a sprained ankle"
        ]
        assistant.batch_test(test_queries)
    
    elif mode == 'demo':
        logger.info("\n Running demo queries...\n")
        demo_queries = [
            "What should I do if someone is bleeding heavily from a deep cut?",
            "How do I treat a second-degree burn?",
            "What are the steps for CPR on an adult who isn't breathing?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
          
            logger.info(f"Demo Query {i}/{len(demo_queries)}")
          
            
            result = assistant.answer_query(query, verbose=False)
            
            logger.info(f"\n Question: {query}")
            logger.info(f"\n Response:\n{result['response'][:500]}...")
            logger.info(f"\n Sources: {len(result['sources'])} | Confidence: {result['confidence']}")
    
    logger.info("\n[SUCCESS] PHASE 3 COMPLETE")
    logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def show_status():
    
    logger.info("SYSTEM STATUS")
    # Check Phase 1
    logger.info("\n Phase 1 - Data Collection:")
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
                logger.info(f"   {filename}: {len(scenarios)} scenarios")
                phase1_complete = True
            except Exception as e:
                logger.info(f"  [ERROR] {filename}: {str(e)}")
        else:
            logger.info(f"  [ERROR] {filename}: Not found")
    
    if not phase1_complete:
        logger.info("  Phase 1 not complete - run data collection first")
    
    logger.info("\n Phase 2 - Pinecone Integration:")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("first-aid-assistant")
        stats = index.describe_index_stats()
        logger.info(f"   Pinecone vectors: {stats.total_vector_count}")
        phase2_complete = stats.total_vector_count > 0
    except Exception as e:
        logger.info(f"   Pinecone: Not connected - {str(e)[:50]}")
        phase2_complete = False
    
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client['first_aid_db']
        scenario_count = db.scenarios.count_documents({})
        chunk_count = db.chunks.count_documents({})
        logger.info(f"  MongoDB scenarios: {scenario_count}")
        logger.info(f"  MongoDB chunks: {chunk_count}")
    except Exception as e:
        logger.info(f"  MongoDB: Not connected - {str(e)[:50]}")
    
    if not phase2_complete:
        logger.info("  Phase 2 not complete - run Pinecone integration")
    
    logger.info("\nPhase 3 - RAG Assistant:")
    if phase1_complete and phase2_complete:
        logger.info("  Ready to use")
        logger.info("  Run: python master_pipeline.py --interactive")
    else:
        logger.info(" Not ready - complete Phase 1 and 2 first")
    
    if not phase1_complete:
        logger.info("  Next step: Run Phase 1 (data collection)")
        logger.info("  Command: python master_pipeline.py --phase1")
    elif not phase2_complete:
        logger.info("  Next step: Run Phase 2 (Pinecone integration)")
        logger.info("  Command: python master_pipeline.py --phase2")
    else:
        logger.info("  System ready!")
        logger.info("  Command: python master_pipeline.py --interactive")


def main():
    parser = argparse.ArgumentParser(
        description='First Aid RAG System - Master Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check system status
  python master_pipeline.py --status
  
  # Run complete setup (all phases)
  python master_pipeline.py --full
  
  # Run individual phases
  python master_pipeline.py --phase1          # Existing sources (Red Cross, Mayo, etc.)
  python master_pipeline.py --collect-new     # 10 NEW sources
  python master_pipeline.py --phase2          # Pinecone integration
  python master_pipeline.py --phase3          # RAG assistant demo
  
  # Fast setup (Red Cross only)
  python master_pipeline.py --phase1 --fast
  
  # Interactive mode
  python master_pipeline.py --interactive
  
  # External tools (separate scripts)
  python merge_scenarios.py                   # Merge all checkpoints
  python -m collectors.augmentation           # Augment scenarios
        """
    )
    
    # Phase selection
    parser.add_argument('--full', action='store_true',
                       help='Run all 3 phases (complete setup)')
    parser.add_argument('--phase1', action='store_true',
                       help='Phase 1: Existing sources (RedCross, Mayo, Cleveland, Healthline, CDC, NHS, WebMD)')
    parser.add_argument('--collect-new', action='store_true',
                       help='Phase 1b: Collect from 10 NEW sources')
    parser.add_argument('--phase2', action='store_true',
                       help='Phase 2: Pinecone integration')
    parser.add_argument('--phase3', action='store_true',
                       help='Phase 3: RAG assistant demo')
    
    # Phase 1 options
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: Red Cross only (Phase 1)')
    parser.add_argument('--sources', type=str,
                       help='Comma-separated sources: redcross,mayo,cleveland,healthline,cdc,nhs,webmd')
    
    # Phase 2 options
    parser.add_argument('--scenarios-file', type=str,
                       help='Path to scenarios JSON file for Phase 2')
    
    # Phase 3 options
    parser.add_argument('--interactive', action='store_true',
                       help='Run RAG assistant in interactive mode')
    parser.add_argument('--test', action='store_true',
                       help='Run batch test queries')
    
    parser.add_argument('--status', action='store_true',
                       help='Show system status')
    parser.add_argument('--check', action='store_true',
                       help='Check environment and dependencies')
    
    args = parser.parse_args()
    
    logger.info("FIRST AID RAG SYSTEM")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.status:
            show_status()
            return
        
        if args.check or args.full or args.phase1 or args.phase2 or args.phase3 or args.interactive or args.collect_new:
            if not check_environment():
                logger.info("\n Environment check failed. Fix issues and try again.")
                return
            
            if not check_dependencies():
                logger.info("\n Dependency check failed. Install missing packages.")
                return
            
            check_data_directory()
        
        if args.full:
            logger.info("\nComplete pipeline...")
            print()
            
            scenarios_file = data_collection(
                sources=args.sources.split(',') if args.sources else None,
                fast_mode=args.fast
            )
            
            logger.info("\n Data Collectioncomplete. Continue to Phase 2? (yes/no)")
            if input().lower() not in ['yes', 'y']:
                logger.info("Stopping after Phase 1")
                return
            
            if not phase2_pinecone_integration(scenarios_file):
                logger.info("[ERROR] Phase 2 failed")
                return
            
            logger.info("\n  Pinecone Integration complete. Continue to Phase 3? (yes/no)")
            if input().lower() not in ['yes', 'y']:
                logger.info("Stopping after Phase 2")
                return
            
            phase3_rag_assistant(mode='demo')
            
            logger.info("\n FULL PIPELINE COMPLETE!")
            logger.info("   Run with --interactive to use the assistant")
        
        elif args.phase1:
            sources = args.sources.split(',') if args.sources else None
            data_collection(sources=sources, fast_mode=args.fast)
        
        elif args.collect_new:
            new_sources()
        
        elif args.phase2:
            phase2_pinecone_integration(args.scenarios_file)
        
        elif args.phase3:
            mode = 'test' if args.test else 'demo'
            phase3_rag_assistant(mode=mode)
        
        elif args.interactive:
            phase3_rag_assistant(mode='interactive')
        
        elif args.test:
            phase3_rag_assistant(mode='test')
        
        else:
            parser.print_help()
            print()
            show_status()
    
    except KeyboardInterrupt:
        logger.info("\n\n Interrupted by user")
        logger.info("Progress has been saved and can be resumed")
    except Exception as e:
        logger.info(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    


if __name__ == "__main__":
    main()