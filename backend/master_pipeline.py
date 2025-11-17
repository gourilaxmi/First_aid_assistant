import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_environment():
    """Check if all required environment variables are set"""
    print("\n" + "="*70)
    print(" "*20 + "ENVIRONMENT CHECK")
    print("="*70)
    
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
            print(f"{var:20s}: {masked} ({description})")
        else:
            print(f" {var:20s}: MISSING ({description})")
            missing.append(var)
    
    if missing:
        print("\n Missing environment variables!")
        print("   Create a .env file with:")
        for var in missing:
            print(f"   {var}=your-{var.lower().replace('_', '-')}-here")
        return False
    
    print("\nAll environment variables configured")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*70)
    print(" "*20 + "DEPENDENCY CHECK")
    print("="*70)
    
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
            print(f"{package}")
        except ImportError:
            print(f"{package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        return False
    
    print("\nAll dependencies installed")
    return True


def check_data_directory():
    """Check and create data directory"""
    data_dir = Path("./data")
    if not data_dir.exists():
        print(f"\nCreating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for Red Cross PDF
    pdf_path = data_dir / "Comprehensive_Guide_for_FirstAidCPR_en.pdf"
    if not pdf_path.exists():
        print(f"\nRed Cross PDF not found: {pdf_path}")
        print("   Download from: https://cdn.redcross.ca/prodmedia/crc/documents/Comprehensive_Guide_for_FirstAidCPR_en.pdf")
        print("   Or collection will skip Red Cross source")
    else:
        print(f"\nRed Cross PDF found: {pdf_path}")
    
    return data_dir


def phase1_data_collection(sources: list = None, fast_mode: bool = False):
    """
    Phase 1: Collect data from existing sources
    
    Args:
        sources: List of sources to collect from
        fast_mode: If True, only collect from Red Cross
    """
    print("\n" + "="*70)
    print(" "*15 + "PHASE 1: DATA COLLECTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import here to avoid issues if not installed
    from collectors.merge_pipeline import MasterDataPipeline
    
    pipeline = MasterDataPipeline(data_dir="./data")
    
    # Determine sources (removed meddialog, stjohn)
    if fast_mode and not sources:
        sources = ['redcross']
        print("\nFast Mode: Collecting from Red Cross only")
    elif sources is None:
        sources = ['redcross', 'mayo', 'cleveland', 'healthline', 'cdc', 'nhs', 'webmd']
        print("\nFull Mode: Collecting from 7 sources")
    else:
        print(f"\n Custom Mode: Collecting from {', '.join(sources)}")
    
    # Collect
    scenarios = pipeline.collect_all(sources=sources)
    
    # Deduplicate
    pipeline.deduplicate()
    
    # Save
    output_file = "fast_scenarios.json" if fast_mode else "checkpoint_phase1.json"
    pipeline.save_merged(output_file)
    
    # Summary
    pipeline.print_summary()
    
    print(f"\nPHASE 1 COMPLETE")
    print(f"   Output: ./data/{output_file}")
    print(f"   Scenarios: {len(pipeline.all_scenarios)}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return f"./data/{output_file}"


def phase1_new_sources():
    """Phase 1b: Collect from 10 NEW sources"""
    print("\n" + "="*70)
    print(" "*10 + "PHASE 1b: COLLECT FROM NEW SOURCES")
    print("="*70)
    
    from collectors.new_sources_collector import NewSourcesCollector
    
    collector = NewSourcesCollector(data_dir="./data")
    results = collector.collect_all_new_sources()
    
    print("\nNEW SOURCES COLLECTION COMPLETE")
    print(f"   Total scenarios: {sum(len(s) for s in results.values())}")
    print(f"\nNext steps:")
    print(f"   1. Merge checkpoints: python merge_scenarios.py")
    print(f"   2. Then augment: python -m collectors.augmentation")


def phase2_pinecone_integration(scenarios_file: str = None):
    """
    Phase 2: Process scenarios and upload to Pinecone + MongoDB
    
    Args:
        scenarios_file: Path to scenarios JSON file from Phase 1
    """
    print("\n" + "="*70)
    print(" "*15 + "PHASE 2: PINECONE INTEGRATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find scenarios file if not provided
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
            print("❌ No scenarios file found!")
            print("   Expected locations:")
            for f in possible_files:
                print(f"   - {f}")
            print("\n   Run Phase 1 first: python master_pipeline.py --phase1")
            return False
    
    print(f"\n Input file: {scenarios_file}")
    
    # Import and run
    from scripts.pinecone import PineconeIntegrator
    
    integrator = PineconeIntegrator()
    integrator.process_scenarios_file(scenarios_file)
    
    # Test search
    print("\n🔍 Running test searches...")
    test_queries = [
        "severe bleeding",
        "burn treatment",
        "CPR steps"
    ]
    
    for query in test_queries:
        results = integrator.test_search(query, top_k=2)
        print()
    
    print(f"\n✅ PHASE 2 COMPLETE")
    print(f"   Pinecone index populated")
    print(f"   MongoDB collections updated")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True


def phase3_rag_assistant(mode: str = 'demo'):
    """
    Phase 3: Run RAG Assistant
    
    Args:
        mode: 'demo', 'interactive', or 'test'
    """
    print("\n" + "="*70)
    print(" "*15 + "PHASE 3: RAG ASSISTANT")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
        print("\n🎬 Running demo queries...\n")
        demo_queries = [
            "What should I do if someone is bleeding heavily from a deep cut?",
            "How do I treat a second-degree burn?",
            "What are the steps for CPR on an adult who isn't breathing?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'─'*70}")
            print(f"Demo Query {i}/{len(demo_queries)}")
            print(f"{'─'*70}")
            
            result = assistant.answer_query(query, verbose=False)
            
            print(f"\n❓ Question: {query}")
            print(f"\n💡 Response:\n{result['response'][:500]}...")
            print(f"\n📚 Sources: {len(result['sources'])} | Confidence: {result['confidence']}")
    
    print(f"\n✅ PHASE 3 COMPLETE")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def show_status():
    """Show current system status"""
    print("\n" + "="*70)
    print(" "*25 + "SYSTEM STATUS")
    print("="*70)
    
    # Check Phase 1
    print("\n📊 Phase 1 - Data Collection:")
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
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                scenarios = data.get('scenarios', data if isinstance(data, list) else [])
            print(f"   {filename}: {len(scenarios)} scenarios")
            phase1_complete = True
        else:
            print(f"  ❌ {filename}: Not found")
    
    if not phase1_complete:
        print("  Phase 1 not complete - run data collection first")
    
    # Check Phase 2
    print("\n📊 Phase 2 - Pinecone Integration:")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("first-aid-assistant")
        stats = index.describe_index_stats()
        print(f"   Pinecone vectors: {stats.total_vector_count}")
        phase2_complete = stats.total_vector_count > 0
    except Exception as e:
        print(f"   Pinecone: Not connected - {str(e)[:50]}")
        phase2_complete = False
    
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client['first_aid_db']
        scenario_count = db.scenarios.count_documents({})
        chunk_count = db.chunks.count_documents({})
        print(f"  MongoDB scenarios: {scenario_count}")
        print(f"  MongoDB chunks: {chunk_count}")
    except Exception as e:
        print(f"  MongoDB: Not connected - {str(e)[:50]}")
    
    if not phase2_complete:
        print("  Phase 2 not complete - run Pinecone integration")
    
    # Check Phase 3
    print("\nPhase 3 - RAG Assistant:")
    if phase1_complete and phase2_complete:
        print("  Ready to use")
        print("  Run: python master_pipeline.py --interactive")
    else:
        print(" Not ready - complete Phase 1 and 2 first")
    
    print("\n" + "="*70)
    
    # Summary
    print("\nSummary:")
    if not phase1_complete:
        print("  Next step: Run Phase 1 (data collection)")
        print("  Command: python master_pipeline.py --phase1")
    elif not phase2_complete:
        print("  Next step: Run Phase 2 (Pinecone integration)")
        print("  Command: python master_pipeline.py --phase2")
    else:
        print("  System ready!")
        print("  Command: python master_pipeline.py --interactive")


def main():
    """Main entry point"""
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
    
    # Utility
    parser.add_argument('--status', action='store_true',
                       help='Show system status')
    parser.add_argument('--check', action='store_true',
                       help='Check environment and dependencies')
    
    args = parser.parse_args()
    
    # Show header
    print("\n" + "="*70)
    print("🏥" + " "*10 + "FIRST AID RAG SYSTEM - MASTER PIPELINE" + " "*10 + "🏥")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Status check
        if args.status:
            show_status()
            return
        
        # Environment check
        if args.check or args.full or args.phase1 or args.phase2 or args.phase3 or args.interactive or args.collect_new:
            if not check_environment():
                print("\n❌ Environment check failed. Fix issues and try again.")
                return
            
            if not check_dependencies():
                print("\n❌ Dependency check failed. Install missing packages.")
                return
            
            check_data_directory()
        
        # Full pipeline
        if args.full:
            print("\nRunning FULL PIPELINE (all 3 phases)...")
            print()
            
            # Phase 1
            scenarios_file = phase1_data_collection(
                sources=args.sources.split(',') if args.sources else None,
                fast_mode=args.fast
            )
            
            print("\n⏸️  Phase 1 complete. Continue to Phase 2? (yes/no)")
            if input().lower() not in ['yes', 'y']:
                print("Stopping after Phase 1")
                return
            
            # Phase 2
            if not phase2_pinecone_integration(scenarios_file):
                print("❌ Phase 2 failed")
                return
            
            print("\n⏸️  Phase 2 complete. Continue to Phase 3? (yes/no)")
            if input().lower() not in ['yes', 'y']:
                print("Stopping after Phase 2")
                return
            
            # Phase 3
            phase3_rag_assistant(mode='demo')
            
            print("\n FULL PIPELINE COMPLETE!")
            print("   Run with --interactive to use the assistant")
        
        # Individual phases
        elif args.phase1:
            sources = args.sources.split(',') if args.sources else None
            phase1_data_collection(sources=sources, fast_mode=args.fast)
        
        elif args.collect_new:
            phase1_new_sources()
        
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
            # No arguments - show help and status
            parser.print_help()
            print()
            show_status()
    
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        print("   Progress has been saved and can be resumed")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()