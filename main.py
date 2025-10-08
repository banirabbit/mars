#!/usr/bin/env python3
"""
Multi-Agent API Recommendation System - Main Entry Point

This is the main entry point for the refactored multi-agent API recommendation system.
The system uses a modular architecture with configuration loaded from config.yaml.

Key Features:
- YAML-based configuration management
- Modular service architecture  
- RAG-based API retrieval
- Multi-agent LLM recommendation
- Comprehensive evaluation metrics
- Resumable processing with logging

Usage:
    python main.py                    # Run full pipeline
    python main.py --demo             # Show API demonstration
    python main.py --config-check     # Validate configuration
    python main.py --rag-only         # Run only RAG retrieval
    python main.py --eval-only FILE   # Evaluate pre-computed results

Author: Multi-Agent API Recommendation Team
Version: 2.0 (Refactored)
"""

import sys
import os
import argparse
from pathlib import Path

# Set offline mode environment variables to avoid network requests
offline_env = {
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1", 
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS": "1",
}

for key, value in offline_env.items():
    os.environ[key] = value

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.main_orchestrator import APIRecommendationOrchestrator
from src.rag_service import RAGService
from src.api_recommendation_service import APIRecommendationService
from src.evaluation_service import EvaluationService


def main():
    """
    Main function demonstrating the refactored API recommendation system.
    
    This function shows how to use the new modular architecture:
    1. Initialize configuration from YAML file
    2. Create and configure the orchestrator
    3. Run the complete pipeline or individual components
    4. Display comprehensive evaluation results
    """
    print("=== Multi-Agent API Recommendation System v2.0 ===")
    print("Starting API recommendation pipeline with modular architecture...")
    
    try:
        # Initialize configuration from config.yaml
        print("\n📋 Loading configuration...")
        config = Config()
        
        # Validate configuration and paths
        if not config.validate_paths():
            print("❌ Configuration validation failed!")
            print("Please check your config.yaml file and ensure all required data files exist.")
            return False
        
        print("✅ Configuration loaded and validated successfully")
        print(f"📁 Project root: {config.get_project_root()}")
        
        # Create main orchestrator
        print("\n🚀 Initializing orchestrator...")
        orchestrator = APIRecommendationOrchestrator(config)
        
        # Display system statistics
        print("\n📊 System Statistics:")
        stats = orchestrator.get_statistics()
        dataset_stats = stats['dataset_stats']
        print(f"  • Mashups: {dataset_stats.get('mashups', 'Loading...')}")
        print(f"  • APIs: {dataset_stats.get('apis', 'Loading...')}")
        print(f"  • Training questions: {dataset_stats.get('train_questions', 'Loading...')}")
        print(f"  • Test questions: {dataset_stats.get('test_questions', 'Loading...')}")
        
        # Run the complete pipeline
        print("\n🔄 Running complete pipeline...")
        print("This includes:")
        print("  1. RAG-based API retrieval")
        print("  2. Multi-agent LLM recommendation")
        print("  3. Comprehensive evaluation")
        
        metrics = orchestrator.run_full_pipeline(
            use_rag_cache=True,    # Use cached RAG results if available
            use_llm_retry=True     # Use retry mechanism for better results
        )
        
        # Display comprehensive results
        print("\n🎯 FINAL EVALUATION RESULTS")
        print("=" * 50)
        orchestrator.evaluation_service.print_evaluation_results(metrics)
        
        # Save results summary
        results_summary = {
            'metrics': metrics.to_dict(),
            'config_summary': {
                'model_path': config.model.embed_model_path,
                'retrieval_k': config.retrieval.initial_k,
                'final_limit': config.retrieval.final_api_limit,
                'retry_attempts': config.evaluation.max_retry_attempts
            }
        }
        
        # Optional: Save detailed results
        results_file = config.get_project_root() / "results" / "latest_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline execution: {e}")
        print("Please check your configuration and data paths.")
        import traceback
        if config.system.debug:
            traceback.print_exc()
        return False
    
    print("\n✅ Pipeline completed successfully!")
    return True


def demo_new_api():
    """
    Demonstration of the new API for individual service usage.
    Shows how to use each service independently for advanced use cases.
    """
    print("=== 🔧 New API Demonstration ===")
    print("This demonstrates how to use individual services independently.")
    
    try:
        config = Config()
        print(f"✅ Configuration loaded from: {config.get_project_root() / 'config.yaml'}")
        
        # Example 1: Using RAG service independently
        print("\n📚 RAG Service Example:")
        rag_service = RAGService(config)
        print("  ✅ RAG service initialized")
        print("  • Can be used for: document retrieval, embedding, reranking")
        
        # Example 2: Using API recommendation service independently  
        print("\n🤖 API Recommendation Service Example:")
        rec_service = APIRecommendationService(config)
        print("  ✅ Recommendation service initialized")
        print("  • Can be used for: LLM-based recommendation, retry mechanisms")
        
        # Example 3: Using evaluation service independently
        print("\n📈 Evaluation Service Example:")
        eval_service = EvaluationService(config)
        print("  ✅ Evaluation service initialized")
        print("  • Can be used for: metrics calculation, result analysis")
        
        print("\n🎯 Key Benefits:")
        print("  • Each service can be used independently")
        print("  • Easy to test individual components")
        print("  • Flexible integration with other systems")
        print("  • Configuration-driven behavior")
        
    except Exception as e:
        print(f"❌ Error in API demonstration: {e}")
        return False
    
    print("✅ API demonstration completed!")
    return True


def check_configuration():
    """Check and display configuration details."""
    print("=== 🔍 Configuration Check ===")
    
    try:
        config = Config()
        
        print(f"📁 Project root: {config.get_project_root()}")
        print(f"📋 Configuration file: {config.get_project_root() / 'config.yaml'}")
        
        print("\n🤖 Model Configuration:")
        print(f"  • Embedding model: {config.model.embed_model_path}")
        print(f"  • Reranking model: {config.model.rerank_model_name}")
        print(f"  • API similarity model: {config.model.api_embed_model_name}")
        
        print("\n🔍 Retrieval Configuration:")
        print(f"  • Initial K: {config.retrieval.initial_k}")
        print(f"  • Rerank top N: {config.retrieval.rerank_top_n}")
        print(f"  • Final API limit: {config.retrieval.final_api_limit}")
        
        print("\n📂 File Paths:")
        print(f"  • Mashup data: {config.paths.mashup_data_path}")
        print(f"  • API data: {config.paths.api_data_path}")
        print(f"  • Train data: {config.paths.train_data_path}")
        print(f"  • Test data: {config.paths.test_data_path}")
        print(f"  • Log file: {config.paths.log_file}")
        
        # Validate paths
        is_valid = config.validate_paths()
        if is_valid:
            print("\n✅ All required files found!")
        else:
            print("\n⚠️  Some required files are missing (see warnings above)")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False


def run_rag_only():
    """Run only the RAG retrieval component."""
    print("=== 📚 RAG-Only Mode ===")
    
    try:
        config = Config()
        orchestrator = APIRecommendationOrchestrator(config)
        
        print("Running RAG retrieval pipeline...")
        results = orchestrator.run_rag_only()
        
        print(f"✅ RAG retrieval completed!")
        print(f"📊 Retrieved {len(results)} result sets")
        
        # Save results
        import json
        output_file = config.get_project_root() / "output" / "rag_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error in RAG-only mode: {e}")
        return False


def run_eval_only(recommendations_file: str):
    """Run only the evaluation on pre-computed recommendations."""
    print(f"=== 📈 Evaluation-Only Mode ===")
    print(f"Evaluating recommendations from: {recommendations_file}")
    
    try:
        config = Config()
        orchestrator = APIRecommendationOrchestrator(config)
        
        metrics = orchestrator.run_evaluation_only(recommendations_file)
        
        print("✅ Evaluation completed!")
        orchestrator.evaluation_service.print_evaluation_results(metrics)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in evaluation-only mode: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent API Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full pipeline
  python main.py --demo                   # Show API demonstration  
  python main.py --config-check           # Check configuration
  python main.py --rag-only               # Run only RAG retrieval
  python main.py --eval-only results.json # Evaluate pre-computed results
        """
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Show demonstration of new API usage'
    )
    
    parser.add_argument(
        '--config-check', 
        action='store_true',
        help='Check and display configuration details'
    )
    
    parser.add_argument(
        '--rag-only', 
        action='store_true',
        help='Run only RAG retrieval component'
    )
    
    parser.add_argument(
        '--eval-only', 
        type=str,
        metavar='FILE',
        help='Run only evaluation on pre-computed recommendations file'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    success = False
    
    if args.demo:
        success = demo_new_api()
    elif args.config_check:
        success = check_configuration()
    elif args.rag_only:
        success = run_rag_only()
    elif args.eval_only:
        success = run_eval_only(args.eval_only)
    else:
        # Run main pipeline
        success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎉 SYSTEM INFORMATION")
        print("="*60)
        print("✅ Code has been successfully refactored into modular components")
        print("✅ Configuration is managed through config.yaml file")
        print("✅ All paths are relative to project root")
        print("✅ Each component can be used independently")
        print("✅ Full backward compatibility maintained")
        print("\n📚 For more information, see:")
        print("  • config.yaml - Configuration settings")
        print("  • src/README.md - Architecture documentation")
        print("  • Use --demo flag to see API examples")
        print("="*60)
        sys.exit(0)
    else:
        print("\n❌ Execution failed. Please check the error messages above.")
        sys.exit(1)
