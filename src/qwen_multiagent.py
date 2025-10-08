
"""
Refactored Multi-Agent API Recommendation System

This module provides a clean, modular interface for the multi-agent API recommendation system.
The original functionality has been refactored into separate services for better maintainability.

Key Components:
- Configuration management for all parameters and paths
- RAG service for retrieval-augmented generation
- API recommendation service using multi-agent approach  
- Evaluation service for metrics calculation
- Main orchestrator for coordinating the entire pipeline

Usage:
    For full pipeline: Use APIRecommendationOrchestrator.run_full_pipeline()
    For legacy compatibility: Use the wrapper functions at the bottom of this file
"""

import os
import json
from typing import List, Dict, Any

# Import refactored services
from .config import Config
from .main_orchestrator import APIRecommendationOrchestrator
from .utils import process_json_response, count_hallucinated_apis

# Legacy imports for backward compatibility
from .main_orchestrator import (
    rag_baseline,
    get_topn_mashup_api, 
    call_with_messages_multi_agent
)


# Legacy function wrapper - maintained for backward compatibility
# The actual implementation has been moved to main_orchestrator.py

# Legacy function wrapper - maintained for backward compatibility
# The actual implementation has been moved to main_orchestrator.py

# Legacy function wrapper - maintained for backward compatibility
# The actual implementation has been moved to main_orchestrator.py

# Legacy utility functions - maintained for backward compatibility
# The actual implementations have been moved to utils.py

def process_json(text):
    """Legacy wrapper for process_json_response function."""
    return process_json_response(text)

def count_wrong_answer(api_list, apis):
    """Legacy wrapper for count_hallucinated_apis function."""
    return count_hallucinated_apis(api_list, apis)
    
def main():
    """
    Main function demonstrating the refactored API recommendation system.
    
    This function shows how to use the new modular architecture:
    1. Initialize configuration and orchestrator
    2. Run the complete pipeline or individual components
    3. Display evaluation results
    
    For production use, consider using the orchestrator directly for better control.
    """
    print("=== Refactored Multi-Agent API Recommendation System ===")
    print("Starting API recommendation pipeline with new modular architecture...")
    
    try:
        # Initialize configuration (automatically loads from default paths)
        config = Config()
        
        # Create main orchestrator
        orchestrator = APIRecommendationOrchestrator(config)
        
        # Option 1: Run the complete pipeline
        print("\nRunning complete pipeline...")
        metrics = orchestrator.run_full_pipeline(
            use_rag_cache=True,    # Use cached RAG results if available
            use_llm_retry=True     # Use retry mechanism for better results
        )
        
        # Display comprehensive results
        print("\n=== FINAL RESULTS ===")
        orchestrator.evaluation_service.print_evaluation_results(metrics)
        
        # Option 2: Get system statistics
        print("\n=== SYSTEM STATISTICS ===")
        stats = orchestrator.get_statistics()
        print(f"Dataset sizes: {stats['dataset_stats']}")
        
        # Option 3: Run individual components (examples)
        print("\n=== COMPONENT EXAMPLES ===")
        
        # Example: Run only RAG retrieval
        # rag_results = orchestrator.run_rag_only()
        # print(f"RAG retrieved {len(rag_results)} result sets")
        
        # Example: Evaluate pre-computed recommendations
        # metrics = orchestrator.run_evaluation_only("path/to/recommendations.json")
        
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        print("Please check your configuration and data paths.")
        return False
    
    print("\nPipeline completed successfully!")
    return True


def demo_new_api():
    """
    Demonstration of the new API for individual service usage.
    Shows how to use each service independently.
    """
    print("=== New API Demonstration ===")
    
    config = Config()
    
    # Example 1: Using RAG service independently
    from rag_service import RAGService
    rag_service = RAGService(config)
    print("RAG service initialized")
    
    # Example 2: Using API recommendation service independently  
    from api_recommendation_service import APIRecommendationService
    rec_service = APIRecommendationService(config)
    print("Recommendation service initialized")
    
    # Example 3: Using evaluation service independently
    from evaluation_service import EvaluationService
    eval_service = EvaluationService(config)
    print("Evaluation service initialized")
    
    print("All services can be used independently for maximum flexibility!")


if __name__ == "__main__":
    # Run the main pipeline
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("MIGRATION NOTES:")
        print("="*60)
        print("✅ Code has been successfully refactored into modular components")
        print("✅ All original functionality is preserved through legacy wrappers")
        print("✅ New architecture provides better maintainability and testing")
        print("✅ Configuration is now centralized and easily modifiable")
        print("✅ Each component can be used independently")
        print("\nFor advanced usage, see the demo_new_api() function")
        print("="*60)
    else:
        print("Pipeline execution failed. Please check the error messages above.")
