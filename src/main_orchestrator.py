"""
Main orchestrator for the multi-agent API recommendation system.
Coordinates RAG retrieval, LLM-based recommendation, and evaluation processes.
"""

import json
from typing import List, Dict, Any, Optional

from .config import Config
from .rag_service import RAGService
from .api_recommendation_service import APIRecommendationService
from .evaluation_service import EvaluationService, EvaluationMetrics
from .utils import (
    load_json_file, 
    save_json_file, 
    prepare_answer_list,
    count_hallucinated_apis
)


class APIRecommendationOrchestrator:
    """
    Main orchestrator class that coordinates the entire API recommendation pipeline.
    Manages data loading, RAG retrieval, LLM recommendation, and evaluation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the orchestrator with configuration and services.
        
        Args:
            config (Optional[Config]): Configuration object, creates default if None
        """
        self.config = config or Config()
        
        # Initialize services
        self.rag_service = RAGService(self.config)
        self.recommendation_service = APIRecommendationService(self.config)
        self.evaluation_service = EvaluationService(self.config)
        
        # Data storage
        self.mashups = []
        self.apis = []
        self.train_questions = []
        self.test_questions = []
        
        print("API Recommendation Orchestrator initialized")
    
    def load_data(self) -> bool:
        """
        Load all required datasets from configured paths.
        
        Returns:
            bool: True if all data loaded successfully, False otherwise
        """
        print("Loading datasets...")
        
        # Validate paths first
        if not self.config.validate_paths():
            print("Path validation failed")
            return False
        
        # Load mashup data
        mashup_data = load_json_file(self.config.paths.mashup_data_path)
        if mashup_data is None:
            print("Failed to load mashup data")
            return False
        self.mashups = mashup_data
        print(f"Loaded {len(self.mashups)} mashups")
        
        # Load API data
        api_data = load_json_file(self.config.paths.api_data_path)
        if api_data is None:
            print("Failed to load API data")
            return False
        self.apis = api_data
        print(f"Loaded {len(self.apis)} APIs")
        
        # Load training data
        train_data = load_json_file(self.config.paths.train_data_path)
        if train_data is None:
            print("Failed to load training data")
            return False
        self.train_questions = train_data
        print(f"Loaded {len(self.train_questions)} training questions")
        
        # Load test data
        test_data = load_json_file(self.config.paths.test_data_path)
        test_data = test_data[:5]
        if test_data is None:
            print("Failed to load test data")
            return False
        self.test_questions = test_data
        print(f"Loaded {len(self.test_questions)} test questions")
        
        print("All datasets loaded successfully")
        return True
    
    def run_rag_retrieval(self, questions: List[Dict[str, Any]], 
                         use_cache: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Run RAG-based API retrieval for given questions.
        
        Args:
            questions (List[Dict[str, Any]]): Questions to process
            use_cache (bool): Whether to use cached results if available
            
        Returns:
            List[List[Dict[str, Any]]]: Retrieved APIs for each question
        """
        cache_file = self.config.paths.answer_cache_file
        
        # Try to load from cache first
        if use_cache and cache_file:
            cached_results = load_json_file(cache_file)
            if cached_results is not None:
                print(f"Loaded cached RAG results from {cache_file}")
                return cached_results
        
        # Run RAG pipeline
        print("Running RAG retrieval pipeline...")
        rag_results, total_chars = self.rag_service.run_rag_pipeline(
            self.train_questions,  # Use training data as knowledge base
            questions
        )
        
        # Save results to cache
        if cache_file:
            if save_json_file(rag_results, cache_file):
                print(f"Saved RAG results to {cache_file}")
            else:
                print(f"Failed to save RAG results to {cache_file}")
        
        print(f"RAG retrieval completed, processed {total_chars} characters")
        return rag_results
    
    def run_llm_recommendation(self, questions: List[Dict[str, Any]], 
                             candidate_api_sets: List[List[Dict[str, Any]]], 
                             use_retry: bool = True) -> List[List[str]]:
        """
        Run LLM-based API recommendation for given questions and candidates.
        
        Args:
            questions (List[Dict[str, Any]]): Questions to process
            candidate_api_sets (List[List[Dict[str, Any]]]): Candidate APIs for each question
            use_retry (bool): Whether to use retry mechanism for better results
            
        Returns:
            List[List[str]]: Final API recommendations for each question
        """
        print("Running LLM-based API recommendation...")
        
        def recommendation_function(question: Dict[str, Any], 
                                  candidates: List[Dict[str, Any]]) -> List[str]:
            """Helper function for generating recommendations."""
            # Limit candidates to configured maximum
            limited_candidates = candidates[:self.config.retrieval.final_api_limit]
            
            if use_retry:
                recommendations, attempt_count = (
                    self.recommendation_service.recommend_with_retry(
                        question, limited_candidates
                    )
                )
                if attempt_count > 1:
                    print(f"Required {attempt_count} attempts for question: {question.get('title', 'Unknown')}")
            else:
                response = self.recommendation_service.recommend_apis(
                    question, limited_candidates
                )
                recommendations = self.recommendation_service.process_recommendation_response(response)
            
            return recommendations
        
        # Use evaluation service for incremental processing with logging
        metrics = self.evaluation_service.run_incremental_evaluation(
            questions, recommendation_function, candidate_api_sets, self.apis
        )
        
        # Get final recommendations from logger
        final_recommendations = self.evaluation_service.logger.get_recommendations()
        
        print("LLM recommendation completed")
        return final_recommendations
    
    def evaluate_results(self, questions: List[Dict[str, Any]], 
                        recommendations: List[List[str]]) -> EvaluationMetrics:
        """
        Evaluate recommendation results against ground truth.
        
        Args:
            questions (List[Dict[str, Any]]): Original questions with ground truth
            recommendations (List[List[str]]): Generated recommendations
            
        Returns:
            EvaluationMetrics: Comprehensive evaluation metrics
        """
        print("Evaluating recommendation results...")
        
        metrics = self.evaluation_service.evaluate_recommendations(
            questions, recommendations, self.apis
        )
        
        self.evaluation_service.print_evaluation_results(metrics)
        return metrics
    
    def run_full_pipeline(self, use_rag_cache: bool = True, 
                         use_llm_retry: bool = True) -> EvaluationMetrics:
        """
        Run the complete API recommendation pipeline.
        
        Args:
            use_rag_cache (bool): Whether to use cached RAG results
            use_llm_retry (bool): Whether to use retry mechanism for LLM
            
        Returns:
            EvaluationMetrics: Final evaluation metrics
        """
        print("Starting full API recommendation pipeline...")
        
        # Step 1: Load all required data
        if not self.load_data():
            raise RuntimeError("Failed to load required datasets")
        
        # Step 2: Run RAG retrieval to get candidate APIs
        candidate_api_sets = self.run_rag_retrieval(
            self.test_questions, use_cache=use_rag_cache
        )
        
        # Step 3: Run LLM-based recommendation
        final_recommendations = self.run_llm_recommendation(
            self.test_questions, candidate_api_sets, use_retry=use_llm_retry
        )
        
        # Step 4: Evaluate results
        metrics = self.evaluate_results(self.test_questions, final_recommendations)
        
        print("Full pipeline completed successfully")
        return metrics
    
    def run_rag_only(self, questions: Optional[List[Dict[str, Any]]] = None) -> List[List[Dict[str, Any]]]:
        """
        Run only the RAG retrieval part of the pipeline.
        
        Args:
            questions (Optional[List[Dict[str, Any]]]): Questions to process, uses test_questions if None
            
        Returns:
            List[List[Dict[str, Any]]]: RAG retrieval results
        """
        if not self.load_data():
            raise RuntimeError("Failed to load required datasets")
        
        questions = questions or self.test_questions
        return self.run_rag_retrieval(questions, use_cache=False)
    
    def run_evaluation_only(self, recommendations_file: str) -> EvaluationMetrics:
        """
        Run only evaluation on pre-generated recommendations.
        
        Args:
            recommendations_file (str): Path to file containing recommendations
            
        Returns:
            EvaluationMetrics: Evaluation metrics
        """
        if not self.load_data():
            raise RuntimeError("Failed to load required datasets")
        
        # Load recommendations
        recommendations = load_json_file(recommendations_file)
        if recommendations is None:
            raise RuntimeError(f"Failed to load recommendations from {recommendations_file}")
        
        return self.evaluate_results(self.test_questions, recommendations)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics and configuration summary.
        
        Returns:
            Dict[str, Any]: Statistics and configuration information
        """
        stats = {
            "dataset_stats": {
                "mashups": len(self.mashups),
                "apis": len(self.apis),
                "train_questions": len(self.train_questions),
                "test_questions": len(self.test_questions)
            },
            "configuration": self.config.to_dict()
        }
        
        return stats


# Legacy wrapper functions for backward compatibility
def rag_baseline(mashups: List[Dict], questions: List[str], 
                mode: str = "simi", question_origin: List[Dict] = None) -> tuple:
    """
    Legacy wrapper for the original rag_baseline function.
    
    Args:
        mashups (List[Dict]): Mashup dataset
        questions (List[str]): Question strings (unused in new implementation)
        mode (str): Processing mode (unused in new implementation)  
        question_origin (List[Dict]): Original question objects
        
    Returns:
        tuple: (recommendations, total_characters)
    """
    if question_origin is None:
        question_origin = []
    
    config = Config()
    orchestrator = APIRecommendationOrchestrator(config)
    orchestrator.mashups = mashups
    
    # Run RAG pipeline
    rag_service = RAGService(config)
    results, total_chars = rag_service.run_rag_pipeline(mashups, question_origin)
    
    return results, total_chars


def get_topn_mashup_api(rerank_answers: List[List[str]], 
                       mashups: List[Dict[str, Any]], 
                       top_n: int = 50) -> List[List[Dict[str, Any]]]:
    """
    Legacy wrapper for extracting APIs from mashup results.
    
    Args:
        rerank_answers (List[List[str]]): Mashup titles for each query
        mashups (List[Dict[str, Any]]): Complete mashup dataset
        top_n (int): Maximum number of APIs to return (unused)
        
    Returns:
        List[List[Dict[str, Any]]]: API results for each query
    """
    config = Config()
    rag_service = RAGService(config)
    return rag_service.extract_apis_from_mashups(rerank_answers, mashups)


def call_with_messages_multi_agent(question: Dict[str, Any], 
                                 answer_apis: List[Dict[str, Any]], 
                                 function: List = None) -> Dict[str, Any]:
    """
    Legacy wrapper for multi-agent API recommendation.
    
    Args:
        question (Dict[str, Any]): Question object
        answer_apis (List[Dict[str, Any]]): Candidate APIs
        function (List): Additional functions (unused)
        
    Returns:
        Dict[str, Any]: Recommendation result
    """
    config = Config()
    service = APIRecommendationService(config)
    return service.recommend_apis(question, answer_apis, function)
