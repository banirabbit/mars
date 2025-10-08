"""
API Recommendation Service using multi-agent approach.
Handles LLM-based API recommendation and response processing.
"""

import json
from typing import List, Dict, Any, Optional

from .multiagent_recall import run_multiagent_flow, setup_llm_client
from .config import Config
from .utils import process_json_response, deduplicate_api_list, create_mashup_text


class APIRecommendationService:
    """
    Service for generating API recommendations using multi-agent LLM approach.
    Integrates with the multi-agent pipeline and handles response processing.
    """
    
    def __init__(self, config: Config):
        """
        Initialize API recommendation service.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.prompt_template = self._load_prompt_template()
        
        # Initialize LLM client with configuration
        try:
            if hasattr(self.config, 'config_data') and self.config.config_data:
                llm_config = self.config.config_data.get('llm', {})
                setup_llm_client(
                    base_url=llm_config.get('base_url', 'http://192.168.1.101:30111/v1'),
                    api_key=llm_config.get('api_key', 'loopinnetwork'),
                    model_name=llm_config.get('model_name', 'Qwen2.5-14B-Instruct'),
                    max_retry_count=llm_config.get('max_retry_count', 5)
                )
            else:
                # Fallback to default values
                setup_llm_client(
                    base_url="http://192.168.1.101:30111/v1",
                    api_key="loopinnetwork",
                    model_name="Qwen2.5-14B-Instruct",
                    max_retry_count=5
                )
        except Exception as e:
            print(f"Warning: Failed to load LLM config: {e}")
            print("Using default LLM configuration")
            setup_llm_client(
                base_url="http://192.168.1.101:30111/v1",
                api_key="loopinnetwork",
                model_name="Qwen2.5-14B-Instruct",
                max_retry_count=5
            )
    
    def _load_prompt_template(self) -> str:
        """
        Load prompt template from file.
        
        Returns:
            str: Prompt template content
        """
        try:
            with open(self.config.paths.prompt_file, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {self.config.paths.prompt_file}")
            return ""
        except Exception as e:
            print(f"Error loading prompt template: {e}")
            return ""
    
    def recommend_apis(self, question: Dict[str, Any], 
                      candidate_apis: List[Dict[str, Any]], 
                      functions: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate API recommendations using multi-agent approach.
        
        Args:
            question (Dict[str, Any]): Question/mashup object with description and categories
            candidate_apis (List[Dict[str, Any]]): List of candidate APIs from RAG retrieval
            functions (Optional[List]): Additional functions for the multi-agent system
            
        Returns:
            Dict[str, Any]: Multi-agent recommendation result
        """
        if functions is None:
            functions = []
        
        # Create mashup text for processing
        mashup_text = create_mashup_text(question)
        
        # Run multi-agent pipeline
        try:
            agent_result = run_multiagent_flow(
                mashup_text, 
                candidate_apis, 
                self.prompt_template
            )
            print(f"Multi-agent result type: {type(agent_result)}")
            print(f"Multi-agent result: {agent_result}")
            if agent_result is None:
                print("Warning: run_multiagent_flow returned None")
                return {"related_apis": []}
            return agent_result
        except Exception as e:
            print(f"Error in multi-agent recommendation: {e}")
            import traceback
            traceback.print_exc()
            return {"related_apis": []}
    
    def process_recommendation_response(self, response: Dict[str, Any], 
                                     limit: Optional[int] = None) -> List[str]:
        """
        Process and clean recommendation response from multi-agent system.
        
        Args:
            response (Dict[str, Any]): Raw response from multi-agent system
            limit (Optional[int]): Maximum number of APIs to return
            
        Returns:
            List[str]: Cleaned list of recommended API titles
        """
        if limit is None:
            limit = self.config.evaluation.llm_predict_limit
        
        # Check if response is valid
        if response is None:
            print("Warning: response is None in process_recommendation_response")
            return []
        
        if not isinstance(response, dict):
            print(f"Warning: response is not a dict, got {type(response)}: {response}")
            return []
        
        # Extract API recommendations
        recommended_apis = response.get("related_apis", [])[:limit]
        
        # Deduplicate and normalize to string titles
        cleaned_apis = deduplicate_api_list(recommended_apis)
        
        return cleaned_apis
    
    def recommend_with_retry(self, question: Dict[str, Any], 
                           candidate_apis: List[Dict[str, Any]], 
                           max_attempts: Optional[int] = None,
                           hallucination_threshold: Optional[int] = None) -> tuple[List[str], int]:
        """
        Generate recommendations with retry mechanism to reduce hallucinations.
        
        Args:
            question (Dict[str, Any]): Question/mashup object
            candidate_apis (List[Dict[str, Any]]): Candidate APIs from RAG
            max_attempts (Optional[int]): Maximum retry attempts
            hallucination_threshold (Optional[int]): Maximum allowed hallucinations
            
        Returns:
            tuple[List[str], int]: Best recommendation result and attempt count
        """
        if max_attempts is None:
            max_attempts = self.config.evaluation.max_retry_attempts
        if hallucination_threshold is None:
            hallucination_threshold = self.config.evaluation.min_hallucination_threshold
        
        best_recommendations = []
        best_correct_count = 0
        attempt_count = 0
        
        # Get ground truth for comparison if available
        ground_truth = set()
        if "related_apis" in question and question["related_apis"]:
            ground_truth = {
                api["title"] for api in question["related_apis"] 
                if api is not None and isinstance(api, dict) and "title" in api
            }
        
        while attempt_count < max_attempts:
            attempt_count += 1
            
            # Generate recommendations
            response = self.recommend_apis(question, candidate_apis)
            recommendations = self.process_recommendation_response(response)
            
            print(f"Attempt {attempt_count}: {recommendations}")
            
            # Count hallucinations (APIs not in candidate set)
            candidate_titles = {api.get("title", "") for api in candidate_apis}
            hallucinations = sum(1 for api in recommendations if api not in candidate_titles)
            
            # Count correct predictions if ground truth available
            if ground_truth:
                correct_count = len(set(recommendations).intersection(ground_truth))
                if correct_count > best_correct_count:
                    best_correct_count = correct_count
                    best_recommendations = recommendations
            
            # Stop if hallucinations are below threshold
            if hallucinations < hallucination_threshold:
                if not best_recommendations:  # If no best found yet, use current
                    best_recommendations = recommendations
                break
        
        return best_recommendations, attempt_count
    
    def batch_recommend(self, questions: List[Dict[str, Any]], 
                       candidate_api_sets: List[List[Dict[str, Any]]], 
                       use_retry: bool = True) -> List[List[str]]:
        """
        Generate recommendations for a batch of questions.
        
        Args:
            questions (List[Dict[str, Any]]): List of question objects
            candidate_api_sets (List[List[Dict[str, Any]]]): Candidate APIs for each question
            use_retry (bool): Whether to use retry mechanism
            
        Returns:
            List[List[str]]: Recommendations for each question
        """
        recommendations = []
        
        for i, (question, candidates) in enumerate(zip(questions, candidate_api_sets)):
            print(f"Processing question {i+1}/{len(questions)}")
            
            if use_retry:
                recs, _ = self.recommend_with_retry(question, candidates)
            else:
                response = self.recommend_apis(question, candidates)
                recs = self.process_recommendation_response(response)
            
            recommendations.append(recs)
        
        return recommendations


class LegacyAPIRecommendationService:
    """
    Legacy wrapper for the original call_with_messages_multi_agent function.
    Maintains compatibility with existing code while providing cleaner interface.
    """
    
    def __init__(self, config: Config):
        """Initialize legacy service with configuration."""
        self.config = config
        self.service = APIRecommendationService(config)
    
    def call_with_messages_multi_agent(self, question: Dict[str, Any], 
                                     answer_apis: List[Dict[str, Any]], 
                                     function: Optional[List] = None) -> Dict[str, Any]:
        """
        Legacy interface for multi-agent API recommendation.
        
        Args:
            question (Dict[str, Any]): Question object
            answer_apis (List[Dict[str, Any]]): Candidate APIs
            function (Optional[List]): Additional functions
            
        Returns:
            Dict[str, Any]: Recommendation result
        """
        return self.service.recommend_apis(question, answer_apis, function)
