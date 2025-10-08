"""
Evaluation service for measuring API recommendation performance.
Provides metrics calculation, logging, and result analysis functionality.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from .utils.NormalizedDCG import NormalizedDCG
from .config import Config
from .utils import load_json_file, save_json_file


class EvaluationMetrics:
    """
    Container class for evaluation metrics.
    Stores and calculates precision, recall, F1-score, NDCG, and hallucination rates.
    """
    
    def __init__(self):
        """Initialize metrics with default values."""
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.ndcg = 0.0
        self.hallucination_rate = 0.0
        self.total_correct = 0
        self.total_predicted = 0
        self.total_ground_truth = 0
        self.total_hallucinations = 0
    
    def calculate_metrics(self, correct_predictions: int, total_predictions: int, 
                         total_ground_truth: int, hallucinations: int) -> None:
        """
        Calculate all evaluation metrics based on counts.
        
        Args:
            correct_predictions (int): Number of correct predictions
            total_predictions (int): Total number of predictions made
            total_ground_truth (int): Total number of ground truth items
            hallucinations (int): Number of hallucinated predictions
        """
        self.total_correct = correct_predictions
        self.total_predicted = total_predictions
        self.total_ground_truth = total_ground_truth
        self.total_hallucinations = hallucinations
        
        # Calculate precision
        self.precision = (
            correct_predictions / total_predictions 
            if total_predictions > 0 else 0.0
        )
        
        # Calculate recall
        self.recall = (
            correct_predictions / total_ground_truth 
            if total_ground_truth > 0 else 0.0
        )
        
        # Calculate F1-score
        self.f1_score = (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0 else 0.0
        )
        
        # Calculate hallucination rate
        self.hallucination_rate = (
            hallucinations / total_predictions 
            if total_predictions > 0 else 0.0
        )
    
    def set_ndcg(self, ndcg_score: float) -> None:
        """
        Set NDCG score.
        
        Args:
            ndcg_score (float): NDCG score to set
        """
        self.ndcg = ndcg_score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary containing all metrics
        """
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "ndcg": self.ndcg,
            "hallucination_rate": self.hallucination_rate,
            "total_correct": self.total_correct,
            "total_predicted": self.total_predicted,
            "total_ground_truth": self.total_ground_truth,
            "total_hallucinations": self.total_hallucinations
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}, "
            f"F1-Score: {self.f1_score:.4f}, "
            f"NDCG: {self.ndcg:.4f}, "
            f"Hallucination Rate: {self.hallucination_rate:.4f}"
        )


class EvaluationLogger:
    """
    Handles logging and state persistence for evaluation processes.
    Supports resumable evaluation with checkpoint saving.
    """
    
    def __init__(self, log_file_path: str):
        """
        Initialize evaluation logger.
        
        Args:
            log_file_path (str): Path to log file for saving state
        """
        self.log_file_path = log_file_path
        self.log_data = self._load_or_create_log()
    
    def _load_or_create_log(self) -> Dict[str, Any]:
        """
        Load existing log file or create new log structure.
        
        Returns:
            Dict[str, Any]: Log data structure
        """
        if os.path.exists(self.log_file_path):
            log_data = load_json_file(self.log_file_path)
            if log_data is not None:
                print(f"Loaded existing log from {self.log_file_path}")
                return log_data
        
        # Create new log structure
        print(f"Creating new log file at {self.log_file_path}")
        return {
            "last_index": 0,
            "hallu_answer": 0,
            "llm_truth_num": 0,
            "all_num": 0,
            "total_ndcg": 0.0,
            "reco_apis": []
        }
    
    def get_resume_state(self) -> Tuple[int, int, int, int, float, List[Any]]:
        """
        Get current state for resuming evaluation.
        
        Returns:
            Tuple: (start_index, hallucinations, correct_predictions, 
                   total_ground_truth, total_ndcg, recommendations)
        """
        return (
            self.log_data["last_index"],
            self.log_data["hallu_answer"],
            self.log_data["llm_truth_num"],
            self.log_data["all_num"],
            self.log_data["total_ndcg"],
            self.log_data["reco_apis"]
        )
    
    def update_state(self, index: int, hallucinations: int, correct_predictions: int,
                    total_ground_truth: int, total_ndcg: float, 
                    recommendations: List[Any]) -> None:
        """
        Update evaluation state and save to log file.
        
        Args:
            index (int): Current processing index
            hallucinations (int): Total hallucinations so far
            correct_predictions (int): Total correct predictions so far
            total_ground_truth (int): Total ground truth items so far
            total_ndcg (float): Total NDCG score so far
            recommendations (List[Any]): All recommendations so far
        """
        self.log_data.update({
            "last_index": index + 1,
            "hallu_answer": hallucinations,
            "llm_truth_num": correct_predictions,
            "all_num": total_ground_truth,
            "total_ndcg": total_ndcg,
            "reco_apis": recommendations
        })
        
        # Save to file
        save_json_file(self.log_data, self.log_file_path)
    
    def get_recommendations(self) -> List[Any]:
        """Get all stored recommendations."""
        return self.log_data.get("reco_apis", [])


class EvaluationService:
    """
    Main evaluation service for measuring API recommendation performance.
    Handles metrics calculation, NDCG computation, and result analysis.
    """
    
    def __init__(self, config: Config):
        """
        Initialize evaluation service.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.ndcg_calculator = NormalizedDCG(config.evaluation.ndcg_k)
        self.logger = EvaluationLogger(config.paths.log_file)
    
    def extract_ground_truth_apis(self, question: Dict[str, Any]) -> set:
        """
        Extract ground truth API titles from question object.
        
        Args:
            question (Dict[str, Any]): Question object with related_apis field
            
        Returns:
            set: Set of ground truth API titles
        """
        ground_truth = set()
        
        if "related_apis" in question and question["related_apis"]:
            for api in question["related_apis"]:
                if (api is not None and 
                    isinstance(api, dict) and 
                    "title" in api):
                    ground_truth.add(api["title"])
        
        return ground_truth
    
    def calculate_single_evaluation(self, predicted_apis: List[str], 
                                  ground_truth_apis: set, 
                                  valid_api_titles: set) -> Tuple[int, int, float]:
        """
        Calculate evaluation metrics for a single prediction.
        
        Args:
            predicted_apis (List[str]): List of predicted API titles
            ground_truth_apis (set): Set of ground truth API titles
            valid_api_titles (set): Set of valid API titles (for hallucination detection)
            
        Returns:
            Tuple[int, int, float]: (correct_predictions, hallucinations, ndcg_score)
        """
        # Count correct predictions
        correct_predictions = len(set(predicted_apis).intersection(ground_truth_apis))
        
        # Count hallucinations (predicted APIs not in valid set)
        hallucinations = sum(
            1 for api in predicted_apis 
            if api and api not in valid_api_titles
        )
        
        # Calculate NDCG
        ndcg_score = 0.0
        if ground_truth_apis:
            ndcg_score = self.ndcg_calculator.calculate_ndcg(
                predicted_apis, 
                list(ground_truth_apis)
            )
        
        return correct_predictions, hallucinations, ndcg_score
    
    def evaluate_recommendations(self, questions: List[Dict[str, Any]], 
                               all_recommendations: List[List[str]], 
                               valid_apis: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Evaluate a complete set of recommendations.
        
        Args:
            questions (List[Dict[str, Any]]): List of question objects
            all_recommendations (List[List[str]]): Recommendations for each question
            valid_apis (List[Dict[str, Any]]): List of all valid APIs
            
        Returns:
            EvaluationMetrics: Comprehensive evaluation metrics
        """
        # Create set of valid API titles for hallucination detection, filtering out None values
        valid_api_titles = {
            api.get("title", "") for api in valid_apis 
            if api is not None and isinstance(api, dict) and "title" in api
        }
        
        total_correct = 0
        total_hallucinations = 0
        total_ground_truth = 0
        total_ndcg = 0.0
        total_predictions = 0
        
        for question, recommendations in zip(questions, all_recommendations):
            # Extract ground truth
            ground_truth = self.extract_ground_truth_apis(question)
            
            # Calculate metrics for this prediction
            correct, hallucinations, ndcg = self.calculate_single_evaluation(
                recommendations, ground_truth, valid_api_titles
            )
            
            # Accumulate totals
            total_correct += correct
            total_hallucinations += hallucinations
            total_ground_truth += len(ground_truth)
            total_ndcg += ndcg
            total_predictions += len(recommendations)
        
        # Create and populate metrics object
        metrics = EvaluationMetrics()
        metrics.calculate_metrics(
            total_correct, total_predictions, 
            total_ground_truth, total_hallucinations
        )
        metrics.set_ndcg(total_ndcg / len(questions) if questions else 0.0)
        
        return metrics
    
    def run_incremental_evaluation(self, questions: List[Dict[str, Any]], 
                                 recommendation_function, 
                                 candidate_api_sets: List[List[Dict[str, Any]]], 
                                 valid_apis: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Run evaluation with incremental logging and resume capability.
        
        Args:
            questions (List[Dict[str, Any]]): List of questions to evaluate
            recommendation_function: Function that generates recommendations
            candidate_api_sets (List[List[Dict[str, Any]]]): Candidate APIs for each question
            valid_apis (List[Dict[str, Any]]): All valid APIs for hallucination detection
            
        Returns:
            EvaluationMetrics: Final evaluation metrics
        """
        # Get resume state
        (start_index, total_hallucinations, total_correct, 
         total_ground_truth, total_ndcg, all_recommendations) = self.logger.get_resume_state()
        
        print(f"Starting evaluation from index {start_index}")
        
        # Create set of valid API titles, filtering out None values
        valid_api_titles = {
            api.get("title", "") for api in valid_apis 
            if api is not None and isinstance(api, dict) and "title" in api
        }
        
        # Process remaining questions
        pbar = tqdm(
            total=len(questions), 
            desc="Evaluating recommendations", 
            initial=start_index,
            colour="blue"
        )
        
        for index in range(start_index, len(questions)):
            pbar.update(1)
            
            question = questions[index]
            candidates = candidate_api_sets[index] if index < len(candidate_api_sets) else []
            
            # Generate recommendations
            recommendations = recommendation_function(question, candidates)
            all_recommendations.append(recommendations)
            
            # Extract ground truth and calculate metrics
            ground_truth = self.extract_ground_truth_apis(question)
            correct, hallucinations, ndcg = self.calculate_single_evaluation(
                recommendations, ground_truth, valid_api_titles
            )
            
            # Update totals
            total_correct += correct
            total_hallucinations += hallucinations
            total_ground_truth += len(ground_truth)
            total_ndcg += ndcg
            
            # Log progress
            self.logger.update_state(
                index, total_hallucinations, total_correct,
                total_ground_truth, total_ndcg, all_recommendations
            )
            
            # Print progress info
            if (index + 1) % 10 == 0:
                current_precision = total_correct / ((index + 1) * self.config.evaluation.llm_predict_limit)
                current_recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
                print(f"Progress: {index + 1}/{len(questions)}, "
                      f"Precision: {current_precision:.4f}, "
                      f"Recall: {current_recall:.4f}")
        
        pbar.close()
        
        # Calculate final metrics
        total_predictions = len(questions) * self.config.evaluation.llm_predict_limit
        metrics = EvaluationMetrics()
        metrics.calculate_metrics(
            total_correct, total_predictions, 
            total_ground_truth, total_hallucinations
        )
        metrics.set_ndcg(total_ndcg / len(questions) if questions else 0.0)
        
        return metrics
    
    def print_evaluation_results(self, metrics: EvaluationMetrics) -> None:
        """
        Print formatted evaluation results.
        
        Args:
            metrics (EvaluationMetrics): Metrics to display
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"LLM Recall: {metrics.recall:.4f}")
        print(f"LLM Precision: {metrics.precision:.4f}")
        print(f"LLM F1-Score: {metrics.f1_score:.4f}")
        print(f"LLM NDCG: {metrics.ndcg:.4f}")
        print(f"Hallucination Count: {metrics.total_hallucinations}")
        print(f"Hallucination Rate: {metrics.hallucination_rate:.4f}")
        print("="*50)
