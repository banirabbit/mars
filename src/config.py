"""
Configuration management for the multi-agent service recommendation system.
Loads configuration from YAML file and manages all configurable parameters.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for embedding and reranking models."""
    embed_model_path: str
    rerank_model_name: str
    api_embed_model_name: str
    trust_remote_code: bool
    show_progress: bool


@dataclass
class RetrievalConfig:
    """Configuration for retrieval and ranking parameters."""
    initial_k: int
    rerank_top_n: int
    final_api_limit: int
    similarity_top_n: int
    ordered_apis_limit: int
    bm25_k1: float
    bm25_b: float
    bm25_weight: float
    vector_weight: float


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    vectordb_dir: str
    prompt_file: str
    mashup_data_path: str
    api_data_path: str
    train_data_path: str
    test_data_path: str
    answer_cache_file: str
    log_file: str


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    max_retry_attempts: int
    min_hallucination_threshold: int
    ndcg_k: int
    llm_predict_limit: int


@dataclass
class SystemConfig:
    """Configuration for system parameters."""
    debug: bool
    num_workers: int
    enable_cache: bool
    cache_ttl: int


class Config:
    """Main configuration class that loads from YAML and combines all configuration sections."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path (Optional[str]): Path to config file, defaults to config.yaml in project root
        """
        if config_path is None:
            # Find project root (where config.yaml should be)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent  # Go up one level from src/
            config_path = project_root / "config.yaml"
        
        self.project_root = Path(config_path).parent
        self.config_data = self._load_config(config_path)
        
        # Initialize configuration sections
        self.model = self._create_model_config()
        self.retrieval = self._create_retrieval_config()
        self.paths = self._create_path_config()
        self.evaluation = self._create_evaluation_config()
        self.system = self._create_system_config()
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                print(f"Loaded configuration from: {config_path}")
                return config_data
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration values")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            print("Using default configuration values")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available."""
        return {
            'model': {
                'embed_model_path': 'src/embedder/finetuned_bge_singlegpu_2025-07-07_18-47',
                'rerank_model_name': 'BAAI/bge-reranker-v2-m3',
                'api_embed_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'trust_remote_code': True,
                'show_progress': True
            },
            'retrieval': {
                'initial_k': 100,
                'rerank_top_n': 90,
                'final_api_limit': 50,
                'similarity_top_n': 10,
                'ordered_apis_limit': 40,
                'bm25_k1': 1.5,
                'bm25_b': 0.75,
                'bm25_weight': 0.5157,
                'vector_weight': 0.4843
            },
            'paths': {
                'vectordb_dir': 'data/vector_db',
                'prompt_file': 'prompts/qwen_best.txt',
                'mashup_data_path': 'data/origin/active_mashups_data.txt',
                'api_data_path': 'data/origin/active_apis_data.txt',
                'train_data_path': 'data/rewrite/seed42/train_rewrite_data1202.json',
                'test_data_path': 'data/rewrite/seed42/test_rewrite_data1202.json',
                'answer_cache_file': 'output/rag_cache.json',
                'log_file': 'logs/evaluation.json'
            },
            'evaluation': {
                'max_retry_attempts': 15,
                'min_hallucination_threshold': 1,
                'ndcg_k': 10,
                'llm_predict_limit': 10
            },
            'system': {
                'debug': False,
                'num_workers': 4,
                'enable_cache': True,
                'cache_ttl': 3600
            }
        }
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration from loaded data."""
        model_data = self.config_data.get('model', {})
        return ModelConfig(
            embed_model_path=str(self.project_root / model_data.get('embed_model_path', 'src/embedder/finetuned_bge_singlegpu_2025-07-07_18-47')),
            rerank_model_name=model_data.get('rerank_model_name', 'BAAI/bge-reranker-v2-m3'),
            api_embed_model_name=model_data.get('api_embed_model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            trust_remote_code=model_data.get('trust_remote_code', True),
            show_progress=model_data.get('show_progress', True)
        )
    
    def _create_retrieval_config(self) -> RetrievalConfig:
        """Create retrieval configuration from loaded data."""
        retrieval_data = self.config_data.get('retrieval', {})
        return RetrievalConfig(
            initial_k=retrieval_data.get('initial_k', 100),
            rerank_top_n=retrieval_data.get('rerank_top_n', 90),
            final_api_limit=retrieval_data.get('final_api_limit', 50),
            similarity_top_n=retrieval_data.get('similarity_top_n', 10),
            ordered_apis_limit=retrieval_data.get('ordered_apis_limit', 40),
            bm25_k1=retrieval_data.get('bm25_k1', 1.5),
            bm25_b=retrieval_data.get('bm25_b', 0.75),
            bm25_weight=retrieval_data.get('bm25_weight', 0.5157),
            vector_weight=retrieval_data.get('vector_weight', 0.4843)
        )
    
    def _create_path_config(self) -> PathConfig:
        """Create path configuration from loaded data with absolute paths."""
        paths_data = self.config_data.get('paths', {})
        return PathConfig(
            vectordb_dir=str(self.project_root / paths_data.get('vectordb_dir', 'data/vector_db')),
            prompt_file=str(self.project_root / paths_data.get('prompt_file', 'prompts/qwen_best.txt')),
            mashup_data_path=str(self.project_root / paths_data.get('mashup_data_path', 'data/origin/active_mashups_data.txt')),
            api_data_path=str(self.project_root / paths_data.get('api_data_path', 'data/origin/active_apis_data.txt')),
            train_data_path=str(self.project_root / paths_data.get('train_data_path', 'data/rewrite/seed42/train_rewrite_data1202.json')),
            test_data_path=str(self.project_root / paths_data.get('test_data_path', 'data/rewrite/seed42/test_rewrite_data1202.json')),
            answer_cache_file=str(self.project_root / paths_data.get('answer_cache_file', 'output/rag_cache.json')),
            log_file=str(self.project_root / paths_data.get('log_file', 'logs/evaluation.json'))
        )
    
    def _create_evaluation_config(self) -> EvaluationConfig:
        """Create evaluation configuration from loaded data."""
        eval_data = self.config_data.get('evaluation', {})
        return EvaluationConfig(
            max_retry_attempts=eval_data.get('max_retry_attempts', 15),
            min_hallucination_threshold=eval_data.get('min_hallucination_threshold', 1),
            ndcg_k=eval_data.get('ndcg_k', 10),
            llm_predict_limit=eval_data.get('llm_predict_limit', 10)
        )
    
    def _create_system_config(self) -> SystemConfig:
        """Create system configuration from loaded data."""
        system_data = self.config_data.get('system', {})
        return SystemConfig(
            debug=system_data.get('debug', False),
            num_workers=system_data.get('num_workers', 4),
            enable_cache=system_data.get('enable_cache', True),
            cache_ttl=system_data.get('cache_ttl', 3600)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'model': self.model.__dict__,
            'retrieval': self.retrieval.__dict__,
            'paths': self.paths.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__
        }
    
    def validate_paths(self) -> bool:
        """Validate that required files and directories exist."""
        required_files = [
            self.paths.mashup_data_path,
            self.paths.api_data_path,
            self.paths.train_data_path,
            self.paths.test_data_path
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("Warning: Required files not found:")
            for file_path in missing_files:
                print(f"  - {file_path}")
        
        # Check if prompt file exists (optional)
        if not os.path.exists(self.paths.prompt_file):
            print(f"Warning: Prompt file not found: {self.paths.prompt_file}")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.paths.log_file), exist_ok=True)
        os.makedirs(self.paths.vectordb_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.paths.answer_cache_file), exist_ok=True)
        
        return len(missing_files) == 0
    
    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self.project_root


# Global configuration instance
config = Config()
