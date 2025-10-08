"""
Utils package for the multi-agent API recommendation system.
Contains utility functions for data processing, JSON handling, and validation.
"""

# Import all utility functions for easy access
from .utils import (
    process_json_response,
    count_hallucinated_apis,
    deduplicate_api_list,
    extract_api_titles_from_mashup,
    rank_apis_by_frequency,
    load_json_file,
    save_json_file,
    prepare_answer_list,
    validate_api_object,
    create_mashup_text
)

__all__ = [
    'process_json_response',
    'count_hallucinated_apis', 
    'deduplicate_api_list',
    'extract_api_titles_from_mashup',
    'rank_apis_by_frequency',
    'load_json_file',
    'save_json_file',
    'prepare_answer_list',
    'validate_api_object',
    'create_mashup_text'
]
