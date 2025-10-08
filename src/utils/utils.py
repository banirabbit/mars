"""
Utility functions for the multi-agent service recommendation system.
Contains helper functions for data processing, JSON handling, and validation.
"""

import json
import re
from typing import List, Dict, Any, Set, Tuple
from collections import Counter


def process_json_response(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON content from text response.
    
    Args:
        text (str): Text containing JSON code blocks
        
    Returns:
        Dict[str, Any]: Parsed JSON data or empty structure if parsing fails
    """
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            json_data = json.loads(match.strip())
            print(f"Successfully parsed JSON: {json_data}")
            
            # Validate that required key exists
            if "related_apis" not in json_data:
                print("Parsed JSON missing 'related_apis' key")
                return {"related_apis": []}
            
            return json_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return {"related_apis": []}
        except Exception as e:
            print(f"Unexpected error in JSON processing: {e}")
            return {"related_apis": []}
    
    print("No JSON code blocks found in text")
    return {"related_apis": []}


def count_hallucinated_apis(predicted_apis: List[str], valid_apis: List[Dict[str, Any]]) -> int:
    """
    Count the number of predicted APIs that don't exist in the valid API set.
    
    Args:
        predicted_apis (List[str]): List of predicted API names
        valid_apis (List[Dict[str, Any]]): List of valid API objects with 'title' field
        
    Returns:
        int: Number of hallucinated (invalid) APIs
    """
    if len(predicted_apis) == 0:
        return 10  # Penalty for empty predictions
    
    valid_api_titles = {api['title'] for api in valid_apis}
    hallucination_count = 0
    
    for api in predicted_apis:
        if api and api not in valid_api_titles:
            hallucination_count += 1
    
    return hallucination_count


def deduplicate_api_list(api_list: List[Any]) -> List[str]:
    """
    Remove duplicates from API list and normalize to string titles.
    
    Args:
        api_list (List[Any]): List of API objects (dict or str)
        
    Returns:
        List[str]: Deduplicated list of API titles
    """
    unique_apis = []
    seen_titles = set()
    
    for item in api_list:
        title = None
        
        if isinstance(item, dict) and 'title' in item:
            title = item['title']
        elif isinstance(item, str):
            title = item
        
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_apis.append(title)
    
    return unique_apis


def extract_api_titles_from_mashup(mashup_data: Dict[str, Any]) -> List[str]:
    """
    Extract API titles from mashup's related_apis field.
    
    Args:
        mashup_data (Dict[str, Any]): Mashup object with related_apis field
        
    Returns:
        List[str]: List of API titles
    """
    api_titles = []
    
    if "related_apis" in mashup_data and mashup_data["related_apis"]:
        for api in mashup_data["related_apis"]:
            if isinstance(api, dict) and api is not None and "title" in api:
                api_titles.append(api["title"])
    
    return api_titles


def rank_apis_by_frequency(api_doc_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank APIs by their frequency of occurrence and remove duplicates.
    
    Args:
        api_doc_set (List[Dict[str, Any]]): List of API documents
        
    Returns:
        List[Dict[str, Any]]: Sorted unique APIs by frequency
    """
    # Count occurrences of each API title
    title_counts = Counter(obj["title"] for obj in api_doc_set)
    
    # Remove duplicates, keeping the first occurrence
    unique_objects = {}
    for obj in api_doc_set:
        if obj["title"] not in unique_objects:
            unique_objects[obj["title"]] = obj
    
    # Sort by frequency (descending)
    sorted_objects = sorted(
        unique_objects.values(),
        key=lambda x: title_counts[x["title"]],
        reverse=True
    )
    
    return sorted_objects


def load_json_file(file_path: str) -> Any:
    """
    Load and parse JSON file with error handling.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        Any: Parsed JSON data or None if loading fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str, ensure_ascii: bool = False, indent: int = 4) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data (Any): Data to save
        file_path (str): Output file path
        ensure_ascii (bool): Whether to ensure ASCII encoding
        indent (int): JSON indentation level
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=ensure_ascii, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False


def prepare_answer_list(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare answer list from question data for evaluation.
    
    Args:
        questions (List[Dict[str, Any]]): List of question objects
        
    Returns:
        List[Dict[str, Any]]: List of answer objects with title and answers fields
    """
    answer_list = []
    
    for question in questions:
        answer = {
            "title": question["title"],
            "answers": extract_api_titles_from_mashup(question)
        }
        answer_list.append(answer)
    
    return answer_list


def validate_api_object(api: Any) -> bool:
    """
    Validate that an API object has required fields.
    
    Args:
        api (Any): API object to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return (
        api is not None and
        isinstance(api, dict) and
        "title" in api and
        "tags" in api
    )


def create_mashup_text(question: Dict[str, Any]) -> str:
    """
    Create formatted text representation of mashup for processing.
    
    Args:
        question (Dict[str, Any]): Question/mashup object
        
    Returns:
        str: Formatted mashup text
    """
    description = question.get("description", "")
    categories = question.get("categories", [])
    
    return f"description:{description}, categories:{categories}"
