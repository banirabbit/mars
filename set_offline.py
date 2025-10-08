#!/usr/bin/env python3
"""
Set environment variables to force offline mode for Hugging Face models.
Run this before starting the main application to avoid network requests.
"""

import os

def set_offline_mode():
    """Set environment variables for offline mode."""
    offline_env = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1", 
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    
    print("Setting offline mode environment variables...")
    for key, value in offline_env.items():
        os.environ[key] = value
        print(f"  {key} = {value}")
    
    print("Offline mode activated!")
    print("Models will be loaded from local cache only.")
    return True

if __name__ == "__main__":
    set_offline_mode()
    
    # Import and run main after setting environment
    try:
        print("\nStarting main application in offline mode...")
        import sys
        sys.path.insert(0, "src")
        
        # Run configuration check
        from src.config import Config
        config = Config()
        print(f"Configuration loaded successfully!")
        
        print("\nYou can now run: python main.py")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required models are cached locally.")
