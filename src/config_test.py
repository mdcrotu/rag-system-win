"""Optimized configuration for testing"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
DB_DIR = PROJECT_ROOT / "data" / "vector_db_test"
LOGS_DIR = PROJECT_ROOT / "logs"

# Optimized settings for testing
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # Better for Q&A
DEFAULT_LLM_MODEL = "llama3.2:1b"  # Fastest model

# Smaller chunks for better matching
MAX_CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

# More results for better context
DEFAULT_RESULTS_PER_SPEC = 5
MAX_CONTEXT_LENGTH = 6000

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
