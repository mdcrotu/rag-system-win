"""Configuration settings for the RAG system (Windows)"""
import os
from pathlib import Path

# Paths (Windows-compatible)
PROJECT_ROOT = Path(__file__).parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
DB_DIR = PROJECT_ROOT / "data" / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# Embedding model settings
#EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # Better for Q&A
# Alternative: "all-mpnet-base-v2"  # Slower, better quality
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Slower, better quality

# LLM settings
DEFAULT_LLM_MODEL = "llama3.2:3b"
# Alternative models: "llama3.2:1b", "llama3.1:8b"

# Chunking settings
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Query settings
DEFAULT_RESULTS_PER_SPEC = 3
MAX_CONTEXT_LENGTH = 4000

# Create directories (Windows-safe)
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
