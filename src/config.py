"""
config.py
---------
Central configuration for the Document Q&A RAG bot.
All tunable constants live here so the rest of the codebase never hardcodes
paths, model names, or chunking parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model Configuration

GENERATION_MODEL = "gemini-2.5-flash"

EMBEDDING_MODEL = "models/gemini-embedding-001"


# Path Configuration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

# Chunking Configuration

CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # character overlap between consecutive chunks

# Retrieval Configuration

TOP_K = 4                          # number of chunks to retrieve per query
DISTANCE_THRESHOLD = 1.0           # cosine distance cutoff; chunks beyond this are dropped as irrelevant
COLLECTION_NAME = "document_knowledge_base"


# Validation helper

def validate_config():
    """Raise a clear error early if required configuration is missing."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Create a .env file in the project root "
            "(see .env.example) and add: GEMINI_API_KEY=your_key_here"
        )
