"""Configuration settings – load from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# ------------------- API Keys -------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")   # optional, for evaluation

# ------------------- Paths -------------------
BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")          # policy markdown files
STORAGE_DIR = os.path.join(BASE_DIR, "storage")    # chroma DB persistence
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ------------------- Chunking -------------------
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks

# ------------------- Retrieval -------------------
TOP_K = 3                 # number of chunks to retrieve
USE_RERANKING = True      # enable keyword reranking

# ------------------- LLM -------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
GOOGLE_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.1     # low = factual, deterministic

# ------------------- Memory -------------------
MEMORY_MAX_TURNS = 5      # keep last N Q&A pairs in history

# ------------------- Embedding -------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"