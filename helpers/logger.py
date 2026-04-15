"""Simple logger to trace queries, context, and answers."""

import datetime
import os

# Log file location (relative to project root)
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_trace.log")

def _ensure_log_dir():
    """Create logs directory if it doesn't exist."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def _write(text):
    _ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(stage: str, message: str):
    """Log a simple event (e.g., GREET, QUERY, LLM_ERROR)."""
    _write(f"[{_now()}] [{stage}] {message}\n")

def log_query(query: str, context: str, answer: str, model_used: str):
    """Log a full Q&A trace with context and answer."""
    sep = "=" * 80
    _write(
        f"\n{sep}\n"
        f"[{_now()}] QUERY TRACE (Model: {model_used})\n"
        f"Q: {query}\n"
        f"CONTEXT (first 500 chars): {context[:500]}...\n"
        f"ANSWER: {answer}\n"
        f"{sep}\n"
    )