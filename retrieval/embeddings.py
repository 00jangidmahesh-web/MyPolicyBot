"""Google Generative AI embeddings using LangChain."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # Load GOOGLE_API_KEY from .env file

# Model name from Google's embedding models
EMBEDDING_MODEL = "models/gemini-embedding-001"

def get_embedding_function():
    """
    Returns a callable embedding function (LangChain compatible)
    that uses Google's Gemini embedding model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Set it in .env file.")
    
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )