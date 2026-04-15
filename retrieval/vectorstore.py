"""ChromaDB vector store – build from chunks or load from disk."""

import os
from langchain_chroma import Chroma
from retrieval.embeddings import get_embedding_function

COLLECTION_NAME = "policy_kb"
PERSIST_DIR = "chroma_db"   # folder where vector DB will be saved

def build_vectorstore(chunks):
    """
    Create a new vector store from document chunks.
    """
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"[VectorStore] Built with {len(chunks)} chunks")
    return store

def load_vectorstore():
    """
    Load existing vector store from disk.
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=PERSIST_DIR,
    )
    print("[VectorStore] Loaded from disk")
    return store

def vectorstore_exists():
    return os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)