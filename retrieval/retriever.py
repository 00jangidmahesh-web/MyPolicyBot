"""Retriever – similarity search + simple keyword-based reranking."""

def get_retriever(vectorstore, top_k=3):
    """
    Returns a base retriever (semantic similarity only).
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

def rerank_by_keywords(query, docs_with_scores):
    """
    Rerank retrieved chunks by keyword overlap with query.
    docs_with_scores: list of (Document, score) tuples.
    Returns reranked list of (Document, score).
    """
    query_words = set(query.lower().split())
    scored = []
    for doc, score in docs_with_scores:
        chunk_words = set(doc.page_content.lower().split())
        overlap = len(query_words & chunk_words)
        # Higher overlap is better; if tie, preserve original order (lower score index)
        scored.append((doc, score, overlap))
    # Sort by overlap desc, then by original score (ascending distance)
    scored.sort(key=lambda x: (-x[2], x[1]))
    return [(doc, score) for doc, score, _ in scored]

def retrieve_and_rerank(vectorstore, query, top_k=3, use_reranking=True):
    """
    Full retrieval pipeline: semantic search + optional reranking.
    Returns (context_string, list_of_source_filenames)
    """
    retriever = get_retriever(vectorstore, top_k)
    docs = retriever.invoke(query)
    
    if use_reranking:
        # Convert to (doc, score) format; score is distance (lower better)
        doc_tuples = [(doc, doc.metadata.get("score", 0.0)) for doc in docs]
        reranked = rerank_by_keywords(query, doc_tuples)
        docs = [doc for doc, _ in reranked]
    
    context = " ".join([doc.page_content for doc in docs])
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        src = src.replace("\\", "/").split("/")[-1]   # get filename only
        sources.append(src)
    sources = list(dict.fromkeys(sources))   # remove duplicates
    
    return context, sources