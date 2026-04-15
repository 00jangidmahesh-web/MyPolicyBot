"""Split documents into smaller chunks for embedding and retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chunk size: 500 characters (approx 1 policy section)
# Overlap: 50 characters to avoid losing context at boundaries
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Separators that respect markdown structure (headings, horizontal rules, paragraphs)
SEPARATORS = ["\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " ", ""]

def chunk_documents(docs):
    """
    Split a list of documents into smaller chunks.
    Returns a list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    print(f"[Chunker] Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks