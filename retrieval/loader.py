"""Load all markdown policy documents from the data directory."""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os

# Assuming your project structure:
# MyPolicyBot/
#   data/           <-- put your .md files here
#   retrieval/
#       loader.py
#   ...

def load_all_documents(data_folder="data"):
    """
    Load all .md files from the given folder.
    Returns a list of LangChain Document objects.
    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder '{data_folder}' not found. Create it and add .md files.")
    
    loader = DirectoryLoader(
        data_folder,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} document(s) from '{data_folder}'")
    return docs