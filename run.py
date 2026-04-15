"""Interactive CLI – chat with the RAG assistant."""

import os
import sys
from collections import deque
from langchain_core.messages import HumanMessage, AIMessage

# Add project root to path (if running directly)
sys.path.insert(0, os.path.dirname(__file__))

import settings
from retrieval.loader import load_all_documents
from retrieval.chunker import chunk_documents
from retrieval.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists
from core.nodes import init_tool
from core.graph import ask
from helpers.logger import log, log_query

def build_or_load_vectorstore():
    """Build vectorstore if not exists, otherwise load."""
    if not vectorstore_exists():
        print("\n📁 No vector store found. Building from docs...")
        docs = load_all_documents(settings.DOCS_DIR)
        chunks = chunk_documents(docs)
        store = build_vectorstore(chunks)
        print("✅ Vector store built and saved.\n")
    else:
        store = load_vectorstore()
        print("📂 Loaded existing vector store.\n")
    return store

def interactive_chat(prompt_version="v2", use_reranking=True):
    """Main interactive loop."""
    # Load or build vectorstore
    store = build_or_load_vectorstore()
    init_tool(store, use_reranking)

    # Conversation memory (last N turns)
    history = deque(maxlen=settings.MEMORY_MAX_TURNS)

    print("\n" + "=" * 70)
    print("  🤖 NeuraRAG – Policy Assistant")
    print(f"  Prompt: {prompt_version} | Reranking: {'ON' if use_reranking else 'OFF'}")
    print(f"  Memory: last {settings.MEMORY_MAX_TURNS} exchanges")
    print("  Type 'quit' to exit, 'clear' to reset memory, 'prompt <v1/v2>' to switch")
    print("=" * 70 + "\n")

    current_prompt = prompt_version

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("quit", "exit"):
            print("👋 Bye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("🧹 Memory cleared.\n")
            continue
        if user_input.lower().startswith("prompt "):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in ("v1", "v2"):
                current_prompt = parts[1]
                print(f"✅ Switched to prompt {current_prompt}\n")
            else:
                print("⚠️ Usage: prompt v1  or  prompt v2\n")
            continue

        # Send to RAG pipeline
        result = ask(user_input, prompt_version=current_prompt, chat_history=list(history))

        # Store in memory
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=result["answer"]))

        # Print answer
        print("\n" + "─" * 70)
        print(result["answer"])
        print("─" * 70)
        if result["sources"]:
            print(f"📄 Sources: {', '.join(result['sources'])}")
        print(f"🤖 Model: {result['model_used']} | Prompt: {result['prompt_version']}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NeuraRAG Interactive Policy Assistant")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2", help="Prompt version")
    parser.add_argument("--no-rerank", action="store_true", help="Disable keyword reranking")
    args = parser.parse_args()

    interactive_chat(prompt_version=args.prompt, use_reranking=not args.no_rerank)

if __name__ == "__main__":
    main()