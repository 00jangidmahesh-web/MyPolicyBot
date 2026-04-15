# MyPolicyBot – RAG Policy Assistant

A Retrieval-Augmented Generation (RAG) chatbot for company policies using LangGraph, ChromaDB, and OpenRouter free LLMs.

## Features
- Intent classification (greeting/inquiry/out-of-scope)
- Semantic search + keyword reranking
- LLM generation via OpenRouter (free tier)
- Conversation memory

## Setup
1. Clone repo
2. `pip install -r requirements.txt`
3. Add `.env` with `OPENROUTER_API_KEY`
4. Place policy `.md` files in `docs/`
5. `python run.py`

## Example
You: what is the refund policy?
Bot: Approved refunds are processed within 7–10 business days...

## License
MIT
