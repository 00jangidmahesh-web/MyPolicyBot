# MyPolicyBot – RAG Policy Assistant

[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-blue)](https://openrouter.ai)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green)](https://langchain.com/langgraph)
[![Python](https://img.shields.io/badge/Python-3.13-yellow)]()

A **Retrieval-Augmented Generation (RAG)** chatbot that answers company policy questions with high accuracy using **LangGraph**, **ChromaDB**, and **OpenRouter (free LLMs)**.

---

## Key Highlights

-  **Agentic RAG System** built using LangGraph  
-  **Hybrid Retrieval** (Semantic Search + Keyword Reranking)  
-  **Conversational Memory** (last 5 interactions)  
-  **Source-grounded Answers** with citations  
-  **100% Free Stack** (OpenRouter free-tier models)  
-  **Modular & Extensible Architecture**

---

##  Features

-  **Intent Classification**  
  Detects user intent: `GREETING / INQUIRY / OUT_OF_SCOPE`

-  **Advanced Retrieval Pipeline**  
  Combines:
  - Vector similarity search (ChromaDB)  
  - Keyword-based reranking  

-  **Conversation Memory**  
  Maintains context for better multi-turn responses  

- **Source Citations**  
  Displays exact policy documents used in answers  

-  **Dynamic LLM Selection**  
  Uses `openrouter/free` to auto-select best free model  

-  **Prompt Versioning**  
  Switch between multiple prompt strategies (`v1`, `v2`)

---

##  Tech Stack

| Component        | Technology |
|----------------|-----------|
|  Workflow     | LangGraph |
|  Vector Store | ChromaDB |
|  Embeddings   | Google Generative AI (`gemini-embedding-001`) |
|  LLM          | OpenRouter (`openrouter/free`) |
|  Framework    | LangChain |
|  Chunking     | RecursiveCharacterTextSplitter (500 / 50 overlap) |

---

##  Installation

### 1️ Clone the repository
```bash
git clone https://github.com/00jangidmahesh-web/MyPolicyBot.git
cd MyPolicyBot
```

### 2️ Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3️ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️ Setup environment variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

Get free API key from: https://openrouter.ai

---

##  Add Policy Documents

Place your `.md` files inside:

```
docs/
```

The bot will automatically index them.

---

##  Build Vector Store (First Run)

```bash
python run.py
```

Vector database will be created automatically if not present.

---

##  Usage

Run the chatbot:

```bash
python run.py
```

---

## CLI Commands

| Command       | Action |
|--------------|--------|
| `quit / exit` | Exit the bot |
| `clear`       | Clear conversation memory |
| `prompt v1`   | Switch to prompt version 1 |
| `prompt v2`   | Switch to prompt version 2 |

---

## Example Conversation

```
You: who are you?
Bot: I'm your policy assistant for Neura Dynamics. Ask me about refunds, cancellations, pricing, or delivery policies.

You: what is the refund policy for annual subscriptions?
Bot: Annual subscription fees are generally non-refundable once the term starts...
Sources: refund_policy.md

You: what discounts are available?
Bot: This information is not available in the provided policy documents.
```

---

##  Project Structure

```
MyPolicyBot/
├── core/               # LangGraph workflow (state, nodes, graph)
├── retrieval/          # RAG pipeline (retriever, embeddings, prompts, generation)
├── helpers/            # LLM factories, logging utilities
├── evaluation/         # DeepEval test cases
├── docs/               # Policy markdown files
├── storage/            # ChromaDB persistence
├── logs/               # Query logs
├── run.py              # CLI entry point
├── settings.py         # Configuration
├── requirements.txt
└── .env                # Environment variables (ignored)
```

---

## Evaluation

Run evaluation using DeepEval:

```bash
python evaluation/evaluate.py --prompt v2
```

Compare prompt versions:

```bash
python evaluation/evaluate.py --compare
```

Metrics evaluated:
-  Answer Relevancy  
-  Faithfulness  
-  Context Utilization  

---

## Future Improvements

-  Web UI (Streamlit / React)  
-  PDF & DOCX support  
-  Role-based access to policies  
-  Analytics dashboard for queries  
-  API deployment (FastAPI)  

---

##  Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

---

##  License

MIT License

---
