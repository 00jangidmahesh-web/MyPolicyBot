"""LangGraph nodes: intent classifier, retriever, greeter, out-of-scope handler."""

from langchain_core.messages import SystemMessage, AIMessage
from core.state import AgentState

_vectorstore = None
_use_reranking = True

def init_tool(vectorstore, use_reranking=True):
    global _vectorstore, _use_reranking
    _vectorstore = vectorstore
    _use_reranking = use_reranking

# ----------------------------------------------
# 1. Intent Classifier (LLM + keyword fallback)
# ----------------------------------------------
def intent_classifier(state: AgentState) -> dict:
    from helpers.llms import get_primary_llm
    
    last_msg = state["messages"][-1].content.lower().strip()
    
    # Keyword fallback (no LLM)
    greetings = {"hello", "hi", "hey", "greetings", "who are you", "what can you do", "help"}
    out_of_scope = {"weather", "news", "stock", "movie", "game", "hack", "crack"}
    
    if any(word in last_msg for word in greetings):
        return {"intents": "GREETING"}
    if any(word in last_msg for word in out_of_scope):
        return {"intents": "OUT_OF_SCOPE"}
    
    # LLM attempt for unclear/inquiry
    system_prompt = """
    You are an intent classifier. Classify the user's message into one of:
    - GREETING: hello, hi, bye, thanks, etc.
    - INQUIRY: question about policies, refunds, cancellations, pricing.
    - OUT_OF_SCOPE: anything not related to company policies.

    Respond with ONLY ONE WORD: GREETING, INQUIRY, or OUT_OF_SCOPE.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    try:
        llm = get_primary_llm()
        response = llm.invoke(messages)
        intent_text = response.content.strip().upper()
    except Exception as e:
        print(f"[Intent] OpenRouter failed: {e}. Using keyword INQUIRY.")
        intent_text = "INQUIRY"
    
    if "OUT_OF_SCOPE" in intent_text:
        intent = "OUT_OF_SCOPE"
    elif "INQUIRY" in intent_text:
        intent = "INQUIRY"
    else:
        intent = "GREETING"
    
    return {"intents": intent}

# ----------------------------------------------
# 2. Router after classification
# ----------------------------------------------
def route_after_classify(state: AgentState) -> str:
    intent = state.get("intents", "GREETING").upper()
    if intent == "OUT_OF_SCOPE":
        return "out_of_scope_handler"
    elif intent == "INQUIRY":
        return "retriever"
    else:
        return "greeter"

# ----------------------------------------------
# 3. Greeting node
# ----------------------------------------------
def greet(state: AgentState) -> dict:
    from helpers.llms import get_primary_llm
    from retrieval.prompts import get_prompt

    last_msg = state["messages"][-1].content
    prompt_template = get_prompt("greet")
    system_msg = SystemMessage(content=prompt_template.format(last_user_message=last_msg))
    messages = [system_msg] + state["messages"]
    
    try:
        llm = get_primary_llm()
        response = llm.invoke(messages)
        answer = response.content
        model = "openrouter"
    except Exception as e:
        print(f"[Greet] OpenRouter failed: {e}. Using static response.")
        answer = "Hello! I'm your policy assistant for Neura Dynamics. Ask me about refunds, cancellations, pricing, or delivery policies."
        model = "static"
    
    return {"messages": [AIMessage(content=answer)], "model_used": model}

# ----------------------------------------------
# 4. Out of scope handler
# ----------------------------------------------
def out_of_scope_handler(state: AgentState) -> dict:
    msg = "Sorry, I can only answer questions about Neura Dynamics policies. Please ask something related to refunds, cancellations, pricing, or delivery."
    return {"messages": [AIMessage(content=msg)], "model_used": "none"}

# ----------------------------------------------
# 5. Retriever node (semantic + keyword rerank)
# ----------------------------------------------
def retrieve(state: AgentState) -> dict:
    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialized. Call init_tool() first.")
    
    query = state["messages"][-1].content
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    if _use_reranking and docs:
        query_words = set(query.lower().split())
        scored = []
        for doc in docs:
            chunk_words = set(doc.page_content.lower().split())
            overlap = len(query_words & chunk_words)
            scored.append((doc, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in scored]
    
    context = " ".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "unknown").split("\\")[-1] for d in docs]))
    return {"context": context, "sources": sources}