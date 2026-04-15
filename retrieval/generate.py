"""LLM generation node – uses context + prompt to produce final answer."""

from langchain_core.messages import SystemMessage, AIMessage
from retrieval.prompts import get_prompt
from helpers.llms import get_primary_llm

def run_llm(state):
    question = state["messages"][-1].content
    context = state.get("context", "")
    prompt_version = state.get("prompt_version", "v2")
    
    system_content = get_prompt(prompt_version).format(
        last_user_message=question,
        context=context
    )
    messages = [SystemMessage(content=system_content)] + state["messages"]
    
    try:
        llm = get_primary_llm()
        response = llm.invoke(messages)
        return {
            "messages": [AIMessage(content=response.content)],
            "model_used": "openrouter"
        }
    except Exception as e:
        print(f"[Generate] OpenRouter failed: {e}. Returning raw context.")
        error_msg = f"LLM unavailable. Here is the raw context:\n\n{context}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "model_used": "none"
        }