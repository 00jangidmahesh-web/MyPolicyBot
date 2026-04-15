"""State definition for the LangGraph agent."""

from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Main state object that flows through the graph nodes."""
    messages: Annotated[List[BaseMessage], add_messages]   # Chat history
    intents: str                                           # GREETING / INQUIRY / OUT_OF_SCOPE
    context: str                                           # Retrieved text chunks
    sources: List[str]                                     # Source filenames
    model_used: str                                        # Which LLM was used
    prompt_version: str                                    # v1 / v2
    user_info: Dict[str, Any]                              # Extra (optional)