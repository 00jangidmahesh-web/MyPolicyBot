import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Official OpenRouter Free Models Router ID
OPENROUTER_MODEL = "openrouter/free"  # ✅ Using the official free router
TEMPERATURE = 0.1

def get_primary_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=OPENROUTER_MODEL,  # ✅ Updated model ID
        temperature=TEMPERATURE,
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "MyPolicyBot",
        }
    )