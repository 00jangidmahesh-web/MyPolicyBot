"""
DeepEval evaluation pipeline for NeuraRAG (OpenRouter Free Tier).
Usage:
    python evaluation/evaluate.py --prompt v2
    python evaluation/evaluate.py --compare
    python evaluation/evaluate.py --prompt v2 --no-rerank
"""

import os
import sys
import re
import json
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI

from retrieval.vectorstore import load_vectorstore, vectorstore_exists
from retrieval.loader import load_all_documents
from retrieval.chunker import chunk_documents
from retrieval.vectorstore import build_vectorstore
from core.graph import ask
from core.nodes import init_tool
from evaluation.questions import EVAL_QUESTIONS

load_dotenv()

# ------------------- OpenRouter Judge -------------------
class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="google/gemini-2.0-flash-001"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "NeuraRAG Eval"
        }
        self.sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key, default_headers=self.headers)
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, default_headers=self.headers)

    def load_model(self):
        return self.async_client

    def _clean_json(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(json)?", "", content, flags=re.IGNORECASE)
            content = content.rstrip("`")
        return content.strip()

    def generate(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return self._clean_json(response.choices[0].message.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self.async_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return self._clean_json(response.choices[0].message.content)

    def get_model_name(self):
        return f"OpenRouter ({self.model_name})"

# ------------------- Helper: Vectorstore init -------------------
def get_vectorstore(rebuild=False):
    from retrieval.vectorstore import vectorstore_exists, load_vectorstore, build_vectorstore
    from retrieval.loader import load_all_documents
    from retrieval.chunker import chunk_documents

    if not rebuild and vectorstore_exists():
        return load_vectorstore()
    print("Building vector store from docs/...")
    docs = load_all_documents("docs")   # as per your schema
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)

# ------------------- Build test cases -------------------
def build_test_cases(prompt_version="v2"):
    test_cases = []
    for q in EVAL_QUESTIONS:
        print(f"  Q{q['id']}: {q['question'][:60]}...")
        try:
            result = ask(q["question"], prompt_version)
            tc = LLMTestCase(
                input=q["question"],
                actual_output=result["answer"],
                expected_output=q["ground_truth"],
                retrieval_context=[result.get("context", "")],
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            tc = LLMTestCase(
                input=q["question"],
                actual_output=f"Error: {e}",
                expected_output=q["ground_truth"],
                retrieval_context=[],
            )
        test_cases.append(tc)
    return test_cases

# ------------------- Run evaluation -------------------
def run_evaluation(prompt_version="v2", use_reranking=True):
    print(f"\n{'='*60}")
    print(f"  DEEPEVAL EVALUATION — Prompt: {prompt_version} | Reranking: {use_reranking}")
    print(f"{'='*60}\n")

    # Setup vectorstore and init tool
    store = get_vectorstore()
    init_tool(store, use_reranking)

    print("Running questions through RAG pipeline...\n")
    test_cases = build_test_cases(prompt_version)

    judge = OpenRouterJudge(model_name="google/gemini-2.0-flash-001")
    metrics = [
        AnswerRelevancyMetric(model=judge, threshold=0.6),
        FaithfulnessMetric(model=judge, threshold=0.6),
        ContextualPrecisionMetric(model=judge, threshold=0.6),
        ContextualRecallMetric(model=judge, threshold=0.6),
    ]

    print(f"Running DeepEval using {judge.get_model_name()}...\n")
    results = evaluate(test_cases=test_cases, metrics=metrics)

    print(f"\n{'='*60}")
    print(f"  DEEPEVAL SCORES (Prompt: {prompt_version})")
    print(f"{'='*60}")
    for metric in metrics:
        print(f"  {metric.__class__.__name__:<30} {metric.score:.4f}  {'PASS' if metric.is_successful() else 'FAIL'}")
        if metric.reason:
            print(f"    Reason: {metric.reason[:100]}...")
    print(f"{'='*60}\n")
    return results

def compare_prompts():
    r1 = run_evaluation("v1")
    r2 = run_evaluation("v2")
    print(f"\n{'='*60}")
    print("  V1 vs V2 COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'V1':>8} {'V2':>8}")
    print(f"  {'-'*46}")
    # Extract scores from results objects (they have .metrics attribute)
    metrics_v1 = {m.__class__.__name__: m.score for m in r1.metrics}
    metrics_v2 = {m.__class__.__name__: m.score for m in r2.metrics}
    for metric in metrics_v1:
        print(f"  {metric:<30} {metrics_v1[metric]:>8.4f} {metrics_v2[metric]:>8.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuraRAG DeepEval Evaluation")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_prompts()
    else:
        run_evaluation(args.prompt, not args.no_rerank)