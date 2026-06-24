"""Search client abstraction for ResearcherAgent."""

import logging
import os

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# Diverse mock corpus covering all README2 topics
_MOCK_CORPUS: list[dict] = [
    # GraphRAG / RAG
    {
        "title": "GraphRAG: Unlocking LLM discovery on narrative private data",
        "url": "https://arxiv.org/abs/2404.16130",
        "snippet": (
            "GraphRAG is a structured, hierarchical approach to retrieval-augmented generation "
            "that uses LLM-derived knowledge graphs to improve Q&A over private text corpora. "
            "It outperforms baseline RAG on complex multi-hop questions."
        ),
        "tags": ["graphrag", "rag", "knowledge graph", "retrieval"],
    },
    # Multi-agent vs single-agent
    {
        "title": "Do multi-agent LLM systems outperform single agents? An empirical study",
        "url": "https://arxiv.org/abs/2402.01680",
        "snippet": (
            "Empirical comparison across 10 benchmarks shows multi-agent systems outperform "
            "single agents on tasks requiring parallel subgoals or specialised roles, but "
            "underperform on tasks where communication overhead exceeds task complexity. "
            "Key finding: gains often come from extra inference compute, not coordination per se."
        ),
        "tags": ["multi-agent", "single-agent", "benchmark", "comparison", "outperform"],
    },
    {
        "title": "Scaling LLM inference with multi-agent debate",
        "url": "https://arxiv.org/abs/2305.14325",
        "snippet": (
            "Multi-agent debate improves factual accuracy by 15% on MMLU vs single-agent. "
            "Multiple agents propose and critique answers over rounds. However, the gains "
            "largely disappear when giving the single agent an equivalent token budget for "
            "self-reflection, suggesting token budget—not coordination—drives improvements."
        ),
        "tags": ["multi-agent", "debate", "single-agent", "accuracy", "token budget"],
    },
    # LangGraph / workflow
    {
        "title": "LangGraph: Building stateful, multi-actor applications with LLMs",
        "url": "https://langchain-ai.github.io/langgraph/",
        "snippet": (
            "LangGraph models agent workflows as directed graphs. Nodes are callables; edges "
            "can be conditional. State is passed between nodes, enabling memory, retries, and "
            "human-in-the-loop checkpointing."
        ),
        "tags": ["langgraph", "workflow", "graph", "agent", "state"],
    },
    # Guardrails / production
    {
        "title": "Building effective agents — Anthropic",
        "url": "https://www.anthropic.com/engineering/building-effective-agents",
        "snippet": (
            "Agentic systems range from simple pipelines to autonomous multi-agent networks. "
            "Key guardrails: limit tool scope, add human-in-the-loop for irreversible actions, "
            "use prompt caching, and prefer simple architectures when possible."
        ),
        "tags": ["guardrail", "production", "agent", "safety", "anthropic"],
    },
    # ReAct
    {
        "title": "ReAct: Synergizing reasoning and acting in language models",
        "url": "https://arxiv.org/abs/2210.03629",
        "snippet": (
            "ReAct interleaves chain-of-thought reasoning with tool-use actions. "
            "Combining reasoning traces and action plans improves factuality and reduces "
            "hallucination compared to action-only or reasoning-only baselines."
        ),
        "tags": ["react", "reasoning", "acting", "tool-use", "chain-of-thought"],
    },
    # Production failures
    {
        "title": "Agents in production: lessons from Cognition, Adept, and others",
        "url": "https://www.latent.space/p/agents-in-production",
        "snippet": (
            "Production agent deployments highlight four failure modes: tool misconfiguration, "
            "context window overflow, irreversible side-effects, and cascading LLM errors. "
            "Multi-agent architectures help by isolating responsibility per sub-task."
        ),
        "tags": ["production", "failure", "agent", "guardrail", "deployment"],
    },
    # Benchmark design
    {
        "title": "BenchAgent: A comprehensive benchmark for LLM agent evaluation",
        "url": "https://arxiv.org/abs/2403.12345",
        "snippet": (
            "BenchAgent covers 8 task categories: planning, tool use, code generation, "
            "research synthesis, multi-step reasoning, adversarial robustness, safety, "
            "and memory. Evaluation includes both automatic metrics and human rubrics. "
            "Human evaluation protocol uses a 5-point Likert scale with blind grading."
        ),
        "tags": ["benchmark", "evaluation", "rubric", "human evaluation", "design"],
    },
    # Customer support
    {
        "title": "Multi-agent customer support: case study at scale",
        "url": "https://arxiv.org/abs/2404.09876",
        "snippet": (
            "A three-agent pipeline (intent classifier, knowledge retriever, response writer) "
            "outperforms a single GPT-4 agent on customer support by 18% CSAT. "
            "The specialist routing reduces hallucinations and speeds up resolution. "
            "Latency overhead is 1.4x; cost overhead is 2.1x per query."
        ),
        "tags": ["customer support", "multi-agent", "single-agent", "comparison", "csat"],
    },
    # Survey / research assistance
    {
        "title": "AI agents for research assistance: a survey",
        "url": "https://arxiv.org/abs/2405.11111",
        "snippet": (
            "Survey of 45 papers on LLM-based research assistants covering literature review, "
            "hypothesis generation, experiment design, and writing. Key gap: no standard "
            "benchmark distinguishes useful assistance from fluent but wrong summaries. "
            "Open problems: evaluation, long-context, citation grounding."
        ),
        "tags": ["survey", "research", "benchmark", "evaluation", "literature"],
    },
]


def _score_relevance(source: dict, query: str) -> int:
    """Simple keyword overlap score between source tags/title/snippet and query."""
    q_lower = query.lower()
    score = 0
    for tag in source.get("tags", []):
        if tag in q_lower:
            score += 3
    for word in source["title"].lower().split():
        if len(word) > 4 and word in q_lower:
            score += 1
    for word in source["snippet"].lower().split():
        if len(word) > 5 and word in q_lower:
            score += 1
    return score


class SearchClient:
    """Search client with Tavily integration and relevance-ranked mock fallback."""

    def __init__(self) -> None:
        self._tavily_api_key = os.getenv("TAVILY_API_KEY")
        self._tavily_client = None
        if self._tavily_api_key:
            try:
                from tavily import TavilyClient  # type: ignore[import-untyped]
                self._tavily_client = TavilyClient(api_key=self._tavily_api_key)
                logger.info("SearchClient: using Tavily")
            except ImportError:
                logger.warning("tavily-python not installed - falling back to mock search")
        else:
            logger.info("SearchClient: TAVILY_API_KEY not set - using mock search")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search documents relevant to query, with mock fallback."""
        if self._tavily_client is not None:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            result = self._tavily_client.search(query=query, max_results=max_results)  # type: ignore[union-attr]
            docs = []
            for r in result.get("results", []):
                docs.append(SourceDocument(
                    title=r.get("title", "Untitled"),
                    url=r.get("url"),
                    snippet=r.get("content", ""),
                ))
            logger.debug("Tavily returned %d results for: %s", len(docs), query)
            return docs
        except Exception as exc:
            logger.warning("Tavily search failed (%s) - falling back to mock", exc)
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        logger.info("Mock search for: %s", query)
        ranked = sorted(_MOCK_CORPUS, key=lambda s: _score_relevance(s, query), reverse=True)
        return [
            SourceDocument(title=s["title"], url=s["url"], snippet=s["snippet"])
            for s in ranked[:max_results]
        ]
