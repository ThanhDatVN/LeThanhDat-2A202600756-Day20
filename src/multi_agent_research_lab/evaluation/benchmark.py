"""Benchmark runner: single-agent vs multi-agent comparison.

Metrics captured:
- latency_seconds  : wall-clock time
- estimated_cost_usd: total LLM token cost (0 for mock/ollama)
- quality_score    : auto-scored 0-10 via LLM judge (optional)
- citation_coverage: fraction of claims with source citations
- failure_rate     : 1 if run raised an exception, else 0
"""

import logging
import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import AgentResult, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_JUDGE_SYSTEM = """You are an impartial quality evaluator for research summaries.

Evaluate ONLY the answer's quality relative to the query. Do NOT penalise for length.

Scoring rubric (0-10):
  9-10: Directly answers the query, well-structured, multiple source citations [Source N], accurate, no hallucinations
  7-8:  Mostly answers the query, minor gaps, at least some citations or clear sourcing
  5-6:  Partially answers, missing key aspects, few or no citations
  3-4:  Off-topic partially or significant inaccuracies
  0-2:  Completely off-topic, empty, or fabricated

Rules:
- An answer WITH citations [Source N] should score at least 1 point higher than an equivalent answer without.
- If the answer is clearly a stub or mock, score 0-1.

Respond with ONLY valid JSON (no explanation outside JSON):
{"score": <integer 0-10>, "reason": "<one sentence>"}
"""


def _count_citations(text: str) -> int:
    """Count [Source N] or [N] style citations in text."""
    return len(re.findall(r"\[(?:Source\s*)?\d+\]", text, re.IGNORECASE))


def _total_cost(results: list[AgentResult]) -> float:
    return sum((r.metadata.get("cost_usd") or 0.0) for r in results)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    quality_judge: "Callable[[str, str], float] | None" = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner, measure latency, cost, quality, and citation coverage."""

    started = perf_counter()
    error_flag = False
    try:
        state = runner(query)
    except Exception as exc:
        logger.error("Benchmark runner failed for %r: %s", run_name, exc)
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        error_flag = True
    latency = perf_counter() - started

    answer = state.final_answer or ""
    cost = _total_cost(state.agent_results)
    citations = _count_citations(answer)
    total_sentences = max(1, len([s for s in answer.split(".") if len(s.strip()) > 10]))
    citation_coverage = min(1.0, citations / total_sentences)

    quality: float | None = None
    if quality_judge and answer:
        try:
            quality = quality_judge(query, answer)
        except Exception as exc:
            logger.warning("Quality judge failed: %s", exc)

    notes = (
        f"citations={citations} coverage={citation_coverage:.0%}"
        + (" | FAILED" if error_flag else "")
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost if cost > 0 else None,
        quality_score=quality,
        notes=notes,
    )
    logger.info(
        "Benchmark %r: latency=%.2fs cost=%.5f citations=%d quality=%s",
        run_name, latency, cost, citations, quality,
    )
    return state, metrics


def make_llm_quality_judge(llm: "object") -> "Callable[[str, str], float]":
    """Return a judge function that scores answer quality via LLM (0-10)."""
    import json as _json

    def judge(query: str, answer: str) -> float:
        user_prompt = f"Query: {query}\n\nAnswer:\n{answer[:2000]}"
        response = llm.complete(system_prompt=_JUDGE_SYSTEM, user_prompt=user_prompt)  # type: ignore[union-attr]
        try:
            data = _json.loads(response.content)
            return float(data["score"])
        except Exception:
            # Try regex fallback
            m = re.search(r'"score"\s*:\s*([0-9.]+)', response.content)
            return float(m.group(1)) if m else 5.0

    return judge
