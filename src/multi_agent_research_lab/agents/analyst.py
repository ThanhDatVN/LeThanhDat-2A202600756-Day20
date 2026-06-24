"""Analyst agent — extracts insights from research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, make_llm_client

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert analyst. Given research notes, produce a structured analysis (200-350 words).

Format:
## Key Claims
<numbered list of the most important factual claims>

## Strengths of Evidence
<what the sources do well>

## Gaps / Weak Evidence
<what is missing, disputed, or poorly supported>

## Recommendation
<one sentence: what should the writer emphasise?>

Be critical. Flag any inconsistencies across sources.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or make_llm_client(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.analysis_notes."""

        if not state.research_notes:
            logger.warning("Analyst: no research_notes available - skipping")
            state.analysis_notes = "No research notes to analyse."
            return state

        with trace_span("analyst", {"notes_len": len(state.research_notes)}) as span:
            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}"
            )
            state.add_trace_event("analyst_prompt", {
                "system_prompt": _SYSTEM,
                "user_prompt_preview": user_prompt[:300],
            })
            response = self._llm.complete(system_prompt=_SYSTEM, user_prompt=user_prompt)
            state.analysis_notes = response.content
            span["tokens"] = response.output_tokens

        state.agent_results.append(AgentResult(
            agent=AgentName.ANALYST,
            content=state.analysis_notes or "",
            metadata={"cost_usd": response.cost_usd, "output_tokens": response.output_tokens},
        ))
        state.add_trace_event("analyst_done", {"analysis_len": len(state.analysis_notes or "")})
        logger.info("Analyst: done - analysis=%d chars", len(state.analysis_notes or ""))
        return state
