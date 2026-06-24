"""Writer agent — synthesizes final answer from research and analysis notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, make_llm_client

logger = logging.getLogger(__name__)

_SYSTEM = """You are a technical writer producing a final research answer for: {audience}.

Write a clear, well-structured response (approximately 500 words) that:
1. Directly answers the query.
2. Incorporates key findings from the research notes with [Source N] citations.
3. Reflects the analyst's insights — emphasise strengths, acknowledge gaps.
4. Ends with a short "Further Reading" section listing the top 2-3 sources.

Tone: informative, precise, accessible to the target audience.
Do NOT copy-paste research notes verbatim — synthesise them.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or make_llm_client(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.final_answer."""

        with trace_span("writer", {"query": state.request.query}) as span:
            system = _SYSTEM.format(audience=state.request.audience)
            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research Notes:\n{state.research_notes or '(none)'}\n\n"
                f"Analysis:\n{state.analysis_notes or '(none)'}"
            )
            state.add_trace_event("writer_prompt", {
                "system_prompt": system,
                "user_prompt_preview": user_prompt[:300],
            })
            response = self._llm.complete(system_prompt=system, user_prompt=user_prompt)
            state.final_answer = response.content
            span["tokens"] = response.output_tokens

        state.agent_results.append(AgentResult(
            agent=AgentName.WRITER,
            content=state.final_answer or "",
            metadata={"cost_usd": response.cost_usd, "output_tokens": response.output_tokens},
        ))
        state.add_trace_event("writer_done", {"answer_len": len(state.final_answer or "")})
        logger.info("Writer: done - final_answer=%d chars", len(state.final_answer or ""))
        return state
