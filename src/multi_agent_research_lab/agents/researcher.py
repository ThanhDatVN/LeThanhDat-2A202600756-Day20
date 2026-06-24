"""Researcher agent — fetches sources and writes research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, make_llm_client
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM = """You are a research assistant. Given a query and a set of source documents,
write concise but thorough research notes (300-500 words).

Format:
## Research Notes

<bullet-point key findings with [Source N] citations>

## Sources Used
<list sources actually referenced>

Be factual. Do not hallucinate. If sources are insufficient, say so.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self._llm = llm or make_llm_client(temperature=0.2)
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes."""

        with trace_span("researcher", {"query": state.request.query}) as span:
            logger.info("Researcher: searching for '%s'", state.request.query)
            docs = self._search.search(state.request.query, max_results=state.request.max_sources)
            state.sources = docs

            source_text = "\n\n".join(
                f"[Source {i+1}] {doc.title}\nURL: {doc.url}\n{doc.snippet}"
                for i, doc in enumerate(docs)
            )
            user_prompt = f"Query: {state.request.query}\n\nSources:\n{source_text}"
            state.add_trace_event("researcher_prompt", {
                "system_prompt": _SYSTEM,
                "user_prompt_preview": user_prompt[:300],
            })
            response = self._llm.complete(system_prompt=_SYSTEM, user_prompt=user_prompt)
            state.research_notes = response.content
            span["sources_found"] = len(docs)
            span["tokens"] = response.output_tokens

        state.agent_results.append(AgentResult(
            agent=AgentName.RESEARCHER,
            content=state.research_notes or "",
            metadata={
                "sources_count": len(docs),
                "cost_usd": response.cost_usd,
                "output_tokens": response.output_tokens,
            },
        ))
        state.add_trace_event("researcher_done", {"sources": len(docs)})
        logger.info("Researcher: done - %d sources, notes=%d chars", len(docs), len(state.research_notes or ""))
        return state
