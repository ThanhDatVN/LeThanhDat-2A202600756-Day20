"""Supervisor / router — decides which worker should run next."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, make_llm_client

logger = logging.getLogger(__name__)

_SYSTEM = """You are a routing supervisor for a research pipeline.
Given the current pipeline state, decide which step to run next.

Available routes:
- researcher: fetch sources and write research notes (run when research_notes is missing)
- analyst:    analyse research notes and extract key insights (run when analysis_notes is missing)
- writer:     write the final answer for the user (run when final_answer is missing)
- done:       all work is complete

Rules:
1. Follow the order: researcher → analyst → writer → done.
2. Skip a step only if its output is already populated.
3. Reply with EXACTLY one word from the list above — nothing else.
"""


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm: LLMClient | None = None) -> None:
        settings = get_settings()
        self._llm = llm or make_llm_client(temperature=0.0)
        self._max_iterations = settings.max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Update state.route_history with the next route."""

        if state.iteration >= self._max_iterations:
            logger.warning("Max iterations (%d) reached - forcing done", self._max_iterations)
            state.record_route("done")
            state.errors.append(f"Stopped: max_iterations={self._max_iterations} reached")
            return state

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"research_notes: {'<populated>' if state.research_notes else '<missing>'}\n"
            f"analysis_notes: {'<populated>' if state.analysis_notes else '<missing>'}\n"
            f"final_answer:   {'<populated>' if state.final_answer else '<missing>'}\n"
            f"errors so far:  {len(state.errors)}\n"
            "What is the next route?"
        )

        state.add_trace_event("supervisor_prompt", {
            "system_prompt": _SYSTEM,
            "user_prompt": user_prompt,
        })
        response = self._llm.complete(system_prompt=_SYSTEM, user_prompt=user_prompt)
        route = response.content.strip().lower().split()[0] if response.content.strip() else "done"

        valid = {"researcher", "analyst", "writer", "done"}
        if route not in valid:
            logger.warning("Supervisor returned invalid route %r - defaulting heuristic", route)
            route = self._heuristic_route(state)

        logger.info("Supervisor -> %s (iteration=%d)", route, state.iteration)
        state.record_route(route)
        state.add_trace_event("supervisor_route", {"route": route, "iteration": state.iteration})
        return state

    def _heuristic_route(self, state: ResearchState) -> str:
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"
