"""LangGraph multi-agent workflow.

Graph topology:
  START → supervisor → (researcher | analyst | writer | END)
  Each worker loops back to supervisor after completing its step.

State is passed as a plain dict between nodes and converted from/to
ResearchState at the boundaries.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    return state.model_dump()


def _dict_to_state(d: dict[str, Any]) -> ResearchState:
    return ResearchState.model_validate(d)


class MultiAgentWorkflow:
    """Builds and runs the Supervisor → Worker LangGraph graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    def build(self) -> Any:
        """Compile a LangGraph StateGraph and return the runnable."""
        try:
            from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langgraph is not installed. Run: pip install 'langgraph>=0.2'"
            ) from exc

        graph: StateGraph = StateGraph(dict)

        # ── Node wrappers (with per-agent error recovery) ──────────────
        def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
            s = _dict_to_state(state)
            s = self._supervisor.run(s)
            return _state_to_dict(s)

        def _safe_run(agent_name: str, run_fn: "Any", state: dict[str, Any]) -> dict[str, Any]:
            s = _dict_to_state(state)
            try:
                s = run_fn(s)
            except Exception as exc:
                logger.error("Agent %s failed: %s", agent_name, exc)
                s.errors.append(f"{agent_name} failed: {exc}")
                s.add_trace_event(f"{agent_name}_error", {"error": str(exc)})
            return _state_to_dict(s)

        def researcher_node(state: dict[str, Any]) -> dict[str, Any]:
            return _safe_run("researcher", self._researcher.run, state)

        def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            return _safe_run("analyst", self._analyst.run, state)

        def writer_node(state: dict[str, Any]) -> dict[str, Any]:
            return _safe_run("writer", self._writer.run, state)

        # ── Routing function ───────────────────────────────────────────────
        def route(state: dict[str, Any]) -> str:
            history: list[str] = state.get("route_history", [])
            last = history[-1] if history else "done"
            if last == "done":
                return END
            return last

        # ── Build graph ────────────────────────────────────────────────────
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            route,
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", END: END},
        )

        # Each worker goes back to supervisor
        for worker in ("researcher", "analyst", "writer"):
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph with timeout enforcement, return final ResearchState."""
        compiled = self.build()
        settings = get_settings()
        timeout = settings.timeout_seconds
        logger.info(
            "MultiAgentWorkflow: starting query='%s' timeout=%ds",
            state.request.query, timeout,
        )

        def _invoke() -> dict[str, Any]:
            return compiled.invoke(_state_to_dict(state))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke)
            try:
                raw = future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.error("MultiAgentWorkflow: timed out after %ds", timeout)
                state.errors.append(f"Workflow timed out after {timeout}s")
                state.final_answer = state.final_answer or (
                    f"Workflow timed out after {timeout}s. "
                    f"Partial research notes: {(state.research_notes or '')[:300]}"
                )
                return state

        result = _dict_to_state(raw)
        logger.info(
            "MultiAgentWorkflow: done - iterations=%d routes=%s",
            result.iteration,
            result.route_history,
        )
        return result
