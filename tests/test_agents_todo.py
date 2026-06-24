"""Tests for agent routing and state transitions (post-implementation)."""

from unittest.mock import MagicMock

import pytest

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def _make_state(**kwargs) -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), **kwargs)


def _mock_llm(route: str) -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content=route)
    return llm


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    state = _make_state()
    agent = SupervisorAgent(llm=_mock_llm("researcher"))
    result = agent.run(state)
    assert result.route_history[-1] == "researcher"


def test_supervisor_routes_to_analyst_when_research_done() -> None:
    state = _make_state(research_notes="some notes")
    agent = SupervisorAgent(llm=_mock_llm("analyst"))
    result = agent.run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_analysis_done() -> None:
    state = _make_state(research_notes="notes", analysis_notes="analysis")
    agent = SupervisorAgent(llm=_mock_llm("writer"))
    result = agent.run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_done_when_all_populated() -> None:
    state = _make_state(research_notes="notes", analysis_notes="analysis", final_answer="answer")
    agent = SupervisorAgent(llm=_mock_llm("done"))
    result = agent.run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_stops_at_max_iterations() -> None:
    state = _make_state()
    # Artificially set iteration to max
    state.iteration = 6
    agent = SupervisorAgent(llm=_mock_llm("researcher"))
    result = agent.run(state)
    assert result.route_history[-1] == "done"
    assert any("max_iterations" in e for e in result.errors)


def test_supervisor_falls_back_to_heuristic_on_invalid_llm_output() -> None:
    state = _make_state()
    agent = SupervisorAgent(llm=_mock_llm("INVALID_ROUTE_XYZ"))
    result = agent.run(state)
    # Heuristic: no research_notes → researcher
    assert result.route_history[-1] == "researcher"
