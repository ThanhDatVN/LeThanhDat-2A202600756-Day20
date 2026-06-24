"""Generate HTML benchmark report for README2 Prompt 2."""

import os
import sys

sys.path.insert(0, "src")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_MODEL", "kamekichi128/qwen3-4b-instruct-2507:latest")
os.environ.setdefault("TIMEOUT_SECONDS", "300")

from multi_agent_research_lab.core.config import get_settings
get_settings.cache_clear()

from multi_agent_research_lab.observability.logging import configure_logging
configure_logging("WARNING")

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import make_llm_quality_judge, run_benchmark
from multi_agent_research_lab.evaluation.html_report import RunResult, render_html_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import make_llm_client
from multi_agent_research_lab.services.storage import LocalArtifactStore
from pathlib import Path

QUERY = (
    "Do multi-agent LLM systems actually outperform single-agent systems on complex tasks? "
    "Produce a structured research briefing covering: main claim, literature positions, "
    "evidence for and against, methodological concerns, 3 proposed experiments, "
    "and a balanced final judgment."
)

BASELINE_SYS = (
    "You are a research assistant. Given a query, write a structured, thorough answer "
    "using headings and numbered lists where appropriate. Target ~500 words."
)

llm = make_llm_client(temperature=0.3)
judge = make_llm_quality_judge(make_llm_client(temperature=0.0))


def baseline_runner(q: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=q))
    state.add_trace_event("baseline_prompt", {
        "system_prompt": BASELINE_SYS,
        "user_prompt_preview": q[:400],
    })
    resp = llm.complete(system_prompt=BASELINE_SYS, user_prompt=q)
    state.final_answer = resp.content
    state.add_trace_event("baseline_done", {
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cost_usd": resp.cost_usd,
    })
    return state


def multi_runner(q: str) -> ResearchState:
    return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=q)))


print("=== README2 Prompt 2: Multi-agent vs Single-agent Research Briefing ===")
print(f"Query: {QUERY[:80]}...")
print()

print("[1/2] Running Baseline...", flush=True)
sb, mb = run_benchmark("Baseline", QUERY, baseline_runner, quality_judge=judge)
print(f"      Latency={mb.latency_seconds:.1f}s  Quality={mb.quality_score}/10  {mb.notes}")

print("[2/2] Running Multi-Agent (timeout=300s)...", flush=True)
sm, mm = run_benchmark("Multi-Agent", QUERY, multi_runner, quality_judge=judge)
print(f"      Latency={mm.latency_seconds:.1f}s  Quality={mm.quality_score}/10  {mm.notes}")
print(f"      Routes: {sm.route_history}")
print(f"      Answer length: {len(sm.final_answer or '')} chars")

if sm.errors:
    print(f"      Errors: {sm.errors}")

results = [
    RunResult(name="Baseline (Single-Agent)", query=QUERY, state=sb, metrics=mb),
    RunResult(
        name="Multi-Agent (Supervisor -> Researcher -> Analyst -> Writer)",
        query=QUERY,
        state=sm,
        metrics=mm,
    ),
]

html = render_html_report(results)
path = LocalArtifactStore(root=Path("reports")).write_text("benchmark_report.html", html)
print(f"\nHTML report saved: {path}")
print("Open reports/benchmark_report.html in browser to view.")
