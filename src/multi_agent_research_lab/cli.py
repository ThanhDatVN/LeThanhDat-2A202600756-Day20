"""Command-line entrypoint for the lab starter."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark, make_llm_quality_judge
from multi_agent_research_lab.evaluation.html_report import RunResult, render_html_report
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient, make_llm_client
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ── Baseline (single-agent) ────────────────────────────────────────────────────

_BASELINE_SYSTEM = """You are a research assistant. Given a query, write a thorough
research summary (~500 words) that answers the question directly.
Include key concepts, comparisons where relevant, and a short further-reading list.
"""


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call, no search)."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = make_llm_client(temperature=0.3)
    console.print(f"[dim]Provider: {llm.provider} | Model: {llm.model}[/dim]")

    t0 = time.perf_counter()
    response = llm.complete(system_prompt=_BASELINE_SYSTEM, user_prompt=query)
    latency = time.perf_counter() - t0

    state.final_answer = response.content
    state.add_trace_event("baseline_done", {
        "latency_seconds": round(latency, 3),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    })

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(
        f"[green]Latency:[/green] {latency:.2f}s  "
        f"[green]Tokens:[/green] {response.input_tokens}->{response.output_tokens}  "
        f"[green]Cost:[/green] ${response.cost_usd or 0:.5f}"
    )


# ── Multi-agent workflow ───────────────────────────────────────────────────────

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit full JSON state")] = False,
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()

    # Show provider info using a temporary client (no actual call)
    _probe = make_llm_client()
    console.print(f"[dim]Provider: {_probe.provider} | Model: {_probe.model}[/dim]")

    t0 = time.perf_counter()
    result = workflow.run(state)
    latency = time.perf_counter() - t0

    if json_out:
        console.print(result.model_dump_json(indent=2))
        return

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Result"))

    total_cost = sum(
        (r.metadata.get("cost_usd") or 0) for r in result.agent_results
    )
    table = Table(title="Agent Summary", show_header=True)
    table.add_column("Agent")
    table.add_column("Output tokens", justify="right")
    table.add_column("Cost (USD)", justify="right")
    for r in result.agent_results:
        table.add_row(
            r.agent,
            str(r.metadata.get("output_tokens", "-")),
            f"${r.metadata.get('cost_usd', 0) or 0:.5f}",
        )
    console.print(table)
    console.print(
        f"[green]Total latency:[/green] {latency:.2f}s  "
        f"[green]Iterations:[/green] {result.iteration}  "
        f"[green]Total cost:[/green] ${total_cost:.5f}"
    )

    if result.errors:
        console.print(f"[yellow]Warnings:[/yellow] {result.errors}")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    queries_file: Annotated[str | None, typer.Option("--queries", help="JSON file with query list")] = None,
    out_dir: Annotated[str, typer.Option("--out", help="Output directory for report")] = "reports",
) -> None:
    """Run benchmark comparing single-agent baseline vs multi-agent workflow."""

    _init()

    default_queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]
    queries: list[str] = default_queries
    if queries_file:
        with open(queries_file) as f:
            queries = json.load(f)

    console.print(f"[bold]Benchmark:[/bold] {len(queries)} queries × 2 runs")

    llm = make_llm_client(temperature=0.3)

    def _baseline_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        resp = llm.complete(system_prompt=_BASELINE_SYSTEM, user_prompt=q)
        state.final_answer = resp.content
        state.agent_results = []
        return state

    def _multi_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(state)

    all_metrics = []
    all_results: list[RunResult] = []
    store = LocalArtifactStore(root=Path(out_dir))
    judge = make_llm_quality_judge(make_llm_client(temperature=0.0))

    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold cyan]Query {i}/{len(queries)}:[/bold cyan] {query[:80]}")

        s_single, m_single = run_benchmark(f"baseline-q{i}", query, _baseline_runner, quality_judge=judge)
        console.print(f"  baseline:    {m_single.latency_seconds:.2f}s  quality={m_single.quality_score}/10")
        all_results.append(RunResult(name=f"Baseline Q{i}", query=query, state=s_single, metrics=m_single))

        s_multi, m_multi = run_benchmark(f"multi-agent-q{i}", query, _multi_runner, quality_judge=judge)
        console.print(f"  multi-agent: {m_multi.latency_seconds:.2f}s  quality={m_multi.quality_score}/10")
        all_results.append(RunResult(name=f"Multi-Agent Q{i}", query=query, state=s_multi, metrics=m_multi))

        all_metrics.extend([m_single, m_multi])

    report_md = render_markdown_report(all_metrics)
    store.write_text("benchmark_report.md", report_md)

    html = render_html_report(all_results)
    html_path = store.write_text("benchmark_report.html", html)
    console.print(f"\n[green]HTML report:[/green] {html_path}")


# ── HTML report (single query) ────────────────────────────────────────────────

@app.command()
def report(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    out_dir: Annotated[str, typer.Option("--out", help="Output directory")] = "reports",
) -> None:
    """Run baseline + multi-agent for one query and generate an HTML report with traces."""

    _init()
    llm = make_llm_client(temperature=0.3)
    judge = make_llm_quality_judge(make_llm_client(temperature=0.0))
    store = LocalArtifactStore(root=Path(out_dir))

    console.print(f"[bold]Running baseline...[/bold]")
    s_b, m_b = run_benchmark(
        "Baseline", query,
        lambda q: _run_baseline(q, llm),
        quality_judge=judge,
    )
    console.print(f"  quality={m_b.quality_score}/10  latency={m_b.latency_seconds:.1f}s")

    console.print(f"[bold]Running multi-agent...[/bold]")
    s_m, m_m = run_benchmark(
        "Multi-Agent", query,
        lambda q: MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=q))),
        quality_judge=judge,
    )
    console.print(f"  quality={m_m.quality_score}/10  latency={m_m.latency_seconds:.1f}s")

    results = [
        RunResult(name="Baseline", query=query, state=s_b, metrics=m_b),
        RunResult(name="Multi-Agent", query=query, state=s_m, metrics=m_m),
    ]
    html_content = render_html_report(results)
    path = store.write_text("report.html", html_content)
    console.print(f"\n[green]Report saved:[/green] {path}")


def _run_baseline(query: str, llm: object) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    state.add_trace_event("baseline_prompt", {
        "system_prompt": _BASELINE_SYSTEM,
        "user_prompt_preview": query[:300],
    })
    resp = llm.complete(system_prompt=_BASELINE_SYSTEM, user_prompt=query)  # type: ignore[union-attr]
    state.final_answer = resp.content
    state.add_trace_event("baseline_done", {
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    })
    return state


# ── Search probe ───────────────────────────────────────────────────────────────

@app.command("search-test")
def search_test(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query")],
) -> None:
    """Quick smoke-test for the search client."""

    _init()
    search = SearchClient()
    docs = search.search(query, max_results=3)
    for i, doc in enumerate(docs, 1):
        console.print(Panel.fit(f"[bold]{doc.title}[/bold]\n{doc.url}\n\n{doc.snippet}", title=f"Source {i}"))


if __name__ == "__main__":
    app()
