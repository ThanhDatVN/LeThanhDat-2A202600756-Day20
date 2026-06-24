"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a Markdown table with summary."""

    lines = [
        "# Benchmark Report",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")

    # ── Pair-wise summary ──────────────────────────────────────────────────
    baseline = [m for m in metrics if m.run_name.startswith("baseline")]
    multi = [m for m in metrics if m.run_name.startswith("multi")]

    if baseline and multi:
        avg_lat_base = sum(m.latency_seconds for m in baseline) / len(baseline)
        avg_lat_multi = sum(m.latency_seconds for m in multi) / len(multi)
        lines += [
            "",
            "## Summary",
            "",
            f"- **Baseline avg latency**: {avg_lat_base:.2f}s",
            f"- **Multi-agent avg latency**: {avg_lat_multi:.2f}s  "
            f"({avg_lat_multi / avg_lat_base * 100:.0f}% of baseline)",
            "",
            "## When to use multi-agent",
            "",
            "- Queries requiring web research + synthesis (multi-step reasoning).",
            "- When citation traceability matters.",
            "- When independent review (analyst) adds quality over raw generation.",
            "",
            "## When NOT to use multi-agent",
            "",
            "- Simple Q&A with low latency requirements.",
            "- Tasks where a single well-prompted LLM call is sufficient.",
            "- When overhead (latency, cost) exceeds the quality gain.",
        ]

    return "\n".join(lines) + "\n"
