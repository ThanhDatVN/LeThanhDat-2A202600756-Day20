"""HTML benchmark report with full answers and per-step trace viewer."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


@dataclass
class RunResult:
    name: str
    query: str
    state: ResearchState
    metrics: BenchmarkMetrics


def render_html_report(results: list[RunResult]) -> str:
    runs_html = "\n".join(_render_run(r) for r in results)
    return _HTML_TEMPLATE.replace("{{RUNS}}", runs_html).replace(
        "{{GENERATED}}", _now()
    )


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _h(text: str) -> str:
    return html.escape(str(text))


def _md_to_html(text: str) -> str:
    """Minimal markdown-to-HTML: bold, headers, bullets, newlines."""
    import re
    text = _h(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^\* (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^- (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^(\d+)\. (.+)$", r"<li>\2</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li>.*</li>(\n|$))+", lambda m: f"<ul>{m.group()}</ul>", text)
    text = re.sub(r"\[Source (\d+)\]", r'<cite class="cite">[Source \1]</cite>', text)
    text = text.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{text}</p>"


_PROMPT_EVENTS = {"supervisor_prompt", "researcher_prompt", "analyst_prompt", "writer_prompt"}

_EVENT_ICON = {
    "supervisor_route": "&#128260;",
    "supervisor_prompt": "&#128203;",
    "researcher_prompt": "&#128203;",
    "analyst_prompt": "&#128203;",
    "writer_prompt": "&#128203;",
    "researcher_done": "&#9989;",
    "analyst_done": "&#9989;",
    "writer_done": "&#9989;",
}


def _render_trace(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return "<p class='muted'>No trace events recorded.</p>"
    rows = []
    for i, event in enumerate(trace):
        name = _h(event.get("name", ""))
        payload = event.get("payload", {})
        icon = _EVENT_ICON.get(event.get("name", ""), "&#128313;")

        if event.get("name") in _PROMPT_EVENTS:
            # Special prompt rendering
            sys_prompt = payload.get("system_prompt", "")
            usr_preview = payload.get("user_prompt_preview") or payload.get("user_prompt", "")
            body_html = f"""
              <div class="prompt-block">
                <div class="prompt-label">System Prompt</div>
                <pre class="prompt-pre">{_h(sys_prompt)}</pre>
                <div class="prompt-label">User Prompt (preview)</div>
                <pre class="prompt-pre">{_h(usr_preview)}</pre>
              </div>"""
        else:
            pretty = json.dumps(payload, ensure_ascii=False, indent=2)
            body_html = f'<pre class="trace-body">{_h(pretty)}</pre>'

        rows.append(f"""
        <div class="trace-event">
          <div class="trace-header" onclick="toggleTrace(this)">
            <span class="trace-num">#{i+1}</span>
            <span class="trace-icon">{icon}</span>
            <span class="trace-name">{name}</span>
            <span class="trace-toggle">&#9660;</span>
          </div>
          <div class="trace-body-wrap">{body_html}</div>
        </div>""")
    return "".join(rows)


def _render_agent_results(state: ResearchState) -> str:
    if not state.agent_results:
        return "<p class='muted'>No agent results.</p>"
    tabs = []
    panels = []
    for i, r in enumerate(state.agent_results):
        active = "active" if i == 0 else ""
        label = r.agent.upper()
        tokens = r.metadata.get("output_tokens", "—")
        cost = r.metadata.get("cost_usd")
        cost_str = f"${cost:.5f}" if cost else "$0"
        tabs.append(
            f'<button class="tab-btn {active}" onclick="switchTab(this,\'agent-{i}\')">'
            f"{label} <small>({tokens} tok · {cost_str})</small></button>"
        )
        panels.append(
            f'<div id="agent-{i}" class="tab-panel {active}">'
            f"{_md_to_html(r.content)}</div>"
        )
    return (
        f'<div class="tab-bar">{"".join(tabs)}</div>'
        f'<div class="tab-content">{"".join(panels)}</div>'
    )


def _render_sources(state: ResearchState) -> str:
    if not state.sources:
        return "<p class='muted'>No sources retrieved.</p>"
    items = []
    for i, src in enumerate(state.sources, 1):
        url_part = f'<a href="{_h(src.url or "#")}" target="_blank">{_h(src.url or "—")}</a>'
        items.append(f"""
        <div class="source-card">
          <div class="source-title">[Source {i}] {_h(src.title)}</div>
          <div class="source-url">{url_part}</div>
          <div class="source-snippet">{_h(src.snippet)}</div>
        </div>""")
    return "".join(items)


def _render_run(r: RunResult) -> str:
    m = r.metrics
    quality = f"{m.quality_score:.1f}/10" if m.quality_score is not None else "—"
    cost = f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd else "$0 (local)"
    notes = _h(m.notes)
    routes = " → ".join(r.state.route_history) if r.state.route_history else "—"
    errors_html = ""
    if r.state.errors:
        errs = "".join(f"<li>{_h(e)}</li>" for e in r.state.errors)
        errors_html = f'<div class="error-box"><strong>Errors / Warnings:</strong><ul>{errs}</ul></div>'

    final_answer_html = _md_to_html(r.state.final_answer or "_No answer generated._")

    return f"""
<section class="run-card" id="run-{_h(r.name).replace(' ','-')}">
  <div class="run-header">
    <h2>{_h(r.name)}</h2>
    <div class="run-meta">
      <span class="badge">&#128337; {m.latency_seconds:.1f}s</span>
      <span class="badge quality">&#11088; {quality}</span>
      <span class="badge">&#128176; {cost}</span>
      <span class="badge muted">{notes}</span>
    </div>
  </div>

  <div class="query-box">
    <strong>Query:</strong> {_h(r.query)}
  </div>

  {errors_html}

  <!-- Route history -->
  <div class="section-label">Route History</div>
  <div class="route-history">{_h(routes)}</div>

  <!-- Final answer -->
  <div class="section-label toggle-section" onclick="toggleSection(this, 'answer-{_h(r.name)}')">
    Final Answer &#9660;
  </div>
  <div id="answer-{_h(r.name)}" class="collapsible open answer-box">
    {final_answer_html}
  </div>

  <!-- Per-agent outputs -->
  <div class="section-label toggle-section" onclick="toggleSection(this, 'agents-{_h(r.name)}')">
    Per-Agent Outputs &#9660;
  </div>
  <div id="agents-{_h(r.name)}" class="collapsible open">
    {_render_agent_results(r.state)}
  </div>

  <!-- Sources -->
  <div class="section-label toggle-section" onclick="toggleSection(this, 'sources-{_h(r.name)}')">
    Sources ({len(r.state.sources)}) &#9660;
  </div>
  <div id="sources-{_h(r.name)}" class="collapsible">
    {_render_sources(r.state)}
  </div>

  <!-- Step trace -->
  <div class="section-label toggle-section" onclick="toggleSection(this, 'trace-{_h(r.name)}')">
    Step Trace ({len(r.state.trace)} events) &#9660;
  </div>
  <div id="trace-{_h(r.name)}" class="collapsible">
    {_render_trace(r.state.trace)}
  </div>
</section>
"""


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Agent Research Lab — Benchmark Report</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d2e; --surface2: #252840;
    --border: #2e3250; --accent: #6c8ef7; --accent2: #56d6a0;
    --text: #e2e4f0; --muted: #6b7280; --danger: #f87171;
    --quality: #fbbf24; --radius: 8px; --font: 'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 15px; line-height: 1.6; }
  a { color: var(--accent); }
  h1 { font-size: 1.8rem; font-weight: 700; }
  h2 { font-size: 1.3rem; font-weight: 600; color: var(--accent); }
  h3 { font-size: 1.05rem; font-weight: 600; margin: 0.8em 0 0.3em; }
  p  { margin: 0.5em 0; }
  ul { padding-left: 1.4em; margin: 0.4em 0; }
  li { margin: 0.2em 0; }
  cite.cite { background: #2a3a5c; color: #93c5fd; padding: 0 4px; border-radius: 3px; font-style: normal; font-size: 0.85em; }

  /* Layout */
  .page-header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 24px 32px; }
  .page-header p { color: var(--muted); margin-top: 4px; font-size: 0.9rem; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px 32px; }

  /* Summary table */
  .summary-table { width: 100%; border-collapse: collapse; margin: 16px 0 32px; }
  .summary-table th { background: var(--surface2); padding: 10px 14px; text-align: left; font-weight: 600; font-size: 0.85rem; color: var(--muted); border-bottom: 1px solid var(--border); }
  .summary-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  .summary-table tr:hover td { background: var(--surface2); }

  /* Run cards */
  .run-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 28px; overflow: hidden; }
  .run-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; padding: 18px 24px; border-bottom: 1px solid var(--border); background: var(--surface2); }
  .run-meta { display: flex; gap: 8px; flex-wrap: wrap; }
  .badge { background: #1e2235; border: 1px solid var(--border); padding: 3px 10px; border-radius: 99px; font-size: 0.82rem; color: var(--muted); }
  .badge.quality { color: var(--quality); border-color: var(--quality); }

  .query-box { padding: 12px 24px; background: #141725; font-size: 0.9rem; color: var(--muted); border-bottom: 1px solid var(--border); }

  .section-label { padding: 10px 24px; background: var(--surface2); border-top: 1px solid var(--border); font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  .toggle-section { cursor: pointer; user-select: none; }
  .toggle-section:hover { color: var(--text); }

  .collapsible { overflow: hidden; max-height: 0; transition: max-height 0.3s ease; }
  .collapsible.open { max-height: 9999px; }

  .answer-box { padding: 20px 24px; }
  .route-history { padding: 10px 24px; font-family: monospace; font-size: 0.9rem; color: var(--accent2); }

  .error-box { margin: 12px 24px; padding: 12px 16px; border: 1px solid var(--danger); border-radius: var(--radius); background: #2a1010; color: var(--danger); font-size: 0.88rem; }

  /* Tabs */
  .tab-bar { display: flex; gap: 4px; padding: 12px 24px 0; flex-wrap: wrap; }
  .tab-btn { background: var(--surface2); border: 1px solid var(--border); border-bottom: none; padding: 6px 14px; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 0.82rem; color: var(--muted); transition: all 0.15s; }
  .tab-btn.active, .tab-btn:hover { background: var(--bg); color: var(--text); }
  .tab-content { padding: 16px 24px 20px; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Sources */
  .source-card { border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin: 8px 24px; background: var(--surface2); }
  .source-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }
  .source-url { font-size: 0.8rem; margin-bottom: 6px; }
  .source-snippet { font-size: 0.85rem; color: var(--muted); }

  /* Trace */
  .trace-event { border-bottom: 1px solid var(--border); }
  .trace-header { display: flex; align-items: center; gap: 10px; padding: 10px 24px; cursor: pointer; }
  .trace-header:hover { background: var(--surface2); }
  .trace-num { color: var(--muted); font-size: 0.8rem; width: 28px; }
  .trace-name { font-family: monospace; font-size: 0.9rem; color: var(--accent); flex: 1; }
  .trace-toggle { color: var(--muted); font-size: 0.8rem; }
  .trace-body { padding: 8px 24px 12px 62px; font-size: 0.82rem; color: var(--accent2); white-space: pre-wrap; }
  .trace-body-wrap { display: none; }
  .trace-icon { font-size: 0.9rem; }
  .prompt-block { padding: 8px 24px 12px 62px; }
  .prompt-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 8px 0 4px; }
  .prompt-pre { background: #0d0f1a; border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; font-size: 0.8rem; color: #a5b4fc; white-space: pre-wrap; overflow-x: auto; max-height: 280px; overflow-y: auto; }

  .muted { color: var(--muted); }
  strong { color: var(--text); }
</style>
</head>
<body>

<div class="page-header">
  <h1>Multi-Agent Research Lab — Benchmark Report</h1>
  <p>Generated: {{GENERATED}} &nbsp;|&nbsp; Model: Ollama (local) &nbsp;|&nbsp; Search: Mock</p>
</div>

<div class="container">

  <h2 style="margin-bottom:12px">Summary</h2>
  <table class="summary-table">
    <thead>
      <tr><th>Run</th><th>Latency (s)</th><th>Quality /10</th><th>Cost (USD)</th><th>Citations</th><th>Coverage</th></tr>
    </thead>
    <tbody id="summary-body"></tbody>
  </table>

  {{RUNS}}

</div>

<script>
// Populate summary from run cards
(function() {
  const rows = document.querySelectorAll('.run-card');
  const tbody = document.getElementById('summary-body');
  rows.forEach(card => {
    const name = card.querySelector('h2').textContent;
    const badges = card.querySelectorAll('.badge');
    const latency = badges[0]?.textContent.replace(/[^0-9.]/g,'') + 's';
    const quality = badges[1]?.textContent.replace(/[^0-9./]/g,'');
    const cost    = badges[2]?.textContent.trim().replace(/^[^$]*/, '');
    const notes   = badges[3]?.textContent.trim() || '';
    const citeMatch = notes.match(/citations=([0-9]+)/);
    const coverMatch = notes.match(/coverage=([0-9]+%)/);
    tbody.innerHTML += `<tr>
      <td><a href="#run-${name.replace(/[ ]+/g,'-')}">${name}</a></td>
      <td>${latency}</td>
      <td style="color:#fbbf24">${quality}</td>
      <td>${cost}</td>
      <td>${citeMatch ? citeMatch[1] : '—'}</td>
      <td>${coverMatch ? coverMatch[1] : '—'}</td>
    </tr>`;
  });
})();

function toggleSection(btn, id) {
  const el = document.getElementById(id);
  el.classList.toggle('open');
}

function switchTab(btn, panelId) {
  const bar = btn.closest('.run-card');
  bar.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  bar.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(panelId).classList.add('active');
}

function toggleTrace(header) {
  const wrap = header.nextElementSibling;
  wrap.style.display = wrap.style.display === 'block' ? 'none' : 'block';
}
</script>
</body>
</html>
"""
