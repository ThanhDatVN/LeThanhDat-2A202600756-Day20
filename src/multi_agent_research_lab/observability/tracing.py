"""Tracing hooks.

Provides a lightweight span context manager used by all agents.
Students can augment with LangSmith, Langfuse, or OpenTelemetry by
replacing or wrapping the provider block below.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional LangSmith integration ────────────────────────────────────────────
# If LANGSMITH_API_KEY is set and langsmith is installed, spans are also sent
# to LangSmith as runs. This is a best-effort augmentation — failures are
# logged but never propagate to agents.

def _try_langsmith_trace(name: str, inputs: dict[str, Any]) -> Any:
    """Return a LangSmith run context if available, else None."""
    try:
        import os
        if not os.getenv("LANGSMITH_API_KEY"):
            return None
        from langsmith import traceable  # type: ignore[import-untyped]  # noqa: F401
        # langsmith.traceable is a decorator; for manual spans use RunTree
        from langsmith.run_trees import RunTree  # type: ignore[import-untyped]
        project = os.getenv("LANGSMITH_PROJECT", "multi-agent-research-lab")
        run = RunTree(name=name, run_type="chain", inputs=inputs, project_name=project)
        run.post()
        return run
    except Exception as exc:
        logger.debug("LangSmith trace unavailable: %s", exc)
        return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that measures duration, logs start/end, and optionally traces to LangSmith.

    Usage::

        with trace_span("researcher", {"query": q}) as span:
            ...
            span["sources_found"] = 5  # enrich the span during execution
    """

    attrs = dict(attributes or {})
    span: dict[str, Any] = {"name": name, "attributes": attrs, "duration_seconds": None}
    ls_run = _try_langsmith_trace(name, attrs)

    logger.info("[trace] START %s attrs=%s", name, attrs)
    started = perf_counter()
    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        if ls_run is not None:
            try:
                ls_run.end(error=str(exc))
                ls_run.patch()
            except Exception:
                pass
        logger.error("[trace] ERROR %s: %s", name, exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if ls_run is not None:
            try:
                ls_run.end(outputs=span)
                ls_run.patch()
            except Exception:
                pass
        logger.info("[trace] END %s duration=%.3fs", name, span["duration_seconds"])
