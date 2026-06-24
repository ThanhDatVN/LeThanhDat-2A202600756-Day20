"""LLM client abstraction.

Supports three providers:
- openai:  OpenAI API (requires OPENAI_API_KEY)
- ollama:  Local Ollama server via OpenAI-compatible endpoint (LLM_PROVIDER=ollama)
- mock:    Stub for tests / no-key environments

Agents depend only on LLMClient and LLMResponse — never on an SDK directly.
"""

import logging
import os
from dataclasses import dataclass, field

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

import re as _re

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3)."""
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()


# gpt-4o-mini pricing per 1k tokens
_COST_INPUT_PER_1K = 0.000150
_COST_OUTPUT_PER_1K = 0.000600


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _build_openai_client(base_url: str | None = None, api_key: str | None = None, timeout: float = 60.0) -> object:
    """Return an OpenAI client configured for OpenAI or an OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
        kwargs: dict = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
    except ImportError:
        return None
    except Exception:
        # Missing credentials or other init error — will fall back to mock
        return None


@dataclass
class LLMClient:
    """Provider-agnostic LLM client (OpenAI / Ollama / mock)."""

    model: str = field(default="")
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 60.0
    provider: str = field(default="")  # "openai" | "ollama" | "mock"
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve provider + model from env if not explicitly set
        if not self.provider:
            self.provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if self.provider == "ollama":
            if not self.model:
                self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
            # Ollama's OpenAI-compatible endpoint accepts any non-empty string as key
            self._client = _build_openai_client(base_url=base_url, api_key="ollama", timeout=self.timeout)
            if self._client:
                logger.info("LLMClient: provider=ollama model=%s base_url=%s", self.model, base_url)
            else:
                logger.warning("LLMClient: openai package missing - falling back to mock")
                self.provider = "mock"

        elif self.provider == "openai":
            if not self.model:
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
            self._client = _build_openai_client(api_key=api_key, timeout=self.timeout)
            if self._client and api_key:
                logger.info("LLMClient: provider=openai model=%s", self.model)
            else:
                logger.warning("LLMClient: OPENAI_API_KEY not set - falling back to mock")
                self.provider = "mock"

        else:
            self.provider = "mock"
            logger.info("LLMClient: provider=mock")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with automatic retry."""

        if self.provider == "mock" or self._client is None:
            return self._mock_complete(system_prompt, user_prompt)

        logger.debug(
            "LLM call: provider=%s model=%s sys_len=%d usr_len=%d",
            self.provider, self.model, len(system_prompt), len(user_prompt),
        )
        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = _strip_think_tags(response.choices[0].message.content or "")
            in_tok = getattr(response.usage, "prompt_tokens", None)
            out_tok = getattr(response.usage, "completion_tokens", None)
            cost = _estimate_cost(in_tok, out_tok) if (in_tok and out_tok and self.provider == "openai") else 0.0
            logger.debug("LLM done: in=%s out=%s cost=%.6f", in_tok, out_tok, cost or 0)
            return LLMResponse(content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
        except Exception as exc:
            logger.warning("LLM call failed (%s): %s", type(exc).__name__, exc)
            raise

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        preview = user_prompt[:100].replace("\n", " ")
        content = (
            f"[MOCK] provider=mock | query={preview!r}\n"
            "Set LLM_PROVIDER=ollama + OLLAMA_MODEL or set OPENAI_API_KEY for real responses."
        )
        return LLMResponse(content=content, input_tokens=10, output_tokens=20, cost_usd=0.0)


def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float:
    if not input_tokens or not output_tokens:
        return 0.0
    return (input_tokens / 1000) * _COST_INPUT_PER_1K + (output_tokens / 1000) * _COST_OUTPUT_PER_1K


def make_llm_client(model: str = "", temperature: float = 0.0) -> "LLMClient":
    """Factory that picks provider + model from environment.

    When LLM_PROVIDER=ollama, the model is resolved from OLLAMA_MODEL.
    When LLM_PROVIDER=openai, the model is resolved from OPENAI_MODEL.
    Pass an explicit model to override.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if not model:
        if provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "llama3.2")
        else:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return LLMClient(model=model, temperature=temperature, provider=provider)
