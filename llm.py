"""LLM adapter interface — provider-agnostic text generation.

Supports swapping between Claude (Max subscription), Anthropic API, OpenAI, etc.
Currently only ClaudeCodeAdapter is implemented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-4-6"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter Protocol
# ---------------------------------------------------------------------------


class LLMAdapter(Protocol):
    """Abstract interface for LLM text generation."""

    def generate(self, prompt: str, *, model: str | None = None, **kwargs: Any) -> str:
        """Send a prompt and return the text response."""
        ...


# ---------------------------------------------------------------------------
# Claude Code Adapter (Max subscription — no API key needed)
# ---------------------------------------------------------------------------


class ClaudeCodeAdapter:
    """Generate text via claude-agent-sdk (wraps Claude Code CLI).

    Requires Claude Max subscription.  No API key.
    """

    def __init__(self, default_model: str = DEFAULT_MODEL) -> None:
        self._default_model = default_model

    def generate(self, prompt: str, *, model: str | None = None, **kwargs: Any) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        target_model = model or self._default_model

        async def _query() -> str:
            options = ClaudeAgentOptions(
                model=target_model,
                max_turns=1,
                system_prompt="You are a helpful AI assistant. Respond directly to the prompt with no tool use.",
            )
            parts: list[str] = []
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
            return "".join(parts)

        # Safe async execution — handles both sync and nested-loop contexts
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop — use nest_asyncio or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(_query())).result()
        else:
            import anyio
            return anyio.run(_query)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def extract_json(response: str, *, max_retries: int = 0, retry_fn: Any = None) -> Any:
    """Extract and parse a JSON block from an LLM response.

    Looks for ```json ... ``` fenced blocks first, then tries raw JSON parsing.
    """
    # Try fenced JSON block
    matches = re.findall(r"```json(.*?)```", response, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0].strip())
        except json.JSONDecodeError:
            log.warning("JSON decode failed on fenced block, trying raw parse")

    # Try raw JSON (entire response might be JSON)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try finding any JSON object/array in the response
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON found in LLM response (length={len(response)})")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_adapter: LLMAdapter = ClaudeCodeAdapter()


def set_adapter(adapter: LLMAdapter) -> None:
    """Swap the module-level default adapter."""
    global _default_adapter  # noqa: PLW0603
    _default_adapter = adapter


def generate(prompt: str, *, model: str | None = None, **kwargs: Any) -> str:
    """Generate text using the current default adapter."""
    return _default_adapter.generate(prompt, model=model, **kwargs)
