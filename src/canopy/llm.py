"""LLM adapter interface — provider-agnostic text generation.

Uses claude-agent-sdk with optimized options (tools=[], setting_sources=[])
for fast subprocess startup. Supports parallel generation via generate_many().
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

DEFAULT_MODEL = "claude-haiku-4-5"
HYPOTHESIS_MODEL = "claude-haiku-4-5"  # Fast + cheap for hypothesis gen
EVAL_MODEL = "claude-sonnet-4-6"  # Quality model for evaluation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter Protocol
# ---------------------------------------------------------------------------


class LLMAdapter(Protocol):
    """Abstract interface for LLM text generation."""

    def generate(self, prompt: str, *, model: str | None = None) -> str: ...
    def generate_many(self, prompts: list[str], *, model: str | None = None) -> list[str]: ...


# ---------------------------------------------------------------------------
# Claude Agent SDK Adapter (Max subscription — no API key needed)
# ---------------------------------------------------------------------------


class ClaudeCodeAdapter:
    """Generate text via claude-agent-sdk (wraps Claude Code CLI).

    Optimizations:
    - tools=[] to skip tool loading
    - setting_sources=[] to skip settings loading
    - generate_many() for parallel calls via asyncio.gather
    - Retry on transient failures
    - Timeout protection
    """

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        max_concurrent: int = 8,
        max_retries: int = 2,
        reuse_session: bool = False,
    ) -> None:
        self._default_model = default_model
        self._timeout = timeout
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._reuse_session = reuse_session
        # Session client (lazy-initialized when reuse_session=True)
        self._client: Any = None
        self._client_lock = asyncio.Lock() if reuse_session else None

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        results = self.generate_many([prompt], model=model)
        return results[0]

    def generate_many(self, prompts: list[str], *, model: str | None = None) -> list[str]:
        target_model = model or self._default_model

        async def _run_all() -> list[str]:
            sem = asyncio.Semaphore(self._max_concurrent)

            async def _single(prompt: str) -> str:
                async with sem:
                    return await self._async_generate_with_retry(prompt, target_model)

            return await asyncio.gather(*[_single(p) for p in prompts])

        return asyncio.run(_run_all())

    async def _async_generate_with_retry(self, prompt: str, model: str) -> str:
        """Generate with retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._reuse_session:
                    return await self._session_generate(prompt, model)
                return await self._async_generate(prompt, model)
            except Exception as exc:
                last_exc = exc
                error_str = str(exc).lower()
                # Retry on transient errors (rate limits, CLI crashes)
                if any(kw in error_str for kw in ("rate_limit", "exit code 1", "timeout", "timed out")):
                    wait = 2 ** attempt
                    log.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s",
                                attempt + 1, self._max_retries + 1, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                raise  # Non-transient error — don't retry
        raise last_exc  # type: ignore[misc]

    async def _async_generate(self, prompt: str, model: str) -> str:
        """Single LLM call via claude-agent-sdk query()."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            model=model,
            max_turns=1,
            system_prompt="You are a helpful AI assistant. Respond directly to the prompt.",
            # tools=[] means no tool invocations are possible.
            # bypassPermissions is safe ONLY because tools=[].
            tools=[],
            permission_mode="bypassPermissions",
            setting_sources=[],
        )

        parts: list[str] = []

        async def _collect() -> None:
            gen = query(prompt=prompt, options=options)
            while True:
                try:
                    message = await gen.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    if "Unknown message type" in str(exc):
                        log.debug("Skipping unknown message type: %s", exc)
                        continue
                    raise
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)

        await asyncio.wait_for(_collect(), timeout=self._timeout)
        return "".join(parts)

    async def _session_generate(self, prompt: str, model: str) -> str:
        """Generate using a persistent ClaudeSDKClient session (DISABLED by default).

        Saves ~750ms subprocess overhead per call by reusing a single process.
        Safe for: validation prompts, cluster labeling, wikification.
        NOT safe for: hypothesis generation (contamination between clusters).

        Enable with: ClaudeCodeAdapter(reuse_session=True)
        """
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, TextBlock

        if self._client_lock is None:
            raise RuntimeError("_session_generate requires reuse_session=True")
        async with self._client_lock:
            if self._client is None:
                options = ClaudeAgentOptions(
                    model=model,
                    max_turns=1,
                    system_prompt="You are a helpful AI assistant. Respond directly to the prompt.",
                    tools=[],
                    permission_mode="bypassPermissions",
                    setting_sources=[],
                )
                self._client = ClaudeSDKClient(options)
                await self._client.connect()

        parts: list[str] = []
        async for message in self._client.send_message(prompt):
            for block in getattr(message, "content", []):
                if isinstance(block, TextBlock):
                    parts.append(block.text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def extract_json(response: str) -> Any:
    """Extract and parse a JSON block from an LLM response.

    Strategy order:
    1. Fenced ```json ... ``` block
    2. Raw JSON parse of entire response
    3. First valid JSON object/array found via json.JSONDecoder
    """
    # Try fenced JSON block
    matches = re.findall(r"```json(.*?)```", response, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0].strip())
        except json.JSONDecodeError:
            log.warning("JSON decode failed on fenced block, trying raw parse")

    # Try raw JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try finding first valid JSON object/array via raw_decode
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch in ("{", "["):
            try:
                value, _ = decoder.raw_decode(response, i)
                return value
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


def generate(prompt: str, *, model: str | None = None) -> str:
    """Generate text using the current default adapter."""
    return _default_adapter.generate(prompt, model=model)


def generate_many(prompts: list[str], *, model: str | None = None) -> list[str]:
    """Generate multiple texts in parallel using the current default adapter."""
    return _default_adapter.generate_many(prompts, model=model)
