"""LLM adapter interface — provider-agnostic text generation.

Uses claude-agent-sdk with optimized options (tools=[], setting_sources=[])
for fast subprocess startup. Supports parallel generation via generate_many().

Includes batch_generate() for resilient batch LLM calls with drop tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
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
                    wait = 2**attempt
                    log.warning(
                        "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        wait,
                        exc,
                    )
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
# Batch-resilient generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchResult:
    """Result of a batch LLM generation with drop tracking.

    Attributes:
        successes: Immutable mapping of item ID to LLM response for items that succeeded.
        dropped_ids: Frozenset of IDs that were sent but got no response and
            exhausted retries.
        exhausted_ids: Frozenset of IDs that hit max_attempts or were abandoned
            due to no progress in a retry round.
    """

    successes: MappingProxyType[str, str]
    dropped_ids: frozenset[str]
    exhausted_ids: frozenset[str]

    @property
    def all_succeeded(self) -> bool:
        """True if every item received a non-empty response."""
        return len(self.dropped_ids) == 0 and len(self.exhausted_ids) == 0

    @property
    def success_rate(self) -> float:
        """Fraction of items that succeeded (1.0 when empty input)."""
        total = len(self.successes) + len(self.dropped_ids) + len(self.exhausted_ids)
        return len(self.successes) / total if total > 0 else 1.0


def batch_generate(
    items: list[tuple[str, str]],
    *,
    model: str | None = None,
    max_attempts: int = 3,
    adapter: LLMAdapter | None = None,
) -> BatchResult:
    """Generate LLM responses for a batch of (id, prompt) pairs with retry on drops.

    Handles the case where the LLM silently drops items from a batch —
    returning M<N results with no error. Tracks attempts per ID and
    marks items as exhausted after max_attempts.

    Args:
        items: List of (id, prompt) tuples. IDs must be unique.
        model: Model override for generation.
        max_attempts: Max retry attempts per dropped item before marking exhausted.
        adapter: LLM adapter to use. Uses module default if None.

    Returns:
        BatchResult with successes, dropped_ids, and exhausted_ids.

    Raises:
        ValueError: If duplicate IDs are provided.
    """
    if not items:
        return BatchResult(
            successes=MappingProxyType({}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )

    ids = [item_id for item_id, _ in items]
    id_set = set(ids)
    if len(ids) != len(id_set):
        from collections import Counter
        dupes = sorted(id_ for id_, c in Counter(ids).items() if c > 1)
        raise ValueError(f"Duplicate IDs provided: {dupes}")

    used_adapter = adapter if adapter is not None else _default_adapter

    pending: dict[str, str] = {item_id: prompt for item_id, prompt in items}
    attempts: dict[str, int] = {item_id: 0 for item_id in pending}
    successes: dict[str, str] = {}
    exhausted: set[str] = set()

    while pending:
        batch_ids = list(pending.keys())
        batch_prompts = [pending[bid] for bid in batch_ids]

        try:
            responses = used_adapter.generate_many(batch_prompts, model=model)
        except Exception as exc:
            log.warning(
                "batch_generate: generate_many raised %s for %d items: %s",
                type(exc).__name__,
                len(batch_ids),
                exc,
            )
            for bid in batch_ids:
                attempts[bid] += 1
                if attempts[bid] >= max_attempts:
                    exhausted.add(bid)
                    del pending[bid]
            if not pending:
                break
            continue

        resolved_this_round = 0
        for i, bid in enumerate(batch_ids):
            response = responses[i] if i < len(responses) else ""
            if response:  # non-empty string counts as success
                successes[bid] = response
                del pending[bid]
                resolved_this_round += 1
            else:
                attempts[bid] += 1
                if attempts[bid] >= max_attempts:
                    exhausted.add(bid)
                    del pending[bid]

        # No progress — break to avoid infinite loop
        if resolved_this_round == 0 and pending:
            for bid in list(pending.keys()):
                exhausted.add(bid)
                del pending[bid]
            break

    return BatchResult(
        successes=MappingProxyType(successes),
        dropped_ids=frozenset(pending.keys()),
        exhausted_ids=frozenset(exhausted),
    )


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
