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
EVAL_MODEL = "claude-sonnet-4-6"  # Higher quality for evaluation scoring

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter Protocol
# ---------------------------------------------------------------------------


class LLMAdapter(Protocol):
    """Abstract interface for LLM text generation.

    Note on ``max_tokens``: This is a **post-generation length budget**, not an
    API parameter.  The model generates unconstrained and the result is trimmed
    to approximately ``max_tokens * 4`` characters at the nearest sentence
    boundary.  Not suitable for structured outputs (JSON, code blocks, etc.).
    """

    def generate(self, prompt: str, *, model: str | None = None, max_tokens: int | None = None) -> str: ...
    def generate_many(self, prompts: list[str], *, model: str | None = None, max_tokens: int | None = None) -> list[str]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _apply_token_budget(text: str, max_tokens: int) -> str:
    """Trim *text* to approximately *max_tokens* at the nearest sentence boundary.

    Uses a rough 1 token ≈ 4 characters heuristic.  May over-truncate CJK text.
    """
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text
    truncated = text[:char_limit]
    last_period = truncated.rfind(".")
    return truncated[: last_period + 1] if last_period != -1 else truncated


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

    _DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Respond directly to the prompt."

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        max_concurrent: int = 8,
        max_retries: int = 2,
        reuse_session: bool = False,
        system_prompt: str | None = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._default_model = default_model
        self._timeout = timeout
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._reuse_session = reuse_session
        self._system_prompt = system_prompt
        # Session client (lazy-initialized when reuse_session=True)
        self._client: Any = None
        self._client_lock = asyncio.Lock() if reuse_session else None

    def generate(self, prompt: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
        results = self.generate_many([prompt], model=model, max_tokens=max_tokens)
        return results[0]

    def generate_many(self, prompts: list[str], *, model: str | None = None, max_tokens: int | None = None) -> list[str]:
        target_model = model or self._default_model

        async def _run_all() -> list[str]:
            sem = asyncio.Semaphore(self._max_concurrent)

            async def _single(prompt: str) -> str:
                async with sem:
                    return await self._async_generate_with_retry(prompt, target_model, max_tokens=max_tokens)

            return await asyncio.gather(*[_single(p) for p in prompts])

        return asyncio.run(_run_all())

    async def _async_generate_with_retry(self, prompt: str, model: str, *, max_tokens: int | None = None) -> str:
        """Generate with retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._reuse_session:
                    return await self._session_generate(prompt, model, max_tokens=max_tokens)
                return await self._async_generate(prompt, model, max_tokens=max_tokens)
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

    async def _async_generate(self, prompt: str, model: str, *, max_tokens: int | None = None) -> str:
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
            system_prompt=self._system_prompt,
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
        text = "".join(parts)
        if max_tokens is not None:
            text = _apply_token_budget(text, max_tokens)
        return text

    async def _session_generate(self, prompt: str, model: str, *, max_tokens: int | None = None) -> str:
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
                    system_prompt=self._system_prompt,
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
        text = "".join(parts)
        if max_tokens is not None:
            text = _apply_token_budget(text, max_tokens)
        return text


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

# ---------------------------------------------------------------------------
# Transformers Adapter (local HuggingFace models — Llama, etc.)
# ---------------------------------------------------------------------------


class TransformersAdapter:
    """Generate text using a local HuggingFace model (e.g. Llama-3.1-8B-Instruct).

    Matches the paper's generate_llama() behavior:
    - Appends "\\nAnswer:" to prompts
    - Greedy decoding (do_sample=False)
    - max_new_tokens=64
    - Stops at newline
    - Thread-safe via lock (GPU inference is sequential)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_new_tokens: int = 64,
        load_in_8bit: bool = False,
    ) -> None:
        import threading

        self._model_path = model_path
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._load_in_8bit = load_in_8bit
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use. Thread-safe."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-check after acquiring lock
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Loading local model: %s (8bit=%s)", self._model_path, self._load_in_8bit)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            if self._load_in_8bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    quantization_config=bnb_config,
                    device_map={"": self._device},
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    torch_dtype=torch.float16,
                ).to(self._device)
            log.info("Local model loaded on %s", self._device)

    def generate(self, prompt: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
        """Generate text matching paper's generate_llama() behavior."""
        import torch

        self._ensure_loaded()
        prompt_with_suffix = prompt + "\nAnswer:"
        effective_max = max_tokens if max_tokens is not None else self._max_new_tokens
        with self._lock:
            inputs = self._tokenizer(prompt_with_suffix, return_tensors="pt").to(self._device)
            with torch.no_grad():
                newline_id = self._tokenizer.encode("\n")[-1]
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=effective_max,
                    do_sample=False,
                    eos_token_id=newline_id,
                )
            text = self._tokenizer.decode(
                output[0, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
        return text.strip().split("\n")[0].removesuffix("assistant")

    def generate_many(self, prompts: list[str], *, model: str | None = None, max_tokens: int | None = None) -> list[str]:
        """Generate sequentially (GPU-bound, no parallelism benefit)."""
        return [self.generate(p, model=model, max_tokens=max_tokens) for p in prompts]

    def unload(self) -> None:
        """Free VRAM by unloading model."""
        import gc

        import torch

        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Local model unloaded")


# ---------------------------------------------------------------------------
# Dispatch Adapter (routes by model name)
# ---------------------------------------------------------------------------


class DispatchAdapter:
    """Routes generate() calls to different adapters based on model name.

    Allows using a local model (e.g. Llama) for RP generation while
    keeping Claude for eval scoring, through a single interface.
    """

    def __init__(
        self,
        adapters: dict[str, LLMAdapter],
        default: LLMAdapter | None = None,
    ) -> None:
        """Create a dispatch adapter.

        Args:
            adapters: Map of model name/prefix to adapter.
                Exact match is checked first, then prefix match.
            default: Fallback adapter when no match found.
        """
        self._adapters = adapters
        self._default = default or ClaudeCodeAdapter()

    def _resolve(self, model: str | None) -> tuple[LLMAdapter, str | None]:
        """Find the right adapter for a model string."""
        if model is None:
            return self._default, None
        # Exact match
        if model in self._adapters:
            return self._adapters[model], model
        # Prefix match (e.g. "claude-" matches "claude-sonnet-4-6")
        for prefix, adapter in self._adapters.items():
            if model.startswith(prefix):
                return adapter, model
        return self._default, model

    def generate(self, prompt: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
        adapter, resolved_model = self._resolve(model)
        return adapter.generate(prompt, model=resolved_model, max_tokens=max_tokens)

    def generate_many(self, prompts: list[str], *, model: str | None = None, max_tokens: int | None = None) -> list[str]:
        adapter, resolved_model = self._resolve(model)
        return adapter.generate_many(prompts, model=resolved_model, max_tokens=max_tokens)


_default_adapter: LLMAdapter = ClaudeCodeAdapter()


def set_adapter(adapter: LLMAdapter) -> None:
    """Swap the module-level default adapter."""
    global _default_adapter  # noqa: PLW0603
    _default_adapter = adapter


def generate(prompt: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
    """Generate text using the current default adapter."""
    return _default_adapter.generate(prompt, model=model, max_tokens=max_tokens)


def generate_many(prompts: list[str], *, model: str | None = None, max_tokens: int | None = None) -> list[str]:
    """Generate multiple texts in parallel using the current default adapter."""
    return _default_adapter.generate_many(prompts, model=model, max_tokens=max_tokens)
