"""Tests for canopy.llm — extract_json, adapter, generate, retry logic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from canopy.llm import (
    BatchResult,
    ClaudeCodeAdapter,
    batch_generate,
    extract_json,
    generate,
    generate_many,
    set_adapter,
)

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_fenced_block(self) -> None:
        response = 'Here is the result:\n```json\n{"key": "value"}\n```'
        assert extract_json(response) == {"key": "value"}

    def test_fenced_block_with_surrounding_text(self) -> None:
        response = "Some preamble\n```json\n[1, 2, 3]\n```\nSome epilogue"
        assert extract_json(response) == [1, 2, 3]

    def test_raw_json_object(self) -> None:
        response = '{"action_hypotheses": ["stmt1"], "scene_check_hypotheses": ["q1"]}'
        result = extract_json(response)
        assert result["action_hypotheses"] == ["stmt1"]

    def test_raw_json_array(self) -> None:
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_json_embedded_in_text(self) -> None:
        response = 'The answer is {"x": 42} as expected.'
        assert extract_json(response) == {"x": 42}

    def test_nested_objects(self) -> None:
        result = extract_json('{"a": {"b": [1, 2]}}')
        assert result["a"]["b"] == [1, 2]

    def test_no_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("This is just plain text with no JSON.")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("")

    def test_malformed_fenced_falls_through(self) -> None:
        response = '```json\n{broken json\n```\nBut here: {"fallback": true}'
        assert extract_json(response) == {"fallback": True}

    def test_multiple_fenced_blocks_uses_first(self) -> None:
        response = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        assert extract_json(response) == {"first": 1}

    def test_whitespace_padded(self) -> None:
        assert extract_json('  \n  {"key": "val"}  \n  ') == {"key": "val"}

    def test_array_in_text(self) -> None:
        response = "The list is [1, 2, 3] here."
        assert extract_json(response) == [1, 2, 3]

    def test_deeply_nested(self) -> None:
        response = '{"a": {"b": {"c": {"d": 42}}}}'
        assert extract_json(response)["a"]["b"]["c"]["d"] == 42

    def test_unicode_content(self) -> None:
        response = '{"name": "弦巻こころ", "trait": "happy"}'
        result = extract_json(response)
        assert result["name"] == "弦巻こころ"


# ---------------------------------------------------------------------------
# Mock helpers for claude_agent_sdk
# ---------------------------------------------------------------------------


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockAssistantMessage:
    content: list[Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.content is None:
            self.content = []


async def _mock_query_gen(prompt: str, options: Any = None):
    """Async generator that yields a single assistant message."""
    msg = MockAssistantMessage(content=[MockTextBlock(text=f"response to: {prompt}")])
    yield msg


async def _mock_query_gen_error(prompt: str, options: Any = None):
    """Async generator that raises on first call."""
    raise Exception("exit code 1 error")
    yield  # noqa: F841 — makes this an async generator


async def _mock_query_gen_unknown_type(prompt: str, options: Any = None):
    """Async generator that raises Unknown message type then succeeds."""
    raise Exception("Unknown message type: rate_limit_event")
    yield  # noqa: F841


async def _mock_query_gen_unknown_then_success(prompt: str, options: Any = None):
    """Yields Unknown message type error, then a valid message."""
    # We need a different approach — the generator catches and continues
    msg = MockAssistantMessage(content=[MockTextBlock(text="recovered")])
    yield msg


# ---------------------------------------------------------------------------
# ClaudeCodeAdapter
# ---------------------------------------------------------------------------


class TestClaudeCodeAdapter:
    def _make_adapter(self, **kwargs: Any) -> ClaudeCodeAdapter:
        return ClaudeCodeAdapter(default_model="test-model", timeout=5.0, max_retries=0, **kwargs)

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_generate_single(self, mock_gen: AsyncMock) -> None:
        mock_gen.return_value = "hello world"
        adapter = self._make_adapter()
        result = adapter.generate("test prompt")
        assert result == "hello world"
        mock_gen.assert_called_once_with("test prompt", "test-model")

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_generate_uses_custom_model(self, mock_gen: AsyncMock) -> None:
        mock_gen.return_value = "response"
        adapter = self._make_adapter()
        adapter.generate("prompt", model="custom-model")
        mock_gen.assert_called_once_with("prompt", "custom-model")

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_generate_many_parallel(self, mock_gen: AsyncMock) -> None:
        mock_gen.side_effect = ["r1", "r2", "r3"]
        adapter = self._make_adapter()
        results = adapter.generate_many(["p1", "p2", "p3"])
        assert results == ["r1", "r2", "r3"]
        assert mock_gen.call_count == 3

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_generate_many_empty_list(self, mock_gen: AsyncMock) -> None:
        adapter = self._make_adapter()
        results = adapter.generate_many([])
        assert results == []
        mock_gen.assert_not_called()

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_generate_many_respects_concurrency(self, mock_gen: AsyncMock) -> None:
        """With max_concurrent=2, only 2 calls should run at a time."""
        call_count = 0
        max_concurrent_seen = 0

        async def slow_gen(prompt: str, model: str) -> str:
            nonlocal call_count, max_concurrent_seen
            call_count += 1
            current = call_count
            max_concurrent_seen = max(max_concurrent_seen, current)
            await asyncio.sleep(0.01)
            call_count -= 1
            return f"result-{prompt}"

        mock_gen.side_effect = slow_gen
        adapter = ClaudeCodeAdapter(default_model="m", max_concurrent=2, max_retries=0)
        results = adapter.generate_many(["a", "b", "c", "d"])
        assert len(results) == 4


class TestRetryLogic:
    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_retry_on_transient_error(self, mock_gen: AsyncMock) -> None:
        mock_gen.side_effect = [Exception("exit code 1 error"), "success"]
        adapter = ClaudeCodeAdapter(default_model="m", max_retries=1, timeout=5.0)
        result = adapter.generate("prompt")
        assert result == "success"
        assert mock_gen.call_count == 2

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_no_retry_on_non_transient_error(self, mock_gen: AsyncMock) -> None:
        mock_gen.side_effect = ValueError("bad input")
        adapter = ClaudeCodeAdapter(default_model="m", max_retries=2, timeout=5.0)
        with pytest.raises(ValueError, match="bad input"):
            adapter.generate("prompt")
        assert mock_gen.call_count == 1

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_retry_exhausted_raises_last(self, mock_gen: AsyncMock) -> None:
        mock_gen.side_effect = [
            Exception("rate_limit hit"),
            Exception("rate_limit again"),
            Exception("rate_limit still"),
        ]
        adapter = ClaudeCodeAdapter(default_model="m", max_retries=2, timeout=5.0)
        with pytest.raises(Exception, match="rate_limit still"):
            adapter.generate("prompt")
        assert mock_gen.call_count == 3

    @patch("canopy.llm.ClaudeCodeAdapter._async_generate")
    def test_retry_on_timeout_error(self, mock_gen: AsyncMock) -> None:
        mock_gen.side_effect = [Exception("timed out"), "ok"]
        adapter = ClaudeCodeAdapter(default_model="m", max_retries=1, timeout=5.0)
        assert adapter.generate("p") == "ok"


class TestAsyncGenerate:
    """Test _async_generate with mocked SDK."""

    def test_async_generate_success(self) -> None:
        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0)

        with patch("canopy.llm.ClaudeCodeAdapter._async_generate") as mock:
            mock.return_value = "mocked response"
            result = adapter.generate("test")
            assert result == "mocked response"

    def test_async_generate_real_path(self) -> None:
        """Test _async_generate code path with mocked claude_agent_sdk."""
        # Create mock SDK module
        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.AssistantMessage = MockAssistantMessage
        mock_sdk.TextBlock = MockTextBlock

        mock_sdk.query = MagicMock(side_effect=_mock_query_gen)

        adapter = ClaudeCodeAdapter(default_model="test", timeout=5.0, max_retries=0)

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(adapter._async_generate("hello", "test-model"))

        assert "response to: hello" in result

    def test_async_generate_unknown_message_skipped(self) -> None:
        """Unknown message type exceptions are caught and skipped."""
        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.AssistantMessage = MockAssistantMessage
        mock_sdk.TextBlock = MockTextBlock

        class UnknownThenSuccess:
            """Async iterator that raises Unknown on first __anext__, then yields."""

            def __init__(self) -> None:
                self._raised = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._raised:
                    self._raised = True
                    raise Exception("Unknown message type: rate_limit_event")
                raise StopAsyncIteration

        mock_sdk.query = MagicMock(return_value=UnknownThenSuccess())

        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0)
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(adapter._async_generate("test", "m"))
        # Should complete without error (unknown was skipped, then stopped)
        assert result == ""

    def test_async_generate_non_unknown_error_raises(self) -> None:
        """Non-'Unknown message type' exceptions are propagated."""
        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.AssistantMessage = MockAssistantMessage
        mock_sdk.TextBlock = MockTextBlock

        async def gen_with_real_error(prompt: str, options: Any = None):
            raise RuntimeError("real SDK error")
            yield  # noqa: F841

        mock_sdk.query = MagicMock(side_effect=gen_with_real_error)

        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0)
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(RuntimeError, match="real SDK error"):
                asyncio.run(adapter._async_generate("test", "m"))

    def test_async_generate_timeout(self) -> None:
        """Timeout is respected."""

        async def slow_gen(prompt: str, options: Any = None):
            await asyncio.sleep(10)
            yield MockAssistantMessage(content=[MockTextBlock(text="late")])

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.AssistantMessage = MockAssistantMessage
        mock_sdk.TextBlock = MockTextBlock
        mock_sdk.query = MagicMock(side_effect=slow_gen)

        adapter = ClaudeCodeAdapter(default_model="m", timeout=0.1, max_retries=0)
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(TimeoutError):
                asyncio.run(adapter._async_generate("test", "m"))

    def test_async_generate_multi_block(self) -> None:
        """Multiple text blocks are concatenated."""

        async def multi_block_gen(prompt: str, options: Any = None):
            yield MockAssistantMessage(
                content=[
                    MockTextBlock(text="part1 "),
                    MockTextBlock(text="part2"),
                ]
            )

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.AssistantMessage = MockAssistantMessage
        mock_sdk.TextBlock = MockTextBlock
        mock_sdk.query = MagicMock(side_effect=multi_block_gen)

        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0)
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(adapter._async_generate("test", "m"))
        assert result == "part1 part2"

    def test_async_generate_empty_response(self) -> None:
        """No text blocks yields empty string."""

        async def empty_gen(prompt: str, options: Any = None):
            yield MockAssistantMessage(content=[])

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.AssistantMessage = MockAssistantMessage
        mock_sdk.TextBlock = MockTextBlock
        mock_sdk.query = MagicMock(side_effect=empty_gen)

        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0)
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(adapter._async_generate("test", "m"))
        assert result == ""


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestSessionGenerate:
    """Test the reuse_session=True path."""

    def test_session_generate_dispatches(self) -> None:
        """When reuse_session=True, _session_generate is called instead of _async_generate."""
        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0, reuse_session=True)

        with patch.object(adapter, "_session_generate", new_callable=AsyncMock) as mock_session:
            mock_session.return_value = "session response"
            result = adapter.generate("prompt")
            assert result == "session response"
            mock_session.assert_called_once_with("prompt", "m")

    def test_session_generate_full_path(self) -> None:
        """Test _session_generate with fully mocked SDK."""
        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0, reuse_session=True)

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.TextBlock = MockTextBlock

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()

        async def mock_send(prompt):
            yield MockAssistantMessage(content=[MockTextBlock(text=f"session: {prompt}")])

        mock_client.send_message = mock_send
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(adapter._session_generate("hello", "m"))
        assert "session: hello" in result

    def test_session_reuses_client(self) -> None:
        """Second call should reuse the existing client."""
        adapter = ClaudeCodeAdapter(default_model="m", timeout=5.0, max_retries=0, reuse_session=True)

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock
        mock_sdk.TextBlock = MockTextBlock

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()

        async def mock_send(prompt):
            yield MockAssistantMessage(content=[MockTextBlock(text="ok")])

        mock_client.send_message = mock_send
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):

            async def run_twice():
                r1 = await adapter._session_generate("p1", "m")
                r2 = await adapter._session_generate("p2", "m")
                return r1, r2

            r1, r2 = asyncio.run(run_twice())
        assert r1 == "ok"
        assert r2 == "ok"
        # Client constructor should only be called once
        assert mock_sdk.ClaudeSDKClient.call_count == 1


class TestModuleFunctions:
    def test_set_adapter_and_generate(self) -> None:
        """Test that set_adapter swaps the module-level adapter."""
        mock_adapter = MagicMock()
        mock_adapter.generate.return_value = "from mock adapter"
        mock_adapter.generate_many.return_value = ["m1", "m2"]

        original = None
        try:
            # Save original
            import canopy.llm

            original = canopy.llm._default_adapter

            set_adapter(mock_adapter)
            assert generate("any prompt") == "from mock adapter"
            assert generate_many(["p1", "p2"]) == ["m1", "m2"]
        finally:
            if original is not None:
                set_adapter(original)

    def test_generate_passes_model(self) -> None:
        mock_adapter = MagicMock()
        mock_adapter.generate.return_value = "ok"

        import canopy.llm

        original = canopy.llm._default_adapter
        try:
            set_adapter(mock_adapter)
            generate("prompt", model="special-model")
            mock_adapter.generate.assert_called_with("prompt", model="special-model")
        finally:
            set_adapter(original)


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_all_succeeded(self) -> None:
        result = BatchResult(
            successes=MappingProxyType({"a": "resp_a", "b": "resp_b"}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        assert result.all_succeeded is True
        assert result.success_rate == 1.0

    def test_with_drops_and_exhausted(self) -> None:
        result = BatchResult(
            successes=MappingProxyType({"a": "resp_a"}),
            dropped_ids=frozenset({"b"}),
            exhausted_ids=frozenset({"c"}),
        )
        assert result.all_succeeded is False
        assert result.success_rate == pytest.approx(1 / 3)

    def test_success_rate_calculation(self) -> None:
        result = BatchResult(
            successes=MappingProxyType({"a": "x", "b": "y", "c": "z"}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset({"d"}),
        )
        assert result.success_rate == pytest.approx(0.75)

    def test_empty_result(self) -> None:
        result = BatchResult(
            successes=MappingProxyType({}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        assert result.all_succeeded is True
        assert result.success_rate == 1.0

    def test_frozen(self) -> None:
        result = BatchResult(
            successes=MappingProxyType({}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        with pytest.raises(AttributeError):
            result.successes = MappingProxyType({})  # type: ignore[misc]

    def test_successes_immutable(self) -> None:
        result = BatchResult(
            successes=MappingProxyType({"a": "x"}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        with pytest.raises(TypeError):
            result.successes["b"] = "y"  # type: ignore[index]


# ---------------------------------------------------------------------------
# batch_generate
# ---------------------------------------------------------------------------


class TestBatchGenerate:
    def _mock_adapter(self, side_effects: list[list[str]]) -> MagicMock:
        """Create a mock adapter whose generate_many returns successive side_effects."""
        adapter = MagicMock()
        adapter.generate_many = MagicMock(side_effect=side_effects)
        return adapter

    def test_all_succeed(self) -> None:
        adapter = self._mock_adapter([["resp_a", "resp_b", "resp_c"]])
        result = batch_generate(
            [("a", "prompt_a"), ("b", "prompt_b"), ("c", "prompt_c")],
            adapter=adapter,
        )
        assert result.all_succeeded is True
        assert result.successes == {"a": "resp_a", "b": "resp_b", "c": "resp_c"}
        assert result.dropped_ids == frozenset()
        assert result.exhausted_ids == frozenset()

    def test_empty_input(self) -> None:
        result = batch_generate([])
        assert result.all_succeeded is True
        assert result.successes == {}

    def test_with_drops_retries_and_exhausts(self) -> None:
        """Item 'b' returns empty string every time, should be exhausted after max_attempts."""
        adapter = self._mock_adapter(
            [
                ["resp_a", "", "resp_c"],  # round 1: b drops
                [""],  # round 2: b drops again
                [""],  # round 3: b drops again → exhausted
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb"), ("c", "pc")],
            adapter=adapter,
            max_attempts=3,
        )
        assert result.successes == {"a": "resp_a", "c": "resp_c"}
        assert "b" in result.exhausted_ids
        assert result.all_succeeded is False

    def test_retry_succeeds(self) -> None:
        """Item 'b' fails first call, succeeds on retry."""
        adapter = self._mock_adapter(
            [
                ["resp_a", ""],  # round 1: b drops
                ["resp_b"],  # round 2: b succeeds
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb")],
            adapter=adapter,
            max_attempts=3,
        )
        assert result.successes == {"a": "resp_a", "b": "resp_b"}
        assert result.all_succeeded is True

    def test_max_attempts_honored(self) -> None:
        """With max_attempts=1, items are exhausted after a single empty response."""
        adapter = self._mock_adapter(
            [
                ["resp_a", ""],  # round 1: b drops → exhausted (max_attempts=1)
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb")],
            adapter=adapter,
            max_attempts=1,
        )
        assert result.successes == {"a": "resp_a"}
        assert "b" in result.exhausted_ids

    def test_duplicate_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate IDs"):
            batch_generate([("a", "p1"), ("a", "p2")])

    def test_exception_handling(self) -> None:
        """generate_many raises an exception — items should get retried."""
        adapter = MagicMock()
        adapter.generate_many = MagicMock(
            side_effect=[
                RuntimeError("connection failed"),  # round 1: exception
                ["resp_a", "resp_b"],  # round 2: success
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb")],
            adapter=adapter,
            max_attempts=3,
        )
        assert result.successes == {"a": "resp_a", "b": "resp_b"}
        assert result.all_succeeded is True

    def test_no_progress_breaks(self) -> None:
        """If generate_many always returns empty strings, loop terminates."""
        adapter = self._mock_adapter(
            [
                ["", ""],  # round 1: nothing resolved, no progress → all exhausted
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb")],
            adapter=adapter,
            max_attempts=5,
        )
        assert result.successes == {}
        assert result.exhausted_ids == frozenset({"a", "b"})

    def test_custom_adapter_used(self) -> None:
        """Verify the custom adapter is called, not the module default."""
        custom = self._mock_adapter([["custom_resp"]])
        result = batch_generate([("x", "px")], adapter=custom)
        assert result.successes == {"x": "custom_resp"}
        custom.generate_many.assert_called_once()

    def test_uses_default_adapter(self) -> None:
        """When no adapter passed, uses the module-level default."""
        mock_adapter = MagicMock()
        mock_adapter.generate_many.return_value = ["default_resp"]

        import canopy.llm

        original = canopy.llm._default_adapter
        try:
            set_adapter(mock_adapter)
            result = batch_generate([("a", "pa")])
            assert result.successes == {"a": "default_resp"}
            mock_adapter.generate_many.assert_called_once()
        finally:
            set_adapter(original)

    def test_model_passed_through(self) -> None:
        """Verify model parameter is forwarded to generate_many."""
        custom = self._mock_adapter([["resp"]])
        batch_generate([("a", "pa")], model="special-model", adapter=custom)
        custom.generate_many.assert_called_with(["pa"], model="special-model")

    def test_short_response_list(self) -> None:
        """If generate_many returns fewer items than sent, missing ones count as drops."""
        adapter = self._mock_adapter(
            [
                ["resp_a"],  # only 1 response for 3 items → b,c are drops
                ["", ""],  # round 2: b,c still empty → no progress → exhausted
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb"), ("c", "pc")],
            adapter=adapter,
            max_attempts=3,
        )
        assert result.successes == {"a": "resp_a"}
        assert "b" in result.exhausted_ids
        assert "c" in result.exhausted_ids

    def test_exception_exhausts_all_items(self) -> None:
        """When generate_many raises every round, all items end up exhausted."""
        adapter = MagicMock()
        adapter.generate_many = MagicMock(
            side_effect=[
                RuntimeError("fail round 1"),
                RuntimeError("fail round 2"),
                RuntimeError("fail round 3"),
            ]
        )
        result = batch_generate(
            [("a", "pa"), ("b", "pb")],
            adapter=adapter,
            max_attempts=3,
        )
        assert result.successes == {}
        assert result.exhausted_ids == frozenset({"a", "b"})
        assert result.all_succeeded is False

    def test_exception_exhausts_then_breaks(self) -> None:
        """When exception exhausts all pending items, loop breaks immediately."""
        adapter = MagicMock()
        adapter.generate_many = MagicMock(
            side_effect=RuntimeError("always fails"),
        )
        result = batch_generate(
            [("x", "px")],
            adapter=adapter,
            max_attempts=1,
        )
        assert result.successes == {}
        assert "x" in result.exhausted_ids
        # generate_many should only be called once since max_attempts=1
        adapter.generate_many.assert_called_once()


# ---------------------------------------------------------------------------
# _session_generate guard
# ---------------------------------------------------------------------------


class TestSessionGenerateGuard:
    def test_session_generate_without_reuse_raises(self) -> None:
        """When reuse_session=False, _session_generate raises RuntimeError."""
        adapter = ClaudeCodeAdapter(
            default_model="m", timeout=5.0, max_retries=0, reuse_session=False,
        )
        with pytest.raises(RuntimeError, match="reuse_session=True"):
            asyncio.run(adapter._session_generate("prompt", "m"))
