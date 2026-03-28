"""Tests for canopy.prompts — hypothesis generation with batch_generate integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from unittest.mock import patch

import pytest

from canopy.llm import BatchResult
from canopy.prompts import (
    _extract_python_block,
    _extract_python_list,
    make_hypotheses_batch,
    parse_hypothesis_response,
    summarize_triggers,
)

# ---------------------------------------------------------------------------
# _extract_python_block / _extract_python_list
# ---------------------------------------------------------------------------


class TestExtractPythonBlock:
    def test_extracts_code(self) -> None:
        response = 'Some text\n```python\nx = 1\n```\nMore text'
        assert _extract_python_block(response) == "x = 1\n"

    def test_returns_none_when_missing(self) -> None:
        assert _extract_python_block("no code block here") is None

    def test_ignores_non_python_blocks(self) -> None:
        response = '```json\n{"key": "val"}\n```'
        assert _extract_python_block(response) is None


class TestExtractPythonList:
    def test_simple_list(self) -> None:
        code = 'action_hypotheses = ["stmt1", "stmt2"]'
        result = _extract_python_list(code, "action_hypotheses")
        assert result == ["stmt1", "stmt2"]

    def test_multiline_list(self) -> None:
        code = 'action_hypotheses = [\n    "stmt1",\n    "stmt2",\n]'
        result = _extract_python_list(code, "action_hypotheses")
        assert result == ["stmt1", "stmt2"]

    def test_missing_var_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not find"):
            _extract_python_list("x = 1", "action_hypotheses")

    def test_invalid_syntax_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            _extract_python_list("action_hypotheses = [1 + ]", "action_hypotheses")

    def test_non_list_raises(self) -> None:
        """A string value wrapped in brackets to fool the regex but not ast."""
        with pytest.raises(ValueError, match="Could not find"):
            _extract_python_list('action_hypotheses = "not a list"', "action_hypotheses")

    def test_with_comment(self) -> None:
        code = 'action_hypotheses = ["s1"] # A list of statements'
        result = _extract_python_list(code, "action_hypotheses")
        assert result == ["s1"]


# ---------------------------------------------------------------------------
# parse_hypothesis_response
# ---------------------------------------------------------------------------


class TestParseHypothesisResponse:
    def test_python_code_block(self) -> None:
        response = (
            'Some reasoning\n```python\n'
            'action_hypotheses = ["stmt1", "stmt2"]\n'
            'scene_check_hypotheses = ["q1", "q2"]\n'
            '```'
        )
        actions, scenes = parse_hypothesis_response(response)
        assert actions == ["stmt1", "stmt2"]
        assert scenes == ["q1", "q2"]

    def test_python_code_block_with_comments(self) -> None:
        response = (
            '```python\n'
            'action_hypotheses = ["stmt1"] # A list of statements\n'
            'scene_check_hypotheses = ["q1"] # A list of questions\n'
            '```'
        )
        actions, scenes = parse_hypothesis_response(response)
        assert actions == ["stmt1"]
        assert scenes == ["q1"]

    def test_json_fallback(self) -> None:
        response = '{"action_hypotheses": ["stmt1", "stmt2"], "scene_check_hypotheses": ["q1", "q2"]}'
        actions, scenes = parse_hypothesis_response(response)
        assert actions == ["stmt1", "stmt2"]
        assert scenes == ["q1", "q2"]

    def test_mismatched_lengths_truncates(self) -> None:
        response = '{"action_hypotheses": ["a1", "a2", "a3"], "scene_check_hypotheses": ["s1"]}'
        actions, scenes = parse_hypothesis_response(response)
        assert len(actions) == 1
        assert len(scenes) == 1

    def test_python_block_mismatched_truncates(self) -> None:
        response = (
            '```python\n'
            'action_hypotheses = ["a1", "a2", "a3"]\n'
            'scene_check_hypotheses = ["s1"]\n'
            '```'
        )
        actions, scenes = parse_hypothesis_response(response)
        assert len(actions) == 1
        assert len(scenes) == 1

    def test_no_json_or_python_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            parse_hypothesis_response("plain text with no json")

    def test_malformed_python_falls_back_to_json(self) -> None:
        response = (
            '```python\naction_hypotheses = [broken\n```\n'
            '{"action_hypotheses": ["fallback"], "scene_check_hypotheses": ["q"]}'
        )
        actions, scenes = parse_hypothesis_response(response)
        assert actions == ["fallback"]
        assert scenes == ["q"]


# ---------------------------------------------------------------------------
# make_hypotheses_batch — batch_generate integration
# ---------------------------------------------------------------------------

VALID_JSON_RESPONSE = (
    '{"action_hypotheses": ["Kasumi tends to help others"],'
    ' "scene_check_hypotheses": ["Does the scene involve someone in need?"]}'
)

VALID_JSON_RESPONSE_2 = (
    '{"action_hypotheses": ["Kasumi is cautious"], "scene_check_hypotheses": ["Is there a risky situation?"]}'
)


def _make_clusters(n: int) -> list[list[dict]]:
    """Create n dummy clusters for testing."""
    return [[{"scene": f"scene_{i}", "action": f"action_{i}"}] for i in range(n)]


class TestMakeHypothesesBatchUsesBatchGenerate:
    """Verify make_hypotheses_batch routes through batch_generate."""

    @patch("canopy.prompts.batch_generate")
    def test_successes_extracted(self, mock_bg: object) -> None:
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType(
                {
                    "0": VALID_JSON_RESPONSE,
                    "1": VALID_JSON_RESPONSE_2,
                }
            ),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        clusters = _make_clusters(2)
        stmts, gates = make_hypotheses_batch(
            clusters,
            "Kasumi",
            "identity",
            [],
            [],
        )
        assert "Kasumi tends to help others" in stmts
        assert "Kasumi is cautious" in stmts
        assert "Does the scene involve someone in need?" in gates
        assert "Is there a risky situation?" in gates
        mock_bg.assert_called_once()  # type: ignore[attr-defined]

    @patch("canopy.prompts.batch_generate")
    def test_model_forwarded(self, mock_bg: object) -> None:
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType({"0": VALID_JSON_RESPONSE}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        make_hypotheses_batch(
            _make_clusters(1),
            "Kasumi",
            "identity",
            [],
            [],
            model="claude-sonnet-4-6",
        )
        _, kwargs = mock_bg.call_args  # type: ignore[attr-defined]
        assert kwargs["model"] == "claude-sonnet-4-6"


class TestMakeHypothesesBatchHandlesDrops:
    """Verify partial results and drop warnings."""

    @patch("canopy.prompts.batch_generate")
    def test_partial_results_with_exhausted(self, mock_bg: object, caplog: pytest.LogCaptureFixture) -> None:
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType({"0": VALID_JSON_RESPONSE}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset({"1"}),
        )
        with caplog.at_level(logging.WARNING, logger="canopy.prompts"):
            stmts, gates = make_hypotheses_batch(
                _make_clusters(2),
                "Kasumi",
                "identity",
                [],
                [],
            )
        assert len(stmts) == 1
        assert len(gates) == 1
        assert "1/2 clusters produced no hypotheses" in caplog.text
        assert "llm_drops=1" in caplog.text

    @patch("canopy.prompts.batch_generate")
    def test_partial_results_with_dropped_ids(self, mock_bg: object, caplog: pytest.LogCaptureFixture) -> None:
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType({"0": VALID_JSON_RESPONSE}),
            dropped_ids=frozenset({"1", "2"}),
            exhausted_ids=frozenset(),
        )
        with caplog.at_level(logging.WARNING, logger="canopy.prompts"):
            stmts, gates = make_hypotheses_batch(
                _make_clusters(3),
                "Kasumi",
                "identity",
                [],
                [],
            )
        assert len(stmts) == 1
        assert "2/3 clusters produced no hypotheses" in caplog.text

    @patch("canopy.prompts.batch_generate")
    def test_parsing_failure_skipped(self, mock_bg: object, caplog: pytest.LogCaptureFixture) -> None:
        """A successful LLM response that fails JSON parsing is logged and skipped."""
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType(
                {
                    "0": VALID_JSON_RESPONSE,
                    "1": "not valid json at all",
                }
            ),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        with caplog.at_level(logging.WARNING, logger="canopy.prompts"):
            stmts, gates = make_hypotheses_batch(
                _make_clusters(2),
                "Kasumi",
                "identity",
                [],
                [],
            )
        assert len(stmts) == 1
        assert "Cluster 1 hypothesis parsing failed" in caplog.text


class TestMakeHypothesesBatchAllDropped:
    """Verify empty results when everything is dropped."""

    @patch("canopy.prompts.batch_generate")
    def test_all_exhausted(self, mock_bg: object, caplog: pytest.LogCaptureFixture) -> None:
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType({}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset({"0", "1", "2"}),
        )
        with caplog.at_level(logging.WARNING, logger="canopy.prompts"):
            stmts, gates = make_hypotheses_batch(
                _make_clusters(3),
                "Kasumi",
                "identity",
                [],
                [],
            )
        assert stmts == []
        assert gates == []
        assert "3/3 clusters produced no hypotheses" in caplog.text
        assert "llm_drops=3" in caplog.text

    @patch("canopy.prompts.batch_generate")
    def test_empty_clusters(self, mock_bg: object) -> None:
        mock_bg.return_value = BatchResult(  # type: ignore[attr-defined]
            successes=MappingProxyType({}),
            dropped_ids=frozenset(),
            exhausted_ids=frozenset(),
        )
        stmts, gates = make_hypotheses_batch([], "Kasumi", "identity", [], [])
        assert stmts == []
        assert gates == []


# ---------------------------------------------------------------------------
# summarize_triggers
# ---------------------------------------------------------------------------


class TestSummarizeTriggers:
    def test_passthrough_when_8_or_fewer(self) -> None:
        """Pairs <= 8 pass through without LLM call."""
        gates = ["q1", "q2"]
        stmts = ["s1", "s2"]
        result_gates, result_stmts = summarize_triggers("Kasumi", gates, stmts)
        assert result_gates == gates
        assert result_stmts == stmts

    @patch("canopy.prompts.generate")
    def test_compression_success(self, mock_gen: object) -> None:
        """When >8 pairs, LLM compresses to top 8."""
        import json
        from unittest.mock import MagicMock

        mock_gen = MagicMock(
            return_value=json.dumps(
                {  # type: ignore[assignment]
                    "top8_pairs": [{"scene_check_hypothesis": f"q{i}", "action_hypothesis": f"s{i}"} for i in range(8)]
                }
            )
        )
        with patch("canopy.prompts.generate", mock_gen):
            gates = [f"gate_{i}" for i in range(12)]
            stmts = [f"stmt_{i}" for i in range(12)]
            result_gates, result_stmts = summarize_triggers("Kasumi", gates, stmts)
            assert len(result_gates) == 8
            assert len(result_stmts) == 8
            assert result_gates[0] == "q0"
            mock_gen.assert_called_once()

    @patch("canopy.prompts.generate")
    def test_compression_fallback_on_parse_failure(self, mock_gen: object) -> None:
        """When LLM returns unparseable response, falls back to first 8."""
        from unittest.mock import MagicMock

        mock_gen = MagicMock(return_value="not json at all")  # type: ignore[assignment]
        with patch("canopy.prompts.generate", mock_gen):
            gates = [f"gate_{i}" for i in range(12)]
            stmts = [f"stmt_{i}" for i in range(12)]
            result_gates, result_stmts = summarize_triggers("Kasumi", gates, stmts)
            assert len(result_gates) == 8
            assert result_gates[0] == "gate_0"  # original, not compressed
