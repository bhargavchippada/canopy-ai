"""Tests for canopy.prompts — hypothesis generation with batch_generate integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from unittest.mock import patch

import pytest

from canopy.llm import BatchResult
from canopy.prompts import make_hypotheses_batch, parse_hypothesis_response

# ---------------------------------------------------------------------------
# parse_hypothesis_response
# ---------------------------------------------------------------------------


class TestParseHypothesisResponse:
    def test_valid_response(self) -> None:
        response = '{"action_hypotheses": ["stmt1", "stmt2"], "scene_check_hypotheses": ["q1", "q2"]}'
        actions, scenes = parse_hypothesis_response(response)
        assert actions == ["stmt1", "stmt2"]
        assert scenes == ["q1", "q2"]

    def test_mismatched_lengths_truncates(self) -> None:
        response = '{"action_hypotheses": ["a1", "a2", "a3"], "scene_check_hypotheses": ["s1"]}'
        actions, scenes = parse_hypothesis_response(response)
        assert len(actions) == 1
        assert len(scenes) == 1

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            parse_hypothesis_response("plain text with no json")


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
        assert "1/2 clusters dropped" in caplog.text
        assert "50.0%" in caplog.text

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
        assert "2/3 clusters dropped" in caplog.text

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
        assert "3/3 clusters dropped" in caplog.text
        assert "0.0%" in caplog.text

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
