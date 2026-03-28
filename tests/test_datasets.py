"""Tests for canopy.datasets — registry and PersonaMem loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from canopy.builder import BehavioralObservation
from canopy.datasets import load_observations
from canopy.datasets.personamem import (
    PersonaMemQuestion,
    _extract_topic_from_system,
    _messages_to_observations,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_observations("nonexistent")

    def test_personamem_dispatch(self) -> None:
        with patch("canopy.datasets.personamem.load_personamem_observations") as mock:
            mock.return_value = {"user_0": []}
            result = load_observations("personamem", size="32k", persona_id=0)
            assert result == {"user_0": []}
            mock.assert_called_once_with(size="32k", persona_id=0)

    def test_cdt_paper_dispatch(self) -> None:
        with patch("canopy.datasets._cdt_paper.load_cdt_paper_observations") as mock:
            mock.return_value = {"Kasumi": []}
            result = load_observations("cdt_paper", character="Kasumi")
            assert result == {"Kasumi": []}
            mock.assert_called_once_with(character="Kasumi")


# ---------------------------------------------------------------------------
# PersonaMem topic extraction
# ---------------------------------------------------------------------------


class TestExtractTopic:
    def test_extracts_known_topic(self) -> None:
        content = "Some persona info\nTopic: musicRecommendation\nMore text"
        assert _extract_topic_from_system(content) == "musicRecommendation"

    def test_returns_none_for_unknown(self) -> None:
        content = "Topic: unknownTopic\nSomething"
        assert _extract_topic_from_system(content) is None

    def test_returns_none_for_no_topic(self) -> None:
        content = "Just a persona description with no topic line"
        assert _extract_topic_from_system(content) is None

    def test_case_insensitive_key(self) -> None:
        content = "topic: therapy"
        assert _extract_topic_from_system(content) == "therapy"


# ---------------------------------------------------------------------------
# Message → Observation conversion
# ---------------------------------------------------------------------------


class TestMessagesToObservations:
    def _make_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "Persona info\nTopic: therapy"},
            {"role": "user", "content": "User: I've been feeling stressed lately."},
            {"role": "assistant", "content": "Assistant: I'm sorry to hear that."},
            {"role": "user", "content": "User: It started last month."},
            {"role": "assistant", "content": "Assistant: Can you tell me more?"},
            {"role": "system", "content": "Topic: foodRecommendation"},
            {"role": "user", "content": "What's a good comfort food?"},
        ]

    def test_produces_user_observations_only(self) -> None:
        obs = _messages_to_observations(self._make_messages(), persona_id=5)
        assert len(obs) == 3  # 3 user messages
        assert all(isinstance(o, BehavioralObservation) for o in obs)
        assert all(o.actor == "user_5" for o in obs)

    def test_strips_user_prefix(self) -> None:
        obs = _messages_to_observations(self._make_messages(), persona_id=0)
        assert obs[0].action == "I've been feeling stressed lately."
        assert obs[2].action == "What's a good comfort food?"

    def test_tracks_topic_from_system(self) -> None:
        obs = _messages_to_observations(self._make_messages(), persona_id=0)
        assert obs[0].metadata["topic"] == "therapy"
        assert obs[1].metadata["topic"] == "therapy"
        assert obs[2].metadata["topic"] == "foodRecommendation"

    def test_scene_contains_preceding_turns(self) -> None:
        obs = _messages_to_observations(self._make_messages(), persona_id=0, scene_window=4)
        # Third user message should have preceding user+assistant turns in scene
        assert "stressed" in obs[1].scene
        assert "sorry" in obs[1].scene

    def test_participants_always_assistant(self) -> None:
        obs = _messages_to_observations(self._make_messages(), persona_id=0)
        assert all(o.participants == ("assistant",) for o in obs)

    def test_metadata_includes_dataset_and_index(self) -> None:
        obs = _messages_to_observations(self._make_messages(), persona_id=7)
        assert obs[0].metadata["dataset"] == "personamem"
        assert obs[0].metadata["persona_id"] == 7
        assert obs[0].metadata["message_index"] == 1  # index in original messages

    def test_empty_messages(self) -> None:
        obs = _messages_to_observations([], persona_id=0)
        assert obs == []

    def test_system_only_messages(self) -> None:
        msgs = [{"role": "system", "content": "Topic: therapy"}]
        obs = _messages_to_observations(msgs, persona_id=0)
        assert obs == []


# ---------------------------------------------------------------------------
# PersonaMemQuestion dataclass
# ---------------------------------------------------------------------------


class TestPersonaMemQuestion:
    def test_frozen(self) -> None:
        q = PersonaMemQuestion(
            persona_id=0,
            question_id="abc",
            question_type="recall",
            topic="therapy",
            user_message="How am I?",
            correct_answer="(a)",
            all_options=("(a) Good", "(b) Bad"),
            shared_context_id="hash123",
            end_index=42,
            distance_to_ref_tokens=1000,
        )
        assert q.persona_id == 0
        with pytest.raises(AttributeError):
            q.persona_id = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Size validation
# ---------------------------------------------------------------------------


class TestSizeValidation:
    def test_invalid_size_raises(self) -> None:
        from canopy.datasets.personamem import load_personamem_observations

        with pytest.raises(ValueError, match="Invalid size"):
            load_personamem_observations(size="64k")
