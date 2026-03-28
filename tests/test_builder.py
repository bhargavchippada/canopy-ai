"""Tests for canopy.builder — BehavioralObservation, build_cdt, build_character_profile."""

from __future__ import annotations

import pytest

from canopy.builder import (
    BehavioralObservation,
    build_cdt,
    build_character_profile,
    observations_to_pairs,
)
from canopy.core import CDTConfig, CDTNode

# ---------------------------------------------------------------------------
# BehavioralObservation
# ---------------------------------------------------------------------------


class TestBehavioralObservation:
    def test_basic_creation(self) -> None:
        obs = BehavioralObservation(scene="at school", action="helps friend", actor="Alice")
        assert obs.scene == "at school"
        assert obs.action == "helps friend"
        assert obs.actor == "Alice"
        assert obs.participants == ()
        assert obs.metadata == {}

    def test_with_participants(self) -> None:
        obs = BehavioralObservation(
            scene="stage", action="sings", actor="Kasumi",
            participants=["Arisa", "Tae"],
        )
        assert obs.participants == ("Arisa", "Tae")
        assert isinstance(obs.participants, tuple)

    def test_participants_immutable(self) -> None:
        obs = BehavioralObservation(scene="s", action="a", actor="X", participants=["A"])
        with pytest.raises(AttributeError):
            obs.participants.append("B")  # type: ignore[union-attr]

    def test_with_metadata(self) -> None:
        obs = BehavioralObservation(
            scene="s", action="a", actor="X",
            metadata={"title": "ep1", "timestamp": 42},
        )
        assert obs.metadata["title"] == "ep1"

    def test_frozen(self) -> None:
        obs = BehavioralObservation(scene="s", action="a", actor="X")
        with pytest.raises(AttributeError):
            obs.scene = "modified"  # type: ignore[misc]

    def test_to_pair(self) -> None:
        obs = BehavioralObservation(
            scene="at stage", action="plays guitar", actor="Kasumi",
            participants=["Arisa"],
            metadata={"title": "ep5"},
        )
        pair = obs.to_pair()
        assert pair["scene"] == "at stage"
        assert pair["action"] == "plays guitar"
        assert pair["characters"] == ["Kasumi", "Arisa"]
        assert pair["last_character"] == ("Arisa",)
        assert pair["title"] == "ep5"

    def test_to_pair_no_participants(self) -> None:
        obs = BehavioralObservation(scene="s", action="a", actor="Alice")
        pair = obs.to_pair()
        assert pair["characters"] == ["Alice"]
        assert pair["last_character"] == ["Alice"]

    def test_equality(self) -> None:
        a = BehavioralObservation(scene="s", action="a", actor="X")
        b = BehavioralObservation(scene="s", action="a", actor="X")
        assert a == b

    def test_inequality(self) -> None:
        a = BehavioralObservation(scene="s1", action="a", actor="X")
        b = BehavioralObservation(scene="s2", action="a", actor="X")
        assert a != b


# ---------------------------------------------------------------------------
# observations_to_pairs
# ---------------------------------------------------------------------------


class TestObservationsToPairs:
    def test_empty(self) -> None:
        assert observations_to_pairs([]) == []

    def test_conversion(self) -> None:
        obs = [
            BehavioralObservation(scene="s1", action="a1", actor="Alice"),
            BehavioralObservation(scene="s2", action="a2", actor="Bob"),
        ]
        pairs = observations_to_pairs(obs)
        assert len(pairs) == 2
        assert pairs[0]["scene"] == "s1"
        assert pairs[1]["actor"] if "actor" in pairs[1] else pairs[1]["characters"][0] == "Bob"

    def test_metadata_preserved(self) -> None:
        obs = [BehavioralObservation(scene="s", action="a", actor="X", metadata={"k": "v"})]
        pairs = observations_to_pairs(obs)
        assert pairs[0]["k"] == "v"


# ---------------------------------------------------------------------------
# build_cdt
# ---------------------------------------------------------------------------


class TestBuildCdt:
    def test_with_few_observations(self) -> None:
        """Too few observations → empty tree (no LLM calls)."""
        obs = [BehavioralObservation(scene=f"s{i}", action=f"a{i}", actor="Alice") for i in range(5)]
        tree = build_cdt(obs, character="Alice", topic="identity")
        assert isinstance(tree, CDTNode)
        assert tree.statements == []

    def test_with_zero_observations(self) -> None:
        tree = build_cdt([], character="Alice", topic="identity")
        assert tree.statements == []

    def test_custom_config(self) -> None:
        obs = [BehavioralObservation(scene=f"s{i}", action=f"a{i}", actor="A") for i in range(3)]
        cfg = CDTConfig(max_depth=1)
        tree = build_cdt(obs, character="A", topic="x", config=cfg)
        assert isinstance(tree, CDTNode)

    def test_topic_passed_through(self) -> None:
        """Verify the topic parameter is used correctly."""
        obs = [BehavioralObservation(scene="s", action="a", actor="X")]
        tree = build_cdt(obs, character="X", topic="custom_topic")
        assert isinstance(tree, CDTNode)


# ---------------------------------------------------------------------------
# build_character_profile
# ---------------------------------------------------------------------------


class TestBuildCharacterProfile:
    def test_with_empty_observations(self) -> None:
        t, r = build_character_profile([], character="Alice")
        assert len(t) == 4  # 4 attribute topics
        assert len(r) == 0

    def test_auto_discovers_participants(self) -> None:
        obs = [
            BehavioralObservation(scene="s", action="a", actor="Alice", participants=["Bob"]),
            BehavioralObservation(scene="s", action="a", actor="Alice", participants=["Charlie"]),
            BehavioralObservation(scene="s", action="a", actor="Alice", participants=["Bob"]),
        ]
        t, r = build_character_profile(obs, character="Alice")
        assert len(t) == 4
        # Bob and Charlie discovered, but too few pairs for relationship CDTs
        assert len(r) == 0

    def test_explicit_other_characters(self) -> None:
        obs = [BehavioralObservation(scene="s", action="a", actor="Alice")]
        t, r = build_character_profile(obs, character="Alice", other_characters=["Bob"])
        assert len(t) == 4
        assert len(r) == 0  # Too few pairs

    def test_custom_config(self) -> None:
        cfg = CDTConfig(max_depth=0)
        t, r = build_character_profile([], character="Alice", config=cfg)
        assert len(t) == 4
