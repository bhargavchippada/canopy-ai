"""High-level CDT construction API for external callers.

This is the primary entry point for library users::

    from canopy.builder import build_cdt, BehavioralObservation
    from canopy.core import CDTConfig

    observations = [
        BehavioralObservation(scene="...", action="...", actor="Alice"),
        ...
    ]
    tree = build_cdt(observations, character="Alice", topic="identity", config=CDTConfig())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from canopy.core import CDTConfig, CDTNode, build_character_cdts

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehavioralObservation:
    """A single observed behavior — the primary input to CDT construction.

    Domain-agnostic: works for character RP, user profiling, workflow analysis, etc.

    Attributes:
        scene: Context in which the behavior occurred (preceding actions, environment).
        action: The observed behavior/response.
        actor: Who performed the action.
        participants: Other actors present during the observation (immutable tuple).
        metadata: Optional domain-specific metadata (title, source, timestamp, etc.).
    """

    scene: str
    action: str
    actor: str
    participants: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure participants is always a tuple (accept list at construction)
        if isinstance(self.participants, list):
            object.__setattr__(self, "participants", tuple(self.participants))

    def to_pair(self) -> dict[str, Any]:
        """Convert to the legacy dict format used by CDTNode internals.

        Reserved keys (scene, action, characters, last_character) take
        precedence over metadata to prevent accidental overwriting.
        """
        pair: dict[str, Any] = dict(self.metadata)  # metadata first (lower priority)
        pair.update({
            "scene": self.scene,
            "action": self.action,
            "characters": [self.actor, *self.participants],
            "last_character": self.participants or [self.actor],
        })
        return pair


def observations_to_pairs(observations: list[BehavioralObservation]) -> list[dict[str, Any]]:
    """Convert BehavioralObservations to legacy pair dicts for CDTNode."""
    return [obs.to_pair() for obs in observations]


def build_cdt(
    observations: list[BehavioralObservation],
    *,
    character: str,
    topic: str,
    config: CDTConfig | None = None,
) -> CDTNode:
    """Build a single CDT for a character on a specific topic.

    This is the simplest entry point — one topic, one tree.

    Args:
        observations: Behavioral observations to build the tree from.
        character: The character/subject to profile.
        topic: The behavioral aspect to focus on (e.g. "identity", "personality").
        config: CDT construction parameters. Uses defaults if None.

    Returns:
        A CDTNode tree rooted at the given topic.
    """
    pairs = observations_to_pairs(observations)
    cfg = config or CDTConfig()
    log.info("Building CDT for %s / %s (%d observations)", character, topic, len(pairs))
    return CDTNode(character, topic, pairs, config=cfg)


def build_character_profile(
    observations: list[BehavioralObservation],
    *,
    character: str,
    other_characters: list[str] | None = None,
    config: CDTConfig | None = None,
) -> tuple[dict[str, CDTNode], dict[str, CDTNode]]:
    """Build a full character profile — all attribute and relationship CDTs.

    This is the high-level entry point for complete character profiling.

    Args:
        observations: Behavioral observations for the character.
        character: The character/subject to profile.
        other_characters: Other characters for relationship CDTs. If None, extracted from observations.
        config: CDT construction parameters.

    Returns:
        (topic2cdt, rel_topic2cdt) — attribute and relationship CDT dicts.
    """
    pairs = observations_to_pairs(observations)

    if other_characters is None:
        # Extract unique participants from observations
        all_participants: set[str] = set()
        for obs in observations:
            all_participants.update(obs.participants)
        all_participants.discard(character)
        other_characters = sorted(all_participants)

    log.info("Building profile for %s (%d observations, %d relationships)",
             character, len(pairs), len(other_characters))
    return build_character_cdts(character, pairs, other_characters, config)
