"""CDT paper dataset adapter — wraps existing canopy.data for the registry."""

from __future__ import annotations

from canopy.builder import BehavioralObservation
from canopy.data import load_ar_pairs, load_character_metadata


def load_cdt_paper_observations(
    character: str = "Kasumi",
    split: str = "train",
    scene_window: int = 10,
) -> dict[str, list[BehavioralObservation]]:
    """Load CDT paper RP pairs as BehavioralObservation lists.

    Args:
        character: Character name (e.g. 'Kasumi', 'Arisa').
        split: 'train' or 'test'.
        scene_window: Number of preceding actions for scene context.

    Returns:
        Dict with single key (character name) → list of observations.
    """
    _, character2artifact, band2members = load_character_metadata()
    pairs = load_ar_pairs(character, character2artifact, band2members, scene_window=scene_window)

    observations: list[BehavioralObservation] = []
    for pair in pairs[split]:
        observations.append(
            BehavioralObservation(
                scene=pair["scene"],
                action=pair["action"],
                actor=character,
                participants=tuple(pair.get("last_character", [])),
                metadata={
                    "dataset": "cdt_paper",
                    "split": split,
                },
            )
        )

    return {character: observations}
