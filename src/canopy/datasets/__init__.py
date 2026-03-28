"""Dataset loaders for canopy CDT construction and evaluation.

Registry pattern — call load_observations() with a dataset name to get
BehavioralObservation lists ready for CDT construction.

Supported datasets:
- 'cdt_paper': Bandori/Fandom RP benchmark (CDT paper, arxiv 2601.10080)
- 'personamem': PersonaMem long-context user memory benchmark (COLM 2025)
"""

from __future__ import annotations

from typing import Any

from canopy.builder import BehavioralObservation


def load_observations(
    name: str,
    **kwargs: Any,
) -> dict[str, list[BehavioralObservation]]:
    """Load observations from a named dataset.

    Args:
        name: Dataset identifier ('cdt_paper' or 'personamem').
        **kwargs: Dataset-specific arguments.

    Returns:
        Dict mapping entity ID (character name or persona_id) to observations.

    Raises:
        ValueError: If dataset name is unknown.
    """
    if name == "cdt_paper":
        from canopy.datasets._cdt_paper import load_cdt_paper_observations

        return load_cdt_paper_observations(**kwargs)
    if name == "personamem":
        from canopy.datasets.personamem import load_personamem_observations

        return load_personamem_observations(**kwargs)
    raise ValueError(f"Unknown dataset: {name!r}. Supported: 'cdt_paper', 'personamem'")
