"""Canopy — Evolving decision trees for behavioral profiling.

Library usage::

    from canopy import CDTConfig, CDTNode, build_character_cdts
    from canopy.builder import BehavioralObservation, build_cdt, build_character_profile
    from canopy.wikify import wikify_profile, wikify_tree
    from canopy.cluster import KMeansCluster, HDBSCANCluster
    from canopy.llm import batch_generate, BatchResult
"""

__version__ = "0.1.0"

from canopy.builder import BehavioralObservation, build_cdt, build_character_profile
from canopy.core import CDTConfig, CDTNode, build_character_cdts
from canopy.embeddings import EmbeddingCache
from canopy.episodic import EpisodicIndex, GroundingResult, hybrid_ground
from canopy.provenance import HypothesisQuality, Provenance, TrackedHypothesis

__all__ = [
    "BehavioralObservation",
    "CDTConfig",
    "CDTNode",
    "EmbeddingCache",
    "EpisodicIndex",
    "GroundingResult",
    "HypothesisQuality",
    "Provenance",
    "TrackedHypothesis",
    "build_character_cdts",
    "build_character_profile",
    "build_cdt",
    "hybrid_ground",
    "__version__",
]
