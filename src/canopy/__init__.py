"""Canopy — Evolving decision trees for behavioral profiling.

Quick start::

    from canopy import CDTConfig, CDTNode, build_character_cdts
    from canopy.embeddings import init_models as init_embeddings
    from canopy.validation import init_models as init_validation
    from canopy.llm import set_adapter, ClaudeCodeAdapter
"""

__version__ = "0.1.0"

from canopy.core import CDTConfig, CDTNode, build_character_cdts

__all__ = ["CDTConfig", "CDTNode", "build_character_cdts", "__version__"]
