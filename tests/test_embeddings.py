"""Unit tests for canopy.embeddings — guard checks (no GPU needed)."""

from __future__ import annotations

import pytest


class TestUninitializedGuards:
    def test_generative_encode_uninitialized(self) -> None:
        import canopy.embeddings as emb

        saved = emb._generator_embedding
        emb._generator_embedding = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                emb.generative_encode(["test"])
        finally:
            emb._generator_embedding = saved

    def test_surface_encode_uninitialized(self) -> None:
        import canopy.embeddings as emb

        saved = emb._surface_embedding
        emb._surface_embedding = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                emb.surface_encode(["test"])
        finally:
            emb._surface_embedding = saved
