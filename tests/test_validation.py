"""Unit tests for canopy.validation — guard checks (no GPU needed)."""

from __future__ import annotations

import pytest


class TestUninitializedGuards:
    def test_check_scene_uninitialized(self) -> None:
        import canopy.validation as val

        saved = val._classifier
        val._classifier = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                val.check_scene(["test"], ["question"])
        finally:
            val._classifier = saved

    def test_check_statement_probs_uninitialized(self) -> None:
        import canopy.validation as val

        saved = val._classifier
        val._classifier = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                val.check_statement_probs("A", ["action"], ["stmt"])
        finally:
            val._classifier = saved
