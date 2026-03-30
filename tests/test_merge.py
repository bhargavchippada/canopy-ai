"""Tests for E1: merge_similar_hypotheses — single-LLM-call merge & dedup."""

from __future__ import annotations

import json

import pytest

from canopy.prompts import merge_similar_hypotheses


class TestMergeSimilarHypotheses:
    def test_single_hypothesis_unchanged(self) -> None:
        gates = ["gate"]
        stmts = ["only one"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert kept_gates == ["gate"]
        assert kept_stmts == ["only one"]

    def test_empty_input(self) -> None:
        kept_gates, kept_stmts = merge_similar_hypotheses([], [])
        assert kept_gates == []
        assert kept_stmts == []

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            merge_similar_hypotheses(["g1"], ["s1", "s2"])

    def test_merges_duplicates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM merges 3 inputs into 2 (combining similar ones)."""
        import canopy.prompts as mod

        def mock_generate(prompt: str, **kwargs: object) -> str:
            return json.dumps({"merged_pairs": [
                {"scene_check_hypothesis": "g_merged", "action_hypothesis": "Kasumi shows enthusiasm"},
                {"scene_check_hypothesis": "g2", "action_hypothesis": "Kasumi is confused"},
            ]})

        monkeypatch.setattr(mod, "generate", mock_generate)

        gates = ["g0", "g1", "g2"]
        stmts = ["Kasumi is enthusiastic", "Kasumi shows enthusiasm vocally", "Kasumi is confused"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 2
        assert "enthusiasm" in kept_stmts[0]
        assert "confused" in kept_stmts[1]

    def test_no_duplicates_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import canopy.prompts as mod

        def mock_generate(prompt: str, **kwargs: object) -> str:
            return json.dumps({"merged_pairs": [
                {"scene_check_hypothesis": "g0", "action_hypothesis": "s0"},
                {"scene_check_hypothesis": "g1", "action_hypothesis": "s1"},
            ]})

        monkeypatch.setattr(mod, "generate", mock_generate)

        gates = ["g0", "g1"]
        stmts = ["s0", "s1"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 2

    def test_fallback_on_llm_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On LLM failure, returns all hypotheses unchanged."""
        import canopy.prompts as mod

        def mock_generate(prompt: str, **kwargs: object) -> str:
            return "invalid json {"

        monkeypatch.setattr(mod, "generate", mock_generate)

        gates = ["g0", "g1"]
        stmts = ["s0", "s1"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert kept_gates == gates
        assert kept_stmts == stmts

    def test_rejects_llm_producing_more_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If LLM returns MORE pairs than input, fallback to originals."""
        import canopy.prompts as mod

        def mock_generate(prompt: str, **kwargs: object) -> str:
            return json.dumps({"merged_pairs": [
                {"scene_check_hypothesis": "g0", "action_hypothesis": "s0"},
                {"scene_check_hypothesis": "g1", "action_hypothesis": "s1"},
                {"scene_check_hypothesis": "g2", "action_hypothesis": "s2"},
            ]})

        monkeypatch.setattr(mod, "generate", mock_generate)

        gates = ["g0", "g1"]
        stmts = ["s0", "s1"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert kept_gates == gates  # originals returned
        assert kept_stmts == stmts

    def test_fallback_on_empty_merged_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import canopy.prompts as mod

        def mock_generate(prompt: str, **kwargs: object) -> str:
            return json.dumps({"merged_pairs": []})

        monkeypatch.setattr(mod, "generate", mock_generate)

        gates = ["g0", "g1"]
        stmts = ["s0", "s1"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert kept_gates == gates
        assert kept_stmts == stmts

    def test_merged_pairs_quality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM produces improved merged versions, not just selections."""
        import canopy.prompts as mod

        def mock_generate(prompt: str, **kwargs: object) -> str:
            return json.dumps({"merged_pairs": [
                {
                    "scene_check_hypothesis": "Does the scene show emotional overwhelm?",
                    "action_hypothesis": "Kasumi briefly lapses into quiet before shifting to energetic resolve",
                },
            ]})

        monkeypatch.setattr(mod, "generate", mock_generate)

        gates = ["when quiet?", "when subdued?"]
        stmts = [
            "Kasumi may briefly lapse into quiet or helplessness",
            "Kasumi tends to shift from quietness into energetic rebound",
        ]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 1
        assert "quiet" in kept_stmts[0]  # Merged version preserves key concept
