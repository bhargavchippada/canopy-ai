"""Tests for E1: merge_similar_hypotheses."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.prompts import merge_similar_hypotheses


def _mock_embed_fn(dim: int = 32) -> callable:
    """Create a deterministic embed_fn that maps similar texts to similar vectors."""
    cache: dict[str, np.ndarray] = {}

    def embed_fn(text: str) -> np.ndarray:
        if text not in cache:
            # Use hash to get deterministic but spread-out vectors
            seed = hash(text) % (2**31)
            local_rng = np.random.default_rng(seed)
            vec = local_rng.standard_normal(dim).astype(np.float32)
            cache[text] = vec / np.linalg.norm(vec)
        return cache[text]

    return embed_fn


class TestMergeSimilarHypotheses:
    def test_no_duplicates_unchanged(self) -> None:
        embed_fn = _mock_embed_fn()
        gates = ["gate A", "gate B", "gate C"]
        stmts = ["Kasumi sings when happy", "Arisa scolds Kasumi", "Tae plays guitar quietly"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts, embed_fn=embed_fn)
        assert kept_gates == gates
        assert kept_stmts == stmts

    def test_exact_duplicates_merged(self) -> None:
        embed_fn = _mock_embed_fn()
        gates = ["gate A", "gate B"]
        stmts = ["Kasumi tends to sing loudly", "Kasumi tends to sing loudly"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts, embed_fn=embed_fn)
        assert len(kept_stmts) == 1
        assert len(kept_gates) == 1

    def test_different_statements_not_merged(self) -> None:
        embed_fn = _mock_embed_fn()
        gates = ["gate A", "gate B"]
        stmts = [
            "Kasumi sings when the band is struggling",
            "Arisa criticizes Kasumi's impulsive decisions",
        ]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts, embed_fn=embed_fn)
        assert len(kept_stmts) == 2

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

    def test_three_way_merge_exact(self) -> None:
        embed_fn = _mock_embed_fn()
        gates = ["g1", "g2", "g3"]
        stmts = ["same text here", "same text here", "same text here"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts, embed_fn=embed_fn)
        assert len(kept_stmts) == 1

    def test_preserves_gate_statement_alignment(self) -> None:
        """Kept gates and statements remain aligned by index."""
        embed_fn = _mock_embed_fn()
        gates = ["when sad", "when happy", "when sad again"]
        # stmts[0] and stmts[2] are identical → one merged
        stmts = ["Kasumi comforts others", "Kasumi celebrates loudly", "Kasumi comforts others"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts, embed_fn=embed_fn)
        assert len(kept_gates) == len(kept_stmts)
        assert len(kept_stmts) == 2

    def test_llm_fallback_when_no_embed_fn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When embed_fn is None, falls back to LLM-based comparison."""
        import canopy.prompts as prompts_mod

        call_count = 0

        def mock_generate(prompt: str, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            # Say "yes" for identical statements
            if "same text" in prompt and prompt.count("same text") >= 2:
                return "yes"
            return "no"

        monkeypatch.setattr(prompts_mod, "generate", mock_generate)
        gates = ["g1", "g2"]
        stmts = ["same text here", "same text here"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 1
        assert call_count >= 1

    def test_embed_fn_catches_semantic_duplicates(self) -> None:
        """Embedding similarity catches what lexical matching misses."""
        # Create embed_fn where s1 and s2 map to nearly identical vectors
        def custom_embed(text: str) -> np.ndarray:
            base = np.zeros(32, dtype=np.float32)
            if "quiet" in text or "subdued" in text:
                base[0] = 1.0  # Both map to same direction
                base[1] = 0.1 * hash(text) % 10 / 100  # Tiny variation
            else:
                seed = hash(text) % (2**31)
                rng = np.random.default_rng(seed)
                base = rng.standard_normal(32).astype(np.float32)
            return base / np.linalg.norm(base)

        gates = ["g1", "g2"]
        stmts = [
            "Kasumi becomes quiet when overwhelmed",
            "Kasumi becomes subdued when facing defeat",
        ]
        kept_gates, kept_stmts = merge_similar_hypotheses(
            gates, stmts, embed_fn=custom_embed, similarity_threshold=0.85,
        )
        assert len(kept_stmts) == 1  # Semantic duplicates merged
