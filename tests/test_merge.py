"""Tests for E1: merge_similar_hypotheses."""

from __future__ import annotations

import pytest

from canopy.prompts import merge_similar_hypotheses


class TestMergeSimilarHypotheses:
    def test_no_duplicates_unchanged(self) -> None:
        gates = ["gate A", "gate B", "gate C"]
        stmts = ["Kasumi sings when happy", "Arisa scolds Kasumi", "Tae plays guitar quietly"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert kept_gates == gates
        assert kept_stmts == stmts

    def test_exact_duplicates_merged(self) -> None:
        gates = ["gate A", "gate B"]
        stmts = ["Kasumi tends to sing loudly", "Kasumi tends to sing loudly"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 1
        assert len(kept_gates) == 1

    def test_near_duplicates_merged(self) -> None:
        gates = ["gate A", "gate B"]
        stmts = [
            "Kasumi tends to express enthusiasm vocally when performing",
            "Kasumi tends to express enthusiasm vocally when performing live",
        ]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        # High jaccard similarity → should merge, keeping shorter
        assert len(kept_stmts) == 1
        assert kept_stmts[0] == "Kasumi tends to express enthusiasm vocally when performing"

    def test_different_statements_not_merged(self) -> None:
        gates = ["gate A", "gate B"]
        stmts = [
            "Kasumi sings when the band is struggling",
            "Arisa criticizes Kasumi's impulsive decisions",
        ]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
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

    def test_short_strings(self) -> None:
        gates = ["g1", "g2"]
        stmts = ["ab", "ab"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 1

    def test_custom_threshold(self) -> None:
        gates = ["gate A", "gate B"]
        stmts = ["statement about music", "statement about musical"]
        # With low threshold, these merge
        kept_low = merge_similar_hypotheses(gates, stmts, similarity_threshold=0.5)
        # With very high threshold, these don't merge
        kept_high = merge_similar_hypotheses(gates, stmts, similarity_threshold=0.99)
        assert len(kept_low[1]) <= len(kept_high[1])

    def test_preserves_gate_statement_alignment(self) -> None:
        gates = ["when sad", "when happy", "when sad again"]
        stmts = ["Kasumi comforts others", "Kasumi celebrates loudly", "Kasumi comforts others tenderly"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        # stmts 0 and 2 are similar → one is merged
        # The kept pairs should still be aligned (same index)
        assert len(kept_gates) == len(kept_stmts)
        for g, s in zip(kept_gates, kept_stmts):
            # Each gate should match its original statement
            orig_idx = gates.index(g) if g in gates else -1
            if orig_idx >= 0:
                assert s == stmts[orig_idx]

    def test_three_way_merge(self) -> None:
        gates = ["g1", "g2", "g3"]
        stmts = ["same text here", "same text here", "same text here"]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        assert len(kept_stmts) == 1

    def test_three_way_merge_longest_first(self) -> None:
        """Cover the break path: when i (longest) merges into j (shorter)."""
        gates = ["g1", "g2", "g3"]
        # These are close enough (>0.90 Jaccard) for the longest-first pair
        stmts = [
            "Kasumi tends to express enthusiasm vocally when performing live today",
            "Kasumi tends to express enthusiasm vocally when performing live",
            "Kasumi tends to express enthusiasm vocally when performing live today",
        ]
        kept_gates, kept_stmts = merge_similar_hypotheses(gates, stmts)
        # stmts[0] and stmts[2] are identical → merge
        # stmts[0] is longer than stmts[1] → if compared, merges into stmts[1] via break
        assert len(kept_stmts) <= 2
        # Shortest version should survive
        assert "Kasumi tends to express enthusiasm vocally when performing live" in kept_stmts
