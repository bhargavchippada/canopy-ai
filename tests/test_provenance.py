"""Tests for canopy.provenance — Provenance, TrackedHypothesis, HypothesisQuality."""

from __future__ import annotations

import pickle

import pytest

from canopy.provenance import HypothesisQuality, Provenance, TrackedHypothesis


class TestHypothesisQuality:
    def test_creation(self) -> None:
        hq = HypothesisQuality(
            nli_true_rate=0.7,
            nli_false_rate=0.1,
            nli_irrelevant_rate=0.2,
            specificity=0.3,
            word_count=12,
            grounding_fidelity=0.65,
        )
        assert hq.nli_true_rate == 0.7
        assert hq.nli_false_rate == 0.1
        assert hq.nli_irrelevant_rate == 0.2
        assert hq.specificity == 0.3
        assert hq.word_count == 12
        assert hq.grounding_fidelity == 0.65

    def test_frozen(self) -> None:
        hq = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        with pytest.raises(AttributeError):
            hq.nli_true_rate = 0.9  # type: ignore[misc]

    def test_equality(self) -> None:
        a = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        b = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        assert a == b

    def test_inequality(self) -> None:
        a = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        b = HypothesisQuality(0.8, 0.1, 0.2, 0.3, 12, 0.65)
        assert a != b

    def test_hashable(self) -> None:
        hq = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        d = {hq: "test"}
        assert d[hq] == "test"

    def test_pickle_roundtrip(self) -> None:
        hq = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        restored = pickle.loads(pickle.dumps(hq))
        assert restored == hq


class TestProvenance:
    def test_creation_minimal(self) -> None:
        p = Provenance(source_pair_indices=(0, 1, 2))
        assert p.source_pair_indices == (0, 1, 2)
        assert p.cluster_id is None
        assert p.hypothesis_id is None
        assert p.step == ""
        assert p.metadata == ()

    def test_creation_full(self) -> None:
        p = Provenance(
            source_pair_indices=(3, 7, 12),
            cluster_id=2,
            hypothesis_id="h_001",
            step="hypothesis_gen",
            metadata=(("model", "haiku"), ("temperature", 0.7)),
        )
        assert p.cluster_id == 2
        assert p.hypothesis_id == "h_001"
        assert p.step == "hypothesis_gen"
        assert p.metadata == (("model", "haiku"), ("temperature", 0.7))

    def test_frozen(self) -> None:
        p = Provenance(source_pair_indices=(0,))
        with pytest.raises(AttributeError):
            p.cluster_id = 5  # type: ignore[misc]

    def test_metadata_dict(self) -> None:
        p = Provenance(
            source_pair_indices=(0,),
            metadata=(("key1", "val1"), ("key2", 42)),
        )
        d = p.metadata_dict()
        assert d == {"key1": "val1", "key2": 42}
        assert isinstance(d, dict)

    def test_metadata_dict_empty(self) -> None:
        p = Provenance(source_pair_indices=(0,))
        assert p.metadata_dict() == {}

    def test_equality(self) -> None:
        a = Provenance(source_pair_indices=(1, 2), cluster_id=0)
        b = Provenance(source_pair_indices=(1, 2), cluster_id=0)
        assert a == b

    def test_hashable(self) -> None:
        p = Provenance(source_pair_indices=(1,), cluster_id=3)
        d = {p: "test"}
        assert d[p] == "test"

    def test_pickle_roundtrip(self) -> None:
        p = Provenance(
            source_pair_indices=(1, 2, 3),
            cluster_id=5,
            hypothesis_id="h_test",
            step="validation",
            metadata=(("extra", True),),
        )
        restored = pickle.loads(pickle.dumps(p))
        assert restored == p


class TestTrackedHypothesis:
    def test_creation(self) -> None:
        prov = Provenance(source_pair_indices=(0, 1))
        th = TrackedHypothesis(
            statement="Alice is kind",
            gate="Is there a friend nearby?",
            provenance=prov,
        )
        assert th.statement == "Alice is kind"
        assert th.gate == "Is there a friend nearby?"
        assert th.provenance is prov
        assert th.quality is None

    def test_with_quality(self) -> None:
        prov = Provenance(source_pair_indices=(0,))
        hq = HypothesisQuality(0.7, 0.1, 0.2, 0.3, 12, 0.65)
        th = TrackedHypothesis(
            statement="Alice is brave",
            gate="Danger?",
            provenance=prov,
            quality=hq,
        )
        assert th.quality is hq
        assert th.quality.nli_true_rate == 0.7

    def test_frozen(self) -> None:
        prov = Provenance(source_pair_indices=(0,))
        th = TrackedHypothesis(statement="s", gate="g", provenance=prov)
        with pytest.raises(AttributeError):
            th.statement = "new"  # type: ignore[misc]

    def test_equality(self) -> None:
        prov = Provenance(source_pair_indices=(0,))
        a = TrackedHypothesis(statement="s", gate="g", provenance=prov)
        b = TrackedHypothesis(statement="s", gate="g", provenance=prov)
        assert a == b

    def test_hashable(self) -> None:
        prov = Provenance(source_pair_indices=(0,))
        th = TrackedHypothesis(statement="s", gate="g", provenance=prov)
        d = {th: "test"}
        assert d[th] == "test"

    def test_pickle_roundtrip(self) -> None:
        prov = Provenance(source_pair_indices=(1, 2), cluster_id=3)
        hq = HypothesisQuality(0.5, 0.2, 0.3, 0.5, 8, 0.4)
        th = TrackedHypothesis(statement="s", gate="g", provenance=prov, quality=hq)
        restored = pickle.loads(pickle.dumps(th))
        assert restored == th
        assert restored.quality == hq
