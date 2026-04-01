"""Tests for scripts/cdt_steps.py — step-level CDT pipeline CLI."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test from scripts/ (not a package)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _import_cdt_steps():
    """Import cdt_steps.py without requiring it to be a package.

    Registers the module in sys.modules so that unittest.mock.patch
    can resolve 'cdt_steps.canopy_data' etc.
    """
    spec = importlib.util.spec_from_file_location("cdt_steps", SCRIPTS_DIR / "cdt_steps.py")
    if spec is None or spec.loader is None:
        pytest.skip("scripts/cdt_steps.py not found")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cdt_steps"] = mod
    spec.loader.exec_module(mod)
    return mod


cdt_steps = _import_cdt_steps()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_PAIRS = [
    {"scene": f"scene_{i}", "action": f"action_{i}", "actor": "Kasumi"}
    for i in range(20)
]


def _seed_pairs(cache_path: Path) -> Path:
    """Write pairs.json into data/ subdirectory."""
    data_dir = cache_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = data_dir / "pairs.json"
    pairs_file.write_text(json.dumps(FAKE_PAIRS))
    return pairs_file


def _seed_embeddings(cache_path: Path, n: int = 20, dim: int = 8) -> Path:
    """Write fake cache.npz into embedding/ subdirectory."""
    emb_dir = cache_path / "embedding"
    emb_dir.mkdir(parents=True, exist_ok=True)
    npz_path = emb_dir / "cache.npz"
    surface = np.random.default_rng(42).standard_normal((n, dim)).astype(np.float32)
    generator = np.random.default_rng(43).standard_normal((n, dim)).astype(np.float32)
    np.savez(npz_path, surface=surface, generator=generator)
    return npz_path


def _seed_clustering(cache_path: Path, n: int = 20, k: int = 3) -> Path:
    """Write labels.npy and centroids.npy into clustering/ subdirectory.

    Centroids are 16-dim to match document embeddings (generator 8 + surface 8).
    """
    clust_dir = cache_path / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    labels = rng.integers(0, k, size=n).astype(np.int32)
    centroids = rng.standard_normal((k, 16)).astype(np.float32)
    np.save(clust_dir / "labels.npy", labels)
    np.save(clust_dir / "centroids.npy", centroids)
    return clust_dir


def _seed_hypotheses(cache_path: Path, step_name: str = "hypothesis_gen") -> Path:
    """Write hypotheses.json into given step subdirectory."""
    step_dir = cache_path / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    hyp_file = step_dir / "hypotheses.json"
    hyp_data = {
        "statements": ["stmt1", "stmt2", "stmt3"],
        "gates": ["gate1", "gate2", "gate3"],
    }
    hyp_file.write_text(json.dumps(hyp_data))
    return hyp_file


def _seed_validation_results(cache_path: Path) -> Path:
    """Write results.json into validate/ subdirectory."""
    val_dir = cache_path / "validate"
    val_dir.mkdir(parents=True, exist_ok=True)
    results_file = val_dir / "results.json"
    results = [
        {
            "statement": "stmt1",
            "gate": "gate1",
            "accuracy": 0.85,
            "broadness": 0.7,
            "verdict": "accept",
        },
        {
            "statement": "stmt2",
            "gate": "gate2",
            "accuracy": 0.65,
            "broadness": 0.4,
            "verdict": "gate",
        },
    ]
    results_file.write_text(json.dumps(results))
    return results_file


class _PicklableNode:
    """Picklable stand-in for CDTNode used in seed helpers."""

    def __init__(self) -> None:
        self.statements: list[str] = ["stmt1"]
        self.gates: list[str] = []
        self.children: list[_PicklableNode] = []
        self.depth = 1

    def count_stats(self) -> dict:
        return {"total_nodes": 1, "total_statements": 1, "total_gates": 0, "max_depth": 1}


def _seed_build_tree(cache_path: Path) -> Path:
    """Write fake pickle files into build_tree/ subdirectory."""
    bt_dir = cache_path / "build_tree"
    bt_dir.mkdir(parents=True, exist_ok=True)

    fake_node = _PicklableNode()
    topic2cdt = {"identity": fake_node}
    rel_topic2cdt = {"relationship_Arisa": fake_node}

    with open(bt_dir / "topic2cdt.pkl", "wb") as f:
        pickle.dump(topic2cdt, f)
    with open(bt_dir / "rel_topic2cdt.pkl", "wb") as f:
        pickle.dump(rel_topic2cdt, f)

    return bt_dir


# ---------------------------------------------------------------------------
# next_build_id
# ---------------------------------------------------------------------------


class TestNextBuildId:
    def test_empty_directory(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = cdt_steps.next_build_id(cache_dir, "Kasumi")
        assert result == "001"

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        # Directory does not exist
        result = cdt_steps.next_build_id(cache_dir, "Kasumi")
        assert result == "001"

    def test_existing_001(self, tmp_path: Path) -> None:
        char_dir = tmp_path / "cache" / "Kasumi"
        (char_dir / "001").mkdir(parents=True)
        result = cdt_steps.next_build_id(tmp_path / "cache", "Kasumi")
        assert result == "002"

    def test_gap_uses_max(self, tmp_path: Path) -> None:
        """Existing 001 and 003 -> next is 004 (uses max, not count)."""
        char_dir = tmp_path / "cache" / "Kasumi"
        (char_dir / "001").mkdir(parents=True)
        (char_dir / "003").mkdir(parents=True)
        result = cdt_steps.next_build_id(tmp_path / "cache", "Kasumi")
        assert result == "004"

    def test_non_numeric_dirs_ignored(self, tmp_path: Path) -> None:
        char_dir = tmp_path / "cache" / "Kasumi"
        (char_dir / "001").mkdir(parents=True)
        (char_dir / "notes").mkdir(parents=True)
        (char_dir / "tmp_backup").mkdir(parents=True)
        result = cdt_steps.next_build_id(tmp_path / "cache", "Kasumi")
        assert result == "002"


# ---------------------------------------------------------------------------
# atomic_write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_write_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "output.txt"
        cdt_steps.atomic_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        cdt_steps.atomic_write(target, "nested")
        assert target.read_text() == "nested"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "output.txt"
        target.write_text("old content")
        cdt_steps.atomic_write(target, "new content")
        assert target.read_text() == "new content"

    def test_binary_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "output.bin"
        data = b"\x00\x01\x02\xff"
        cdt_steps.atomic_write(target, data, mode="wb")
        assert target.read_bytes() == data


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_minimal_args(self) -> None:
        args = cdt_steps.parse_args(["--step", "data"])
        assert args.step == "data"

    def test_all_valid_steps(self) -> None:
        valid_steps = [
            "data", "embedding", "clustering", "hypothesis_gen",
            "dedup", "compress", "validate", "build_tree", "wikify",
        ]
        for step in valid_steps:
            args = cdt_steps.parse_args(["--step", step])
            assert args.step == step

    def test_character_default(self) -> None:
        args = cdt_steps.parse_args(["--step", "data"])
        # Default character should be set (likely "Kasumi")
        assert hasattr(args, "character")

    def test_character_specified(self) -> None:
        args = cdt_steps.parse_args(["--step", "data", "--character", "Arisa"])
        assert args.character == "Arisa"

    def test_all_args_specified(self) -> None:
        args = cdt_steps.parse_args([
            "--step", "embedding",
            "--character", "Kasumi",
            "--cache_dir", "/tmp/cache",
            "--engine", "claude-haiku-4-5",
            "--surface_embedder_path", "/models/qwen",
            "--generator_embedder_path", "/models/qwen-gen",
            "--discriminator_path", "/models/deberta",
            "--device_id", "1",
        ])
        assert args.step == "embedding"
        assert args.character == "Kasumi"

    def test_invalid_step_raises(self) -> None:
        with pytest.raises(SystemExit):
            cdt_steps.parse_args(["--step", "nonexistent_step"])

    def test_missing_step_raises(self) -> None:
        with pytest.raises(SystemExit):
            cdt_steps.parse_args([])


# ---------------------------------------------------------------------------
# step_data
# ---------------------------------------------------------------------------


class TestStepData:
    @patch("cdt_steps.canopy_data")
    def test_loads_and_saves_pairs(self, mock_data: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        mock_data.load_character_metadata.return_value = (
            {"artifact1": {"major": ["Kasumi"]}},
            {"Kasumi": "artifact1"},
            {"band1": ["Kasumi", "Arisa"]},
        )
        mock_data.load_ar_pairs.return_value = {
            "train": FAKE_PAIRS[:15],
            "test": FAKE_PAIRS[15:],
        }

        result = cdt_steps.step_data(
            character="Kasumi",
            cache_path=cache_path,
        )

        # Verify pairs.json is written
        pairs_file = cache_path / "data" / "pairs.json"
        assert pairs_file.exists()
        saved_pairs = json.loads(pairs_file.read_text())
        assert len(saved_pairs) == 15  # train split

        # Verify quality metrics returned
        assert "pair_count" in result
        assert result["pair_count"] == 15

    @patch("cdt_steps.canopy_data")
    def test_quality_json_updated(self, mock_data: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        mock_data.load_character_metadata.return_value = (
            {"artifact1": {"major": ["Kasumi"]}},
            {"Kasumi": "artifact1"},
            {"band1": ["Kasumi"]},
        )
        mock_data.load_ar_pairs.return_value = {
            "train": FAKE_PAIRS,
            "test": [],
        }

        cdt_steps.step_data(character="Kasumi", cache_path=cache_path)

        quality_file = cache_path / "quality.json"
        assert quality_file.exists()
        quality = json.loads(quality_file.read_text())
        assert "data" in quality
        assert "pair_count" in quality["data"]


# ---------------------------------------------------------------------------
# step_embedding
# ---------------------------------------------------------------------------


class TestStepEmbedding:
    @patch("cdt_steps.canopy_embeddings")
    def test_runs_precompute(self, mock_emb: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)

        fake_cache = MagicMock()
        fake_cache.surface = np.zeros((20, 8), dtype=np.float32)
        fake_cache.generator = np.zeros((20, 8), dtype=np.float32)
        mock_emb.precompute_embeddings.return_value = fake_cache

        result = cdt_steps.step_embedding(
            character="Kasumi",
            cache_path=cache_path,
            surface_embedder_path="/models/qwen",
            generator_embedder_path="/models/qwen-gen",
        )

        # Verify cache.npz is written
        npz_path = cache_path / "embedding" / "cache.npz"
        assert npz_path.exists()

        # Verify quality metrics
        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_embeddings")
    def test_missing_upstream_raises(self, mock_emb: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        # No data/pairs.json seeded
        with pytest.raises(RuntimeError, match="(?i)upstream|pairs|data"):
            cdt_steps.step_embedding(
                character="Kasumi",
                cache_path=cache_path,
                surface_embedder_path="/models/qwen",
                generator_embedder_path="/models/qwen-gen",
            )


# ---------------------------------------------------------------------------
# step_clustering
# ---------------------------------------------------------------------------


class TestStepClustering:
    @patch("cdt_steps.canopy_cluster")
    def test_runs_clustering(self, mock_cluster: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_embeddings(cache_path)

        fake_labels = np.array([0, 1, 2, 0, 1] * 4, dtype=np.int32)
        fake_centroids = np.random.default_rng(42).standard_normal((3, 8)).astype(np.float32)
        mock_cluster.KMeansCluster.return_value.fit_predict.return_value = (
            fake_labels,
            fake_centroids,
        )

        result = cdt_steps.step_clustering(
            character="Kasumi",
            cache_path=cache_path,
        )

        # Verify outputs written
        clust_dir = cache_path / "clustering"
        assert (clust_dir / "labels.npy").exists()
        assert (clust_dir / "centroids.npy").exists()

        # Verify quality metrics
        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_cluster")
    def test_missing_embedding_raises(self, mock_cluster: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        # No embedding cache seeded
        with pytest.raises(RuntimeError, match="(?i)upstream|embedding|cache"):
            cdt_steps.step_clustering(
                character="Kasumi",
                cache_path=cache_path,
            )


# ---------------------------------------------------------------------------
# step_hypothesis_gen
# ---------------------------------------------------------------------------


class TestStepHypothesisGen:
    @patch("cdt_steps.canopy_prompts")
    @patch("cdt_steps.canopy_cluster")
    def test_generates_hypotheses(self, mock_cluster: MagicMock, mock_prompts: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_embeddings(cache_path)
        _seed_clustering(cache_path)

        mock_cluster.select_representative_samples.return_value = [FAKE_PAIRS[:5]]
        mock_prompts.make_hypotheses_batch.return_value = (
            ["stmt1", "stmt2"],
            ["gate1", "gate2"],
        )

        result = cdt_steps.step_hypothesis_gen(
            character="Kasumi",
            cache_path=cache_path,
        )

        # Verify hypotheses.json written
        hyp_file = cache_path / "hypothesis_gen" / "hypotheses.json"
        assert hyp_file.exists()
        hyp_data = json.loads(hyp_file.read_text())
        assert "statements" in hyp_data
        assert "gates" in hyp_data

        # Verify quality metrics
        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_prompts")
    @patch("cdt_steps.canopy_cluster")
    def test_missing_clustering_raises(self, mock_cluster: MagicMock, mock_prompts: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_embeddings(cache_path)
        # No clustering seeded
        with pytest.raises(RuntimeError, match="(?i)upstream|clustering"):
            cdt_steps.step_hypothesis_gen(
                character="Kasumi",
                cache_path=cache_path,
            )


# ---------------------------------------------------------------------------
# step_dedup
# ---------------------------------------------------------------------------


class TestStepDedup:
    @patch("cdt_steps.canopy_prompts")
    def test_deduplicates(self, mock_prompts: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_hypotheses(cache_path, "hypothesis_gen")

        mock_prompts.merge_similar_hypotheses.return_value = (
            ["merged_gate1", "merged_gate2"],
            ["merged_stmt1", "merged_stmt2"],
        )

        result = cdt_steps.step_dedup(
            character="Kasumi",
            cache_path=cache_path,
        )

        # Verify output written
        out_file = cache_path / "dedup" / "hypotheses.json"
        assert out_file.exists()
        out_data = json.loads(out_file.read_text())
        assert out_data["statements"] == ["merged_stmt1", "merged_stmt2"]

        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_prompts")
    def test_missing_upstream_raises(self, mock_prompts: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        # No hypothesis_gen/hypotheses.json
        with pytest.raises(RuntimeError, match="(?i)upstream|hypothesis"):
            cdt_steps.step_dedup(
                character="Kasumi",
                cache_path=cache_path,
            )


# ---------------------------------------------------------------------------
# step_compress
# ---------------------------------------------------------------------------


class TestStepCompress:
    @patch("cdt_steps.canopy_prompts")
    def test_compresses(self, mock_prompts: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_hypotheses(cache_path, "dedup")

        mock_prompts.summarize_triggers.return_value = (
            ["compressed_gate"],
            ["compressed_stmt"],
        )

        result = cdt_steps.step_compress(
            character="Kasumi",
            cache_path=cache_path,
        )

        out_file = cache_path / "compress" / "hypotheses.json"
        assert out_file.exists()
        out_data = json.loads(out_file.read_text())
        assert out_data["statements"] == ["compressed_stmt"]

        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_prompts")
    def test_missing_upstream_raises(self, mock_prompts: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        # No dedup/hypotheses.json
        with pytest.raises(RuntimeError, match="(?i)upstream|dedup"):
            cdt_steps.step_compress(
                character="Kasumi",
                cache_path=cache_path,
            )


# ---------------------------------------------------------------------------
# step_validate
# ---------------------------------------------------------------------------


class TestStepValidate:
    @patch("cdt_steps.canopy_validation")
    def test_validates(self, mock_val: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_hypotheses(cache_path, "compress")

        mock_val.init_models.return_value = None
        mock_val.validate_hypothesis.return_value = (
            {"True": 8.5, "False": 1.5, "Irrelevant": 2.0},
            FAKE_PAIRS[:5],
        )

        result = cdt_steps.step_validate(
            character="Kasumi",
            cache_path=cache_path,
            discriminator_path="/models/deberta",
        )

        results_file = cache_path / "validate" / "results.json"
        assert results_file.exists()

        assert isinstance(result, dict)
        # True=8.5, False=1.5 → correctness ≈ 0.85 → accepted
        assert result["n_accepted"] > 0
        assert result["mean_correctness"] > 0.8

    @patch("cdt_steps.canopy_validation")
    def test_missing_compress_raises(self, mock_val: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        # No compress/hypotheses.json
        with pytest.raises(RuntimeError, match="(?i)upstream|compress"):
            cdt_steps.step_validate(
                character="Kasumi",
                cache_path=cache_path,
                discriminator_path="/models/deberta",
            )

    @patch("cdt_steps.canopy_validation")
    def test_missing_pairs_raises(self, mock_val: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_hypotheses(cache_path, "compress")
        # No data/pairs.json
        with pytest.raises(RuntimeError, match="(?i)upstream|pairs|data"):
            cdt_steps.step_validate(
                character="Kasumi",
                cache_path=cache_path,
                discriminator_path="/models/deberta",
            )

    @patch("cdt_steps.canopy_validation")
    def test_all_irrelevant_returns_zero_correctness(self, mock_val: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_hypotheses(cache_path, "compress")

        mock_val.init_models.return_value = None
        mock_val.validate_hypothesis.return_value = (
            {"True": 0.0, "False": 0.0, "Irrelevant": 10.0},
            [],
        )

        result = cdt_steps.step_validate(
            character="Kasumi",
            cache_path=cache_path,
            discriminator_path="/models/deberta",
        )

        assert result["mean_correctness"] == 0.0
        assert result["n_rejected"] == len(json.loads(
            (cache_path / "compress" / "hypotheses.json").read_text()
        )["statements"])


# ---------------------------------------------------------------------------
# step_build_tree
# ---------------------------------------------------------------------------


class TestStepBuildTree:
    @patch("cdt_steps.canopy_data")
    @patch("cdt_steps.canopy_core")
    def test_builds_tree(self, mock_core: MagicMock, mock_data: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        _seed_embeddings(cache_path)

        fake_node = _PicklableNode()
        topic2cdt = {"identity": fake_node}
        rel_topic2cdt = {"relationship_Arisa": fake_node}
        mock_core.build_character_cdts.return_value = (topic2cdt, rel_topic2cdt)
        mock_data.load_character_metadata.return_value = (
            {}, {}, {"band1": ["Kasumi", "Arisa"]},
        )

        result = cdt_steps.step_build_tree(
            character="Kasumi",
            cache_path=cache_path,
        )

        # Verify pickle files written
        bt_dir = cache_path / "build_tree"
        assert (bt_dir / "topic2cdt.pkl").exists()
        assert (bt_dir / "rel_topic2cdt.pkl").exists()

        # Verify quality metrics
        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_core")
    def test_missing_pairs_raises(self, mock_core: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_embeddings(cache_path)
        # No data/pairs.json
        with pytest.raises(RuntimeError, match="(?i)upstream|pairs|data"):
            cdt_steps.step_build_tree(
                character="Kasumi",
                cache_path=cache_path,
            )

    @patch("cdt_steps.canopy_core")
    def test_missing_embeddings_raises(self, mock_core: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_pairs(cache_path)
        # No embedding/cache.npz
        with pytest.raises(RuntimeError, match="(?i)upstream|embedding"):
            cdt_steps.step_build_tree(
                character="Kasumi",
                cache_path=cache_path,
            )


# ---------------------------------------------------------------------------
# step_wikify
# ---------------------------------------------------------------------------


class TestStepWikify:
    @patch("cdt_steps.canopy_wikify")
    def test_wikifies(self, mock_wikify: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        _seed_build_tree(cache_path)

        mock_wikify.wikify_profile.return_value = "# Kasumi\n\n- Statement 1"

        result = cdt_steps.step_wikify(
            character="Kasumi",
            cache_path=cache_path,
        )

        profile_file = cache_path / "wikify" / "profile.md"
        assert profile_file.exists()
        assert "Kasumi" in profile_file.read_text()

        assert isinstance(result, dict)

    @patch("cdt_steps.canopy_wikify")
    def test_missing_build_tree_raises(self, mock_wikify: MagicMock, tmp_path: Path) -> None:
        cache_path = tmp_path / "build_001"
        # No build_tree pickle files
        with pytest.raises(RuntimeError, match="(?i)upstream|build_tree|pickle"):
            cdt_steps.step_wikify(
                character="Kasumi",
                cache_path=cache_path,
            )


# ---------------------------------------------------------------------------
# load_json / update_quality / timestamp serialization / main
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Missing upstream cache"):
            cdt_steps.load_json(tmp_path / "nonexistent.json")

    def test_loads_existing_json(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"key": "value"}))
        result = cdt_steps.load_json(path)
        assert result == {"key": "value"}


class TestUpdateQuality:
    def test_creates_new_quality_file(self, tmp_path: Path) -> None:
        cdt_steps.update_quality(tmp_path, "data", {"pair_count": 10})
        quality = json.loads((tmp_path / "quality.json").read_text())
        assert quality["data"]["pair_count"] == 10

    def test_merges_with_existing_quality(self, tmp_path: Path) -> None:
        # Pre-populate quality.json with a prior step
        (tmp_path / "quality.json").write_text(
            json.dumps({"data": {"pair_count": 10}})
        )
        cdt_steps.update_quality(tmp_path, "clustering", {"n_clusters": 3})
        quality = json.loads((tmp_path / "quality.json").read_text())
        assert quality["data"]["pair_count"] == 10
        assert quality["clustering"]["n_clusters"] == 3


class TestTimestampSerialization:
    @patch("cdt_steps.canopy_data")
    def test_pairs_with_timestamp_serialized(self, mock_data: MagicMock, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        cache_path = tmp_path / "build_001"
        pairs_with_ts = [
            {
                "scene": "scene_0",
                "action": "action_0",
                "_timestamp": datetime(2024, 6, 15, tzinfo=timezone.utc),
            },
        ]
        mock_data.load_character_metadata.return_value = (
            {"a": {"major": ["Kasumi"]}}, {"Kasumi": "a"}, {},
        )
        mock_data.load_ar_pairs.return_value = {"train": pairs_with_ts, "test": []}

        cdt_steps.step_data(character="Kasumi", cache_path=cache_path)

        saved = json.loads((cache_path / "data" / "pairs.json").read_text())
        assert saved[0]["_timestamp"] == "2024-06-15T00:00:00+00:00"


class TestMain:
    @patch("cdt_steps.STEP_MAP", {"data": MagicMock(return_value={"pair_count": 5})})
    def test_main_single_step(self, tmp_path: Path) -> None:
        with patch("cdt_steps.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                step="data", character="Kasumi", cache_dir=str(tmp_path),
                build_id="001", surface_embedder_path="/m/q",
                generator_embedder_path="/m/g", discriminator_path="/m/d",
                device_id=0, engine="claude-haiku-4-5",
            )
            cdt_steps.main()
        cdt_steps.STEP_MAP["data"].assert_called_once()

    def test_main_all_steps(self, tmp_path: Path) -> None:
        mock_fns = {name: MagicMock(return_value={}) for name, _ in cdt_steps.ORDERED_STEPS}
        with patch("cdt_steps.parse_args") as mock_parse, \
             patch("cdt_steps.ORDERED_STEPS", [(n, mock_fns[n]) for n, _ in cdt_steps.ORDERED_STEPS]):
            mock_parse.return_value = MagicMock(
                step="all", character="Kasumi", cache_dir=str(tmp_path),
                build_id="001", surface_embedder_path="/m/q",
                generator_embedder_path="/m/g", discriminator_path="/m/d",
                device_id=0, engine="claude-haiku-4-5",
            )
            cdt_steps.main()
        for fn in mock_fns.values():
            fn.assert_called_once()
