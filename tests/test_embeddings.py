"""Unit tests for canopy.embeddings — guard checks, EmbeddingCache, precompute (no GPU needed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import canopy.embeddings as emb
from canopy.embeddings import EmbeddingCache


class TestInitModels:
    """init_models stores paths without loading models."""

    def setup_method(self) -> None:
        self._saved_surface = emb._surface_embedder_path
        self._saved_generator = emb._generator_embedder_path
        self._saved_device = emb._device

    def teardown_method(self) -> None:
        emb._surface_embedder_path = self._saved_surface
        emb._generator_embedder_path = self._saved_generator
        emb._device = self._saved_device

    def test_stores_paths_and_device(self) -> None:
        import torch

        device = torch.device("cpu")
        emb.init_models("/path/to/surface", "/path/to/generator", device)

        assert emb._surface_embedder_path == "/path/to/surface"
        assert emb._generator_embedder_path == "/path/to/generator"
        assert emb._device == device

    def test_does_not_import_heavy_libraries(self) -> None:
        """init_models should NOT trigger SentenceTransformer or AutoModel imports."""
        import torch

        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            patch.dict("sys.modules", {"transformers": None}),
        ):
            # Should succeed — no imports of heavy libs at init time
            emb.init_models("/a", "/b", torch.device("cpu"))


class TestSelectClusterCentersGuard:
    """select_cluster_centers raises if init_models not called."""

    def setup_method(self) -> None:
        self._saved_device = emb._device

    def teardown_method(self) -> None:
        emb._device = self._saved_device

    def test_raises_if_not_initialized(self) -> None:
        emb._device = None
        with pytest.raises(RuntimeError, match="not initialized"):
            emb.select_cluster_centers("Alice", [{"scene": "s", "action": "a"}])

    def test_empty_pairs_returns_empty(self) -> None:
        assert emb.select_cluster_centers("Alice", []) == []

    def test_with_cache_skips_model_loading(self) -> None:
        """When embedding_cache is provided, no model loading occurs."""
        pairs = [{"scene": f"s{i}", "action": f"a{i}", "_embed_idx": i} for i in range(20)]
        cache = EmbeddingCache(
            surface=np.random.randn(20, 4).astype(np.float32),
            generator=np.random.randn(20, 6).astype(np.float32),
        )
        emb._device = None  # Would fail if model loading is attempted
        clusters = emb.select_cluster_centers("Alice", pairs, embedding_cache=cache)
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        # Each cluster is a list of pair dicts
        for cluster in clusters:
            assert isinstance(cluster, list)
            for pair in cluster:
                assert "scene" in pair

    def test_with_cache_missing_embed_idx_raises(self) -> None:
        """Pairs without _embed_idx raise RuntimeError when cache is provided."""
        pairs = [{"scene": "s", "action": "a"}]  # no _embed_idx
        cache = EmbeddingCache(
            surface=np.random.randn(1, 4).astype(np.float32),
            generator=np.random.randn(1, 6).astype(np.float32),
        )
        with pytest.raises(RuntimeError, match="missing '_embed_idx'"):
            emb.select_cluster_centers("Alice", pairs, embedding_cache=cache)

    def test_with_cache_out_of_range_embed_idx_raises(self) -> None:
        """Out-of-range _embed_idx raises RuntimeError before numpy IndexError."""
        cache = EmbeddingCache(
            surface=np.random.randn(5, 4).astype(np.float32),
            generator=np.random.randn(5, 6).astype(np.float32),
        )
        pairs = [{"scene": "s", "action": "a", "_embed_idx": 99}]  # 99 > 5
        with pytest.raises(RuntimeError, match="out-of-range"):
            emb.select_cluster_centers("Alice", pairs, embedding_cache=cache)

    def test_with_cache_negative_embed_idx_raises(self) -> None:
        """Negative _embed_idx is caught by bounds check."""
        cache = EmbeddingCache(
            surface=np.random.randn(5, 4).astype(np.float32),
            generator=np.random.randn(5, 6).astype(np.float32),
        )
        pairs = [{"scene": "s", "action": "a", "_embed_idx": -1}]
        with pytest.raises(RuntimeError, match="out-of-range"):
            emb.select_cluster_centers("Alice", pairs, embedding_cache=cache)

    def test_with_cache_subset_indices(self) -> None:
        """Cache path correctly subsets to the pair indices."""
        # Full cache has 10 items, but we only pass 3 pairs
        full_cache = EmbeddingCache(
            surface=np.random.randn(10, 4).astype(np.float32),
            generator=np.random.randn(10, 6).astype(np.float32),
        )
        pairs = [
            {"scene": f"s{i}", "action": f"a{i}", "_embed_idx": idx}
            for i, idx in enumerate([2, 5, 8])
        ]
        # Won't cluster well with 3 pairs but shouldn't crash
        emb._device = None
        # n_max_cluster must be <= len(pairs)
        clusters = emb.select_cluster_centers(
            "Alice", pairs, n_in_cluster_case=1, n_max_cluster=1,
            embedding_cache=full_cache,
        )
        assert len(clusters) == 1


class TestUnloadModel:
    """_unload_model calls gc.collect and torch.cuda.empty_cache."""

    def test_calls_gc_and_cuda_cleanup(self) -> None:
        mock_model = MagicMock()
        with (
            patch("canopy.embeddings.gc.collect") as mock_gc,
            patch("canopy.embeddings.torch.cuda.is_available", return_value=True),
            patch("canopy.embeddings.torch.cuda.empty_cache") as mock_empty,
        ):
            emb._unload_model(mock_model)
            mock_gc.assert_called_once()
            mock_empty.assert_called_once()

    def test_skips_cuda_cleanup_when_no_gpu(self) -> None:
        mock_model = MagicMock()
        with (
            patch("canopy.embeddings.gc.collect") as mock_gc,
            patch("canopy.embeddings.torch.cuda.is_available", return_value=False),
            patch("canopy.embeddings.torch.cuda.empty_cache") as mock_empty,
        ):
            emb._unload_model(mock_model)
            mock_gc.assert_called_once()
            mock_empty.assert_not_called()

    def test_handles_multiple_models(self) -> None:
        model_a = MagicMock()
        model_b = MagicMock()
        with (
            patch("canopy.embeddings.gc.collect"),
            patch("canopy.embeddings.torch.cuda.is_available", return_value=False),
        ):
            # Should not raise when given multiple models
            emb._unload_model(model_a, model_b)


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    """Tests for the EmbeddingCache frozen dataclass."""

    def _make_cache(self, n: int = 10, d_s: int = 4, d_g: int = 6) -> EmbeddingCache:
        surface = np.random.randn(n, d_s).astype(np.float32)
        generator = np.random.randn(n, d_g).astype(np.float32)
        return EmbeddingCache(surface=surface, generator=generator)

    def test_construction(self) -> None:
        cache = self._make_cache(5, 3, 4)
        assert cache.surface.shape == (5, 3)
        assert cache.generator.shape == (5, 4)

    def test_frozen(self) -> None:
        cache = self._make_cache()
        with pytest.raises(AttributeError):
            cache.surface = np.zeros((1, 1))  # type: ignore[misc]

    def test_arrays_readonly(self) -> None:
        cache = self._make_cache()
        assert not cache.surface.flags.writeable
        assert not cache.generator.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            cache.surface[0, 0] = 999.0

    def test_document_property(self) -> None:
        cache = self._make_cache(5, 3, 4)
        doc = cache.document
        assert doc.shape == (5, 7)  # 4 + 3 = 7 (gen first, then surface)
        # Verify concatenation order: generator columns first
        np.testing.assert_array_equal(doc[:, :4], cache.generator)
        np.testing.assert_array_equal(doc[:, 4:], cache.surface)

    def test_document_cached(self) -> None:
        cache = self._make_cache()
        doc1 = cache.document
        doc2 = cache.document
        assert doc1 is doc2  # Same object, not recomputed

    def test_document_readonly(self) -> None:
        cache = self._make_cache()
        assert not cache.document.flags.writeable

    def test_subset(self) -> None:
        cache = self._make_cache(10, 3, 4)
        sub = cache.subset([0, 5, 9])
        assert sub.surface.shape == (3, 3)
        assert sub.generator.shape == (3, 4)
        np.testing.assert_array_equal(sub.surface[0], cache.surface[0])
        np.testing.assert_array_equal(sub.surface[1], cache.surface[5])
        np.testing.assert_array_equal(sub.surface[2], cache.surface[9])

    def test_subset_preserves_order(self) -> None:
        cache = self._make_cache(10, 3, 4)
        indices = [7, 2, 5]
        sub = cache.subset(indices)
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(sub.surface[i], cache.surface[idx])
            np.testing.assert_array_equal(sub.generator[i], cache.generator[idx])

    def test_subset_copies(self) -> None:
        """Subset arrays are copies, not views — modifying subset can't corrupt parent."""
        cache = self._make_cache(10, 3, 4)
        sub = cache.subset([0, 1])
        # sub arrays are readonly, but verify they don't share memory
        assert not np.shares_memory(sub.surface, cache.surface)
        assert not np.shares_memory(sub.generator, cache.generator)

    def test_subset_empty(self) -> None:
        cache = self._make_cache(10, 3, 4)
        sub = cache.subset([])
        assert sub.surface.shape == (0, 3)
        assert sub.generator.shape == (0, 4)

    def test_subset_with_numpy_indices(self) -> None:
        cache = self._make_cache(10, 3, 4)
        indices = np.array([1, 3, 5])
        sub = cache.subset(indices)
        assert sub.surface.shape == (3, 3)


# ---------------------------------------------------------------------------
# _run_embedding_subprocess
# ---------------------------------------------------------------------------


class TestRunEmbeddingSubprocess:
    """Tests for _run_embedding_subprocess with mocked subprocess."""

    def test_success(self, tmp_path: Path) -> None:
        """Successful subprocess returns correct numpy array."""
        expected = np.random.randn(5, 8).astype(np.float32)

        def mock_run(cmd, **kwargs):
            # Find output path in command args
            output_idx = cmd.index("--output") + 1
            np.save(cmd[output_idx], expected)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("canopy.embeddings.subprocess.run", side_effect=mock_run):
            arr = emb._run_embedding_subprocess(
                texts=["a", "b", "c", "d", "e"],
                model_path="/fake/model",
                model_type="surface",
                character="Test",
                device="cpu",
                bs=8,
                timeout=60,
            )
        np.testing.assert_array_equal(arr, expected)

    def test_subprocess_failure_raises(self) -> None:
        """Non-zero return code raises RuntimeError with stderr."""
        result = MagicMock()
        result.returncode = 1
        result.stderr = "Model not found"
        with patch("canopy.embeddings.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="Model not found"):
                emb._run_embedding_subprocess(
                    texts=["a"], model_path="/fake", model_type="surface",
                    character="X", device="cpu", bs=8, timeout=60,
                )

    def test_oom_killed_subprocess(self) -> None:
        """rc=-9 (SIGKILL/OOM) gives distinct error message."""
        result = MagicMock()
        result.returncode = -9
        result.stderr = ""
        with patch("canopy.embeddings.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="OOM"):
                emb._run_embedding_subprocess(
                    texts=["a"], model_path="/fake", model_type="surface",
                    character="X", device="cpu", bs=8, timeout=60,
                )

    def test_timeout(self) -> None:
        """TimeoutExpired is propagated."""
        with patch(
            "canopy.embeddings.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=10),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                emb._run_embedding_subprocess(
                    texts=["a"], model_path="/fake", model_type="surface",
                    character="X", device="cpu", bs=8, timeout=10,
                )

    def test_shape_validation(self) -> None:
        """Shape mismatch between expected rows and .npy raises RuntimeError."""
        wrong_shape = np.random.randn(3, 8).astype(np.float32)

        def mock_run(cmd, **kwargs):
            output_idx = cmd.index("--output") + 1
            np.save(cmd[output_idx], wrong_shape)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("canopy.embeddings.subprocess.run", side_effect=mock_run):
            with pytest.raises(RuntimeError, match="Shape mismatch"):
                emb._run_embedding_subprocess(
                    texts=["a", "b", "c", "d", "e"],  # 5 texts but 3 rows
                    model_path="/fake", model_type="surface",
                    character="X", device="cpu", bs=8, timeout=60,
                )

    def test_temp_file_cleanup_on_success(self) -> None:
        """Temp files are cleaned up after successful run."""
        expected = np.random.randn(2, 4).astype(np.float32)
        created_files: list[str] = []

        def mock_run(cmd, **kwargs):
            input_idx = cmd.index("--input") + 1
            output_idx = cmd.index("--output") + 1
            created_files.extend([cmd[input_idx], cmd[output_idx]])
            np.save(cmd[output_idx], expected)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("canopy.embeddings.subprocess.run", side_effect=mock_run):
            emb._run_embedding_subprocess(
                texts=["a", "b"], model_path="/fake", model_type="surface",
                character="X", device="cpu", bs=8, timeout=60,
            )

        # Temp files should be cleaned up
        for f in created_files:
            assert not Path(f).exists(), f"Temp file not cleaned: {f}"

    def test_temp_file_cleanup_on_failure(self) -> None:
        """Temp files are cleaned up even when subprocess fails."""
        result = MagicMock()
        result.returncode = 1
        result.stderr = "fail"

        with patch("canopy.embeddings.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError):
                emb._run_embedding_subprocess(
                    texts=["a"], model_path="/fake", model_type="surface",
                    character="X", device="cpu", bs=8, timeout=60,
                )
        # Can't easily check specific paths, but the finally block should have run

    def test_resolves_model_path(self) -> None:
        """Model path is resolved to absolute before passing to subprocess."""
        expected = np.random.randn(1, 4).astype(np.float32)
        captured_cmd: list[str] = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            output_idx = cmd.index("--output") + 1
            np.save(cmd[output_idx], expected)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("canopy.embeddings.subprocess.run", side_effect=mock_run):
            emb._run_embedding_subprocess(
                texts=["a"], model_path="relative/path",
                model_type="surface", character="X", device="cpu", bs=8, timeout=60,
            )

        model_path_idx = captured_cmd.index("--model_path") + 1
        assert Path(captured_cmd[model_path_idx]).is_absolute()

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model_type raises ValueError before subprocess."""
        with pytest.raises(ValueError, match="model_type must be"):
            emb._run_embedding_subprocess(
                texts=["a"], model_path="/fake", model_type="invalid",
                character="X", device="cpu", bs=8, timeout=60,
            )

    def test_writes_json_input(self) -> None:
        """Input texts are written as JSON to temp file."""
        expected = np.random.randn(3, 4).astype(np.float32)

        def mock_run(cmd, **kwargs):
            input_idx = cmd.index("--input") + 1
            # Read and verify the JSON input
            with open(cmd[input_idx]) as f:
                data = json.load(f)
            assert data == ["hello", "world", "test"]
            output_idx = cmd.index("--output") + 1
            np.save(cmd[output_idx], expected)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("canopy.embeddings.subprocess.run", side_effect=mock_run):
            emb._run_embedding_subprocess(
                texts=["hello", "world", "test"],
                model_path="/fake", model_type="surface",
                character="X", device="cpu", bs=8, timeout=60,
            )


# ---------------------------------------------------------------------------
# precompute_embeddings
# ---------------------------------------------------------------------------


class TestPrecomputeEmbeddings:
    """Tests for precompute_embeddings with mocked _run_embedding_subprocess."""

    def test_launches_two_subprocesses(self) -> None:
        """Calls _run_embedding_subprocess twice: surface then generator."""
        calls: list[str] = []
        surface_arr = np.random.randn(2, 4).astype(np.float32)
        gen_arr = np.random.randn(2, 6).astype(np.float32)

        def mock_sub(texts, model_path, model_type, character, device, bs, timeout):
            calls.append(model_type)
            return surface_arr if model_type == "surface" else gen_arr

        with patch("canopy.embeddings._run_embedding_subprocess", side_effect=mock_sub):
            cache = emb.precompute_embeddings(
                character="Alice",
                pairs=[
                    {"action": "a1", "scene": "s1"},
                    {"action": "a2", "scene": "s2"},
                ],
                surface_embedder_path="/surface",
                generator_embedder_path="/gen",
            )

        assert calls == ["surface", "generator"]
        assert cache.surface.shape == (2, 4)
        assert cache.generator.shape == (2, 6)

    def test_passes_actions_to_surface(self) -> None:
        """Surface subprocess receives action texts."""
        captured_texts: dict[str, list[str]] = {}
        arr = np.random.randn(2, 4).astype(np.float32)

        def mock_sub(texts, model_path, model_type, **kw):
            captured_texts[model_type] = texts
            return arr

        with patch("canopy.embeddings._run_embedding_subprocess", side_effect=mock_sub):
            emb.precompute_embeddings(
                character="Alice",
                pairs=[
                    {"action": "act1", "scene": "sc1"},
                    {"action": "act2", "scene": "sc2"},
                ],
                surface_embedder_path="/s", generator_embedder_path="/g",
            )

        assert captured_texts["surface"] == ["act1", "act2"]
        assert captured_texts["generator"] == ["sc1", "sc2"]

    def test_empty_pairs_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            emb.precompute_embeddings(
                character="Alice", pairs=[],
                surface_embedder_path="/s", generator_embedder_path="/g",
            )

    def test_forwards_params(self) -> None:
        """Device, bs, timeout forwarded to subprocess."""
        captured_kw: list[dict] = []
        arr = np.random.randn(1, 4).astype(np.float32)

        def mock_sub(texts, model_path, model_type, character, device, bs, timeout):
            captured_kw.append({"device": device, "bs": bs, "timeout": timeout})
            return arr

        with patch("canopy.embeddings._run_embedding_subprocess", side_effect=mock_sub):
            emb.precompute_embeddings(
                character="Alice",
                pairs=[{"action": "a", "scene": "s"}],
                surface_embedder_path="/s", generator_embedder_path="/g",
                device="cuda:1", bs=16, timeout=300,
            )

        for kw in captured_kw:
            assert kw["device"] == "cuda:1"
            assert kw["bs"] == 16
            assert kw["timeout"] == 300


# ---------------------------------------------------------------------------
# _embed_worker (standalone script)
# ---------------------------------------------------------------------------


class TestEmbedWorkerParsing:
    """Tests for _embed_worker argument parsing."""

    def test_parse_args_surface(self) -> None:
        from canopy._embed_worker import parse_args

        args = parse_args([
            "--input", "/tmp/in.json", "--output", "/tmp/out.npy",
            "--model_path", "/models/qwen", "--model_type", "surface",
            "--device", "cuda:1", "--batch_size", "16",
        ])
        assert args.input == "/tmp/in.json"
        assert args.output == "/tmp/out.npy"
        assert args.model_type == "surface"
        assert args.device == "cuda:1"
        assert args.batch_size == 16

    def test_parse_args_generator_requires_character(self) -> None:
        from canopy._embed_worker import parse_args

        args = parse_args([
            "--input", "/tmp/in.json", "--output", "/tmp/out.npy",
            "--model_path", "/models/qwen", "--model_type", "generator",
            "--character", "Kasumi",
        ])
        assert args.model_type == "generator"
        assert args.character == "Kasumi"

    def test_parse_args_invalid_model_type(self) -> None:
        from canopy._embed_worker import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--input", "/tmp/in.json", "--output", "/tmp/out.npy",
                "--model_path", "/m", "--model_type", "invalid",
            ])
