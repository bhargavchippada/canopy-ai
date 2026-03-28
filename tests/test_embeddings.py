"""Unit tests for canopy.embeddings — guard checks and sequential loading (no GPU needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import canopy.embeddings as emb


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
