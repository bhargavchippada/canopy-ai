"""Embedding models and clustering for CDT construction.

Models are loaded sequentially in select_cluster_centers() to support
VRAM-constrained GPUs (e.g. two 8B models on a 32GB card).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCache:
    """Pre-computed embeddings for all training pairs.

    Arrays are made read-only on construction to prevent accidental mutation
    across concurrent threads. _embed_idx is a reserved key on pair dicts.
    """

    surface: np.ndarray  # (N, D_surface) L2-normalized, read-only
    generator: np.ndarray  # (N, D_gen) L2-normalized, read-only
    _document: np.ndarray = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        # Copy arrays to avoid mutating caller's originals, then make read-only
        s = np.array(self.surface, copy=True)
        g = np.array(self.generator, copy=True)
        if s.ndim != 2 or g.ndim != 2:
            raise ValueError("surface and generator must be 2-D arrays")
        if len(s) != len(g):
            raise ValueError(
                f"surface rows ({len(s)}) != generator rows ({len(g)})"
            )
        s.flags.writeable = False
        object.__setattr__(self, "surface", s)
        g.flags.writeable = False
        object.__setattr__(self, "generator", g)

        # Eagerly compute document (thread-safe — no lazy init race)
        doc = np.concatenate([g, s], axis=-1)
        doc.flags.writeable = False
        object.__setattr__(self, "_document", doc)

    @property
    def document(self) -> np.ndarray:
        """Combined embeddings for clustering (gen + surface concatenated)."""
        return self._document

    def subset(self, indices: list[int] | np.ndarray) -> EmbeddingCache:
        """Return cache with only the specified pair indices. Always copies."""
        return EmbeddingCache(
            surface=self.surface[indices].copy(),
            generator=self.generator[indices].copy(),
        )


def precompute_embeddings(
    character: str,
    pairs: list[dict[str, Any]],
    surface_embedder_path: str,
    generator_embedder_path: str,
    device: str = "cuda:0",
    bs: int = 8,
    timeout: int = 600,
) -> EmbeddingCache:
    """Pre-compute surface and generator embeddings for all pairs via subprocesses.

    Each model runs in a separate subprocess to isolate VRAM usage and allow
    sequential loading on constrained GPUs.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")

    actions = [p["action"] for p in pairs]
    scenes = [p["scene"] for p in pairs]

    surface_arr = _run_embedding_subprocess(
        texts=actions,
        model_path=surface_embedder_path,
        model_type="surface",
        character=character,
        device=device,
        bs=bs,
        timeout=timeout,
    )
    gen_arr = _run_embedding_subprocess(
        texts=scenes,
        model_path=generator_embedder_path,
        model_type="generator",
        character=character,
        device=device,
        bs=bs,
        timeout=timeout,
    )
    return EmbeddingCache(surface=surface_arr, generator=gen_arr)


def _run_embedding_subprocess(
    texts: list[str],
    model_path: str,
    model_type: str,
    character: str,
    device: str,
    bs: int,
    timeout: int,
) -> np.ndarray:
    """Run embedding computation in a subprocess for VRAM isolation."""
    if model_type not in ("surface", "generator"):
        raise ValueError(f"model_type must be 'surface' or 'generator', got '{model_type}'")
    resolved_path = str(Path(model_path).resolve())
    input_fd, input_path = tempfile.mkstemp(suffix=".json", prefix="canopy_input_")
    output_fd, output_path = tempfile.mkstemp(suffix=".npy", prefix="canopy_output_")
    os.close(output_fd)  # close immediately; subprocess writes to path
    try:
        with os.fdopen(input_fd, "w") as f:  # os.fdopen takes ownership of fd
            json.dump(texts, f)

        log.debug("Launching embed worker with executable: %s", sys.executable)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "canopy._embed_worker",
                "--input",
                input_path,
                "--output",
                output_path,
                "--model_path",
                resolved_path,
                "--model_type",
                model_type,
                "--character",
                character,
                "--device",
                device,
                "--batch_size",
                str(bs),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == -9:
            raise RuntimeError(
                f"Embedding subprocess killed (OOM). model_type={model_type}, "
                f"model_path={resolved_path}, device={device}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Embedding subprocess failed (rc={result.returncode}): {result.stderr}"
            )
        arr = np.load(output_path)
        if arr.shape[0] != len(texts):
            raise RuntimeError(
                f"Shape mismatch: expected {len(texts)} rows, got {arr.shape[0]}"
            )
        return arr
    finally:
        for path in (input_path, output_path):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as e:
                log.warning("Failed to clean up temp file %s: %s", path, e)


# Module-level model paths — set by init_models(), loaded on demand
_surface_embedder_path: str | None = None
_generator_embedder_path: str | None = None
_device: torch.device | None = None
_model_lock = threading.Lock()


def init_models(
    surface_embedder_path: str,
    generator_embedder_path: str,
    device: torch.device,
) -> None:
    """Store model paths and device for lazy sequential loading.

    Models are loaded on demand inside select_cluster_centers() and unloaded
    immediately after encoding to free VRAM.
    """
    global _surface_embedder_path, _generator_embedder_path, _device
    _device = device
    _surface_embedder_path = surface_embedder_path
    _generator_embedder_path = generator_embedder_path


def _load_surface_model() -> Any:
    """Load surface embedding model into VRAM."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        _surface_embedder_path,
        device=_device,
        model_kwargs={"torch_dtype": torch.float16},
    )


def _load_generator_model() -> tuple[Any, Any]:
    """Load generator model and tokenizer into VRAM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_generator_embedder_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        _generator_embedder_path,
        torch_dtype=torch.float16,
    ).to(_device)
    return model, tokenizer


def _unload_model(*models: Any) -> None:
    """Delete model references and free VRAM."""
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def surface_encode(texts: list[str], model: Any) -> np.ndarray:
    """Encode texts using a surface (sentence-transformer) model."""
    with _model_lock:
        with torch.no_grad():
            embedding = model.encode(texts, convert_to_tensor=True)
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.detach().cpu().numpy()


def generative_encode(texts: list[str], model: Any, tokenizer: Any) -> np.ndarray:
    """Encode texts using the generative model's last hidden state."""
    with _model_lock:
        with torch.no_grad():
            embedding = model(
                **tokenizer(texts, return_tensors="pt", padding=True).to(_device),
                output_hidden_states=True,
            ).hidden_states[-1][:, -1]
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.detach().cpu().numpy()


def select_cluster_centers(
    character: str,
    pairs: list[dict[str, Any]],
    n_in_cluster_case: int = 16,
    n_in_cluster_sample: int = 8,
    n_max_cluster: int = 8,
    bs: int = 8,
    *,
    embedding_cache: EmbeddingCache | None = None,
) -> list[list[dict[str, Any]]]:
    """Cluster scene-action pairs and return representative samples per cluster.

    When ``embedding_cache`` is provided, uses pre-computed embeddings (no GPU
    model loading). Otherwise falls back to sequential model loading.

    Args:
        embedding_cache: Pre-computed embeddings from ``precompute_embeddings()``.
            When provided, pairs must have ``_embed_idx`` keys mapping to cache rows.
    """
    if not pairs:
        return []

    from canopy.cluster import KMeansCluster, select_representative_samples

    # Fast path: use pre-computed embeddings (no GPU model loading)
    if embedding_cache is not None:
        missing = [i for i, p in enumerate(pairs) if "_embed_idx" not in p]
        if missing:
            raise RuntimeError(
                f"{len(missing)} pairs missing '_embed_idx'. "
                "Stamp pairs with _embed_idx before passing embedding_cache."
            )
        indices = [p["_embed_idx"] for p in pairs]
        # Type + bounds check: ensure all indices are valid ints for the cache
        cache_size = len(embedding_cache.surface)
        bad = [
            (i, idx) for i, idx in enumerate(indices)
            if not isinstance(idx, int) or not (0 <= idx < cache_size)
        ]
        if bad:
            raise RuntimeError(
                f"{len(bad)} pairs have out-of-range _embed_idx (cache size={cache_size}): "
                f"{bad[:5]}"
            )
        subset = embedding_cache.subset(indices)
        document_embeddings = subset.document

        clusterer = KMeansCluster(n_in_cluster_case=n_in_cluster_case, n_max_cluster=n_max_cluster)
        _, centroids = clusterer.fit_predict(document_embeddings)
        return select_representative_samples(pairs, document_embeddings, centroids, n_samples=n_in_cluster_sample)

    # Legacy path: load models sequentially (backward compatible for small models)
    if _device is None:
        raise RuntimeError("Embedding models not initialized — call init_models() first")

    actions = [pair["action"] for pair in pairs]
    scenes = [pair["scene"] for pair in pairs]

    # Phase 1: Surface encoding (load -> encode -> unload)
    # Lock wraps entire load/encode/unload cycle so parallel threads
    # don't load the same 16GB model simultaneously and OOM.
    with _model_lock:
        log.info("Loading surface embedder...")
        surface_model = _load_surface_model()
        surface_embeddings: list[np.ndarray] = []
        for idx in tqdm(range(0, len(actions), bs), desc="Surface encoding...", leave=True):
            with torch.no_grad():
                embedding = surface_model.encode(actions[idx : idx + bs], convert_to_tensor=True)
                embedding = F.normalize(embedding, p=2, dim=1)
                surface_embeddings.append(embedding.detach().cpu().numpy())
        surface_embeddings_np = np.concatenate(surface_embeddings, 0)
        _unload_model(surface_model)
        log.info("Surface encoding complete, model unloaded")

    # Phase 2: Generative encoding (load -> encode -> unload)
    with _model_lock:
        log.info("Loading generator embedder...")
        gen_model, gen_tokenizer = _load_generator_model()
        gen_embeddings: list[np.ndarray] = []
        for idx in tqdm(range(0, len(scenes), bs), desc="Generative encoding...", leave=True):
            scenes_batch = [s + f"\n\nThus, {character} decides to" for s in scenes[idx : idx + bs]]
            with torch.no_grad():
                embedding = gen_model(
                    **gen_tokenizer(scenes_batch, return_tensors="pt", padding=True).to(_device),
                    output_hidden_states=True,
                ).hidden_states[-1][:, -1]
                embedding = F.normalize(embedding, p=2, dim=1)
                gen_embeddings.append(embedding.detach().cpu().numpy())
        gen_embeddings_np = np.concatenate(gen_embeddings, 0)
        _unload_model(gen_model, gen_tokenizer)
        log.info("Generative encoding complete, model unloaded")

    # Phase 3: Clustering (CPU only — no models in VRAM)
    document_embeddings: np.ndarray = np.concatenate([gen_embeddings_np, surface_embeddings_np], -1)

    clusterer = KMeansCluster(n_in_cluster_case=n_in_cluster_case, n_max_cluster=n_max_cluster)
    _, centroids = clusterer.fit_predict(document_embeddings)

    return select_representative_samples(pairs, document_embeddings, centroids, n_samples=n_in_cluster_sample)
