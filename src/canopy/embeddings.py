"""Embedding models and clustering for CDT construction.

Models are loaded sequentially in select_cluster_centers() to support
VRAM-constrained GPUs (e.g. two 8B models on a 32GB card).
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

log = logging.getLogger(__name__)

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
) -> list[list[dict[str, Any]]]:
    """Cluster scene-action pairs and return representative samples per cluster.

    Models are loaded sequentially — surface embedder first, then generator —
    so that two large models never coexist in VRAM simultaneously.
    """
    if _device is None:
        raise RuntimeError("Embedding models not initialized — call init_models() first")

    actions = [pair["action"] for pair in pairs]
    scenes = [pair["scene"] for pair in pairs]

    from canopy.cluster import KMeansCluster, select_representative_samples

    # Phase 1: Surface encoding (load -> encode -> unload)
    log.info("Loading surface embedder...")
    surface_model = _load_surface_model()
    surface_embeddings: list[np.ndarray] = []
    for idx in tqdm(range(0, len(actions), bs), desc="Surface encoding...", leave=True):
        surface_embeddings.append(surface_encode(actions[idx : idx + bs], surface_model))
    surface_embeddings_np = np.concatenate(surface_embeddings, 0)
    _unload_model(surface_model)
    log.info("Surface encoding complete, model unloaded")

    # Phase 2: Generative encoding (load -> encode -> unload)
    log.info("Loading generator embedder...")
    gen_model, gen_tokenizer = _load_generator_model()
    gen_embeddings: list[np.ndarray] = []
    for idx in tqdm(range(0, len(scenes), bs), desc="Generative encoding...", leave=True):
        scenes_batch = [s + f"\n\nThus, {character} decides to" for s in scenes[idx : idx + bs]]
        gen_embeddings.append(generative_encode(scenes_batch, gen_model, gen_tokenizer))
    gen_embeddings_np = np.concatenate(gen_embeddings, 0)
    _unload_model(gen_model, gen_tokenizer)
    log.info("Generative encoding complete, model unloaded")

    # Phase 3: Clustering (CPU only — no models in VRAM)
    document_embeddings: np.ndarray = np.concatenate([gen_embeddings_np, surface_embeddings_np], -1)

    clusterer = KMeansCluster(n_in_cluster_case=n_in_cluster_case, n_max_cluster=n_max_cluster)
    _, centroids = clusterer.fit_predict(document_embeddings)

    return select_representative_samples(pairs, document_embeddings, centroids, n_samples=n_in_cluster_sample)
