"""Embedding models and clustering for CDT construction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

# Module-level model references — set by init_models()
_surface_embedding = None
_generator_embedding = None
_generator_tokenizer = None
_device = None


def init_models(
    surface_embedder_path: str,
    generator_embedder_path: str,
    device: torch.device,
) -> None:
    """Load embedding models onto the given device."""
    global _surface_embedding, _generator_embedding, _generator_tokenizer, _device

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = device
    _surface_embedding = SentenceTransformer(
        surface_embedder_path, device=device, model_kwargs={"torch_dtype": torch.float16},
    )
    _generator_tokenizer = AutoTokenizer.from_pretrained(generator_embedder_path)
    _generator_tokenizer.padding_side = "left"
    _generator_embedding = AutoModelForCausalLM.from_pretrained(
        generator_embedder_path, torch_dtype=torch.float16,
    ).to(device)


def generative_encode(texts: list[str]) -> np.ndarray:
    """Encode texts using the generative model's last hidden state."""
    if _generator_embedding is None:
        raise RuntimeError("Embedding models not initialized — call init_models() first")
    embedding = _generator_embedding(
        **_generator_tokenizer(texts, return_tensors="pt", padding=True).to(_device),
        output_hidden_states=True,
    ).hidden_states[-1][:, -1]
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.detach().cpu().numpy()


def surface_encode(texts: list[str]) -> np.ndarray:
    """Encode texts using the surface (sentence-transformer) model."""
    if _surface_embedding is None:
        raise RuntimeError("Embedding models not initialized — call init_models() first")
    embedding = _surface_embedding.encode(texts, convert_to_tensor=True)
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.detach().cpu().numpy()


def select_cluster_centers(
    character: str,
    pairs: list[dict],
    n_in_cluster_case: int = 16,
    n_in_cluster_sample: int = 8,
    n_max_cluster: int = 8,
    bs: int = 8,
) -> list[list[dict]]:
    """Cluster scene-action pairs and return representative samples per cluster."""
    actions = [pair["action"] for pair in pairs]
    scenes = [pair["scene"] for pair in pairs]

    with torch.no_grad():
        document_embeddings = []
        for idx in tqdm(range(0, len(actions), bs), desc="Embedding...", leave=True):
            scenes_batch = [scene + f"\n\nThus, {character} decides to" for scene in scenes[idx : idx + bs]]
            actions_batch = actions[idx : idx + bs]
            document_embeddings.append(
                np.concatenate([generative_encode(scenes_batch), surface_encode(actions_batch)], -1),
            )
        document_embeddings = np.concatenate(document_embeddings, 0)

    n_clusters = min(int(np.ceil(document_embeddings.shape[0] / n_in_cluster_case)), n_max_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(document_embeddings)

    centroids = kmeans.cluster_centers_
    clusters: list[list[dict]] = []

    for centroid in centroids:
        distances = (((document_embeddings - centroid) ** 2).sum(-1)) ** 0.5
        cluster = [pairs[idx] for idx in distances.argsort(-1)[:n_in_cluster_sample]]
        clusters.append(cluster)

    return clusters
