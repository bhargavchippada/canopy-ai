"""Standalone subprocess entry point for embedding computation.

Invoked via ``python -m canopy._embed_worker``. Loads a model, encodes
texts read from a JSON file, writes embeddings as ``.npy``, then exits
so the OS reclaims all GPU memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the embed worker."""
    parser = argparse.ArgumentParser(
        description="Encode texts into embeddings and save as .npy",
    )
    parser.add_argument("--input", required=True, help="Path to JSON file containing list of strings")
    parser.add_argument("--output", required=True, help="Path to write output .npy file")
    parser.add_argument("--model_path", required=True, help="Path to the model (local directory)")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=("surface", "generator"),
        help="Model type: 'surface' (SentenceTransformer) or 'generator' (CausalLM)",
    )
    parser.add_argument("--character", default=None, help="Character name (required for generator type)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device string (default: cuda:0)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding (default: 8)")
    return parser.parse_args(argv)


def encode_surface(
    texts: list[str],
    model_path: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode texts using a SentenceTransformer model.

    Args:
        texts: Input strings to encode.
        model_path: Local path to the SentenceTransformer model.
        device: CUDA device string.
        batch_size: Number of texts per batch.

    Returns:
        L2-normalized embeddings as a numpy array of shape (len(texts), dim).
    """
    from sentence_transformers import SentenceTransformer

    model: Any = SentenceTransformer(
        model_path,
        device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )

    embeddings_list: list[np.ndarray] = []
    with torch.no_grad():
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx : idx + batch_size]
            embedding = model.encode(batch, convert_to_tensor=True)
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings_list.append(embedding.detach().cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)


def encode_generator(
    texts: list[str],
    model_path: str,
    device: str,
    batch_size: int,
    character: str,
) -> np.ndarray:
    """Encode texts using a CausalLM model's last hidden state.

    Each text is suffixed with ``"\\n\\nThus, {character} decides to"``
    before encoding.

    Args:
        texts: Input strings to encode.
        model_path: Local path to the CausalLM model.
        device: CUDA device string.
        batch_size: Number of texts per batch.
        character: Character name for the scene suffix.

    Returns:
        L2-normalized embeddings as a numpy array of shape (len(texts), dim).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    suffixed_texts = [t + f"\n\nThus, {character} decides to" for t in texts]

    embeddings_list: list[np.ndarray] = []
    with torch.no_grad():
        for idx in range(0, len(suffixed_texts), batch_size):
            batch = suffixed_texts[idx : idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][:, -1]
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings_list.append(embedding.detach().cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)


def main(argv: list[str] | None = None) -> None:
    """Load model, encode texts from JSON input, save embeddings as .npy."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    log.info("Reading input texts from %s", input_path)
    texts: list[str] = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        log.error("Input JSON must be a list of strings")
        sys.exit(1)

    log.info(
        "Encoding %d texts with model_type=%s, model_path=%s, device=%s, batch_size=%d",
        len(texts),
        args.model_type,
        args.model_path,
        args.device,
        args.batch_size,
    )

    if args.model_type == "surface":
        embeddings = encode_surface(
            texts=texts,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.model_type == "generator":
        if args.character is None:
            log.error("--character is required for generator model type")
            sys.exit(1)
        embeddings = encode_generator(
            texts=texts,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            character=args.character,
        )
    else:
        log.error("Unknown model_type: %s", args.model_type)
        sys.exit(1)

    log.info("Saving embeddings with shape %s to %s", embeddings.shape, output_path)
    np.save(output_path, embeddings)
    log.info("Done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Embed worker failed")
        sys.exit(1)
