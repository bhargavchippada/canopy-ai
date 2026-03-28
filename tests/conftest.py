"""Pytest configuration — integration test handling."""

from __future__ import annotations

import torch


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless explicitly requested."""
    run_integration = config.getoption("-m", default="") and "integration" in config.getoption("-m", default="")

    if not run_integration:
        skip_integration = __import__("pytest").mark.skip(reason="integration tests skipped by default (use -m integration)")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip GPU tests if no CUDA
    if not torch.cuda.is_available():
        skip_gpu = __import__("pytest").mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
