"""Backwards-compatible re-export from canopy.llm.

All new code should import from canopy.llm directly.
"""

from canopy.llm import (  # noqa: F401
    DEFAULT_MODEL,
    EVAL_MODEL,
    HYPOTHESIS_MODEL,
    ClaudeCodeAdapter,
    LLMAdapter,
    extract_json,
    generate,
    generate_many,
    set_adapter,
)
