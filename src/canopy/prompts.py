"""LLM prompt templates and hypothesis generation for CDT construction."""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from canopy.llm import batch_generate, extract_json, generate

log = logging.getLogger(__name__)


def make_hypothesis_prompt(
    cluster: list[dict[str, Any]],
    character: str,
    goal_topic: str,
    established_statements: list[str],
    gate_path: list[str],
    k: int = 3,
) -> str:
    """Build the hypothesis prompt for a cluster. Does NOT call LLM."""
    action_scene_context = "\n\n".join(
        ["# Scene:\n" + pair["scene"] + "\n# Action:\n" + pair["action"] for pair in cluster],
    )
    established_statement_verbalized = "\n".join(established_statements) if established_statements else "N/A"
    gate_path_verbalized = "\n".join(gate_path) if gate_path else "N/A"

    return f"""# Scene-Action Pairs
{action_scene_context}

# Established Statements
{established_statement_verbalized}

# Already Proposed Common Points
{gate_path_verbalized}

# Task

Your task is to build the grounding logic for an AI system to understand the behavior of {character} (Current topic: "{goal_topic}"), assert the AI system has no prior knowledge of {character}.
To do this, please propose hypotheses for the general behavior logic of {character} based on the given action-scene pairs, complete the task step by step:

1. What's the main feature of {character}'s behavior  (Focus on the current topic: "{goal_topic}") shown in the given scene-action pairs, **other than the already established statements**?

2. Summarize {k} potential common points (grounding statements) of the actions taken by {character} in the given scenes about the focused topic: "{goal_topic}", **which is other than the already established statements**.
- The grounding statements should be general, avoiding too specific action descriptions. (except when it's a common skill of the character)
- Consider the grounding statements in a general way.
- The grounding statements should be concise, informative, and general sentences.
- Never be assertive! Always make objective description of the character rather than making assertive causal relations.
- IMPORTANT: Include at least one statement about {character}'s ATYPICAL or less frequent reactions (e.g., moments of confusion, hesitation, quiet responses, or subdued behavior). Not every response from {character} follows their dominant pattern.

3. Summarize {k} potential common points of the given scenes that trigger each behavior, **which should be different from already proposed common points.**
- The question should be simple, not ambiguous, and specific to a subset of scenes rather than always applicable.
- Focus on the **next action** when asking! Don't ask whether certain event is involved, instead ask whether the scene might trigger potential behavior for {character}'s **next action**.
- Directly include "{character}'s next action" in the question!

4. Output the hypothesized scene-action triggers in the following Python code block format:
```python
action_hypotheses = [] # A list of syntactically complete statements (always mentioning {character})
scene_check_hypotheses = [] # A list of syntactically complete questions to check the given scene (always mentioning {character})
```
"""


def _extract_python_list(code: str, var_name: str) -> list[str]:
    """Extract a list assignment from Python code using ast.literal_eval.

    Finds `var_name = [...]` and safely parses the list literal.
    Never uses exec/eval.
    """
    # Match var_name = [...] allowing multiline lists
    pattern = rf"^{re.escape(var_name)}\s*=\s*(\[.*?\])"
    match = re.search(pattern, code, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find '{var_name} = [...]' in code block")
    try:
        result = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Failed to parse '{var_name}' list: {exc}") from exc
    if not isinstance(result, list):
        raise ValueError(f"'{var_name}' is not a list")
    return [str(item) for item in result]


def _extract_python_block(response: str) -> str | None:
    """Extract code between ```python and ``` markers."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    return match.group(1) if match else None


def parse_hypothesis_response(response: str) -> tuple[list[str], list[str]]:
    """Parse a hypothesis LLM response into (action_hypotheses, scene_check_hypotheses).

    Tries Python code block format first (paper format), falls back to JSON.
    """
    # Try Python code block format first
    code = _extract_python_block(response)
    if code is not None:
        try:
            action_hyps = _extract_python_list(code, "action_hypotheses")
            scene_hyps = _extract_python_list(code, "scene_check_hypotheses")
            min_len = min(len(action_hyps), len(scene_hyps))
            return action_hyps[:min_len], scene_hyps[:min_len]
        except ValueError:
            log.debug("Python code block parsing failed, trying JSON fallback")

    # Fallback to JSON format
    parsed = extract_json(response)
    action_hyps = parsed.get("action_hypotheses", [])
    scene_hyps = parsed.get("scene_check_hypotheses", [])
    min_len = min(len(action_hyps), len(scene_hyps))
    return action_hyps[:min_len], scene_hyps[:min_len]


def make_hypotheses_batch(
    clusters: list[list[dict[str, Any]]],
    character: str,
    goal_topic: str,
    established_statements: list[str],
    gate_path: list[str],
    k: int = 3,
    model: str | None = None,
) -> tuple[list[str], list[str]]:
    """Generate hypotheses for ALL clusters in parallel."""
    prompts = [
        make_hypothesis_prompt(cluster, character, goal_topic, established_statements, gate_path, k)
        for cluster in clusters
    ]
    items = [(str(i), prompt) for i, prompt in enumerate(prompts)]
    result = batch_generate(items, model=model)

    statement_candidates: list[str] = []
    gates: list[str] = []
    parse_failures = 0
    for cluster_id, response in result.successes.items():
        try:
            action_hyps, scene_hyps = parse_hypothesis_response(response)
            statement_candidates.extend(action_hyps)
            gates.extend(scene_hyps)
        except ValueError as exc:
            log.warning("Cluster %s hypothesis parsing failed: %s", cluster_id, exc)
            parse_failures += 1

    llm_drops = len(result.exhausted_ids) + len(result.dropped_ids)
    total_failed = llm_drops + parse_failures
    if total_failed > 0:
        log.warning(
            "make_hypotheses_batch: %d/%d clusters produced no hypotheses "
            "(llm_drops=%d, parse_failures=%d)",
            total_failed,
            len(prompts),
            llm_drops,
            parse_failures,
        )

    return statement_candidates, gates


def merge_similar_hypotheses(
    gates: list[str],
    statement_candidates: list[str],
    *,
    model: str | None = None,
) -> tuple[list[str], list[str]]:
    """Deduplicate hypothesis pairs via a single LLM call.

    Sends all hypothesis pairs to the LLM with strict dedup instructions.
    Returns deduplicated pairs with semantic duplicates merged — the LLM
    picks the best representative for each group of similar statements.

    One LLM call regardless of input size (vs O(n²) pairwise comparisons).

    Args:
        gates: Scene-check hypothesis questions (aligned with statement_candidates).
        statement_candidates: Action hypothesis statements.
        model: LLM model for dedup. Uses default if None.
    """
    if len(gates) != len(statement_candidates):
        raise ValueError(
            f"gates ({len(gates)}) and statement_candidates ({len(statement_candidates)}) "
            "must have the same length"
        )
    if len(statement_candidates) <= 1:
        return gates, statement_candidates

    # Build numbered list for the LLM
    pairs_text = "\n".join(
        f"{i}: action: {stmt} | scene_check: {gate}"
        for i, (gate, stmt) in enumerate(zip(gates, statement_candidates))
    )

    prompt = f"""# Task: Merge & Deduplicate Hypothesis Pairs

Below are {len(statement_candidates)} hypothesis pairs (index: action | scene_check).
Merge semantic duplicates into improved combined versions.

<input_pairs>
{pairs_text}
</input_pairs>

## Instructions
1. Group pairs that describe the SAME or OVERLAPPING behavioral pattern
2. For each group, produce ONE merged pair that combines the best elements from all members:
   - The merged action hypothesis should be the most precise and complete version
   - The merged scene_check should be the most discriminative question
3. Pairs with NO duplicates pass through unchanged
4. The output should have FEWER pairs than the input (or equal if no duplicates)
5. Every distinct behavioral pattern from the input must be represented in the output

## Constraints
- action: single concise sentence, max 20 words, non-assertive
- scene_check: single question containing the character's name and "next action"

## Output Format (JSON only)
```json
{{"merged_pairs": [
  {{"scene_check_hypothesis": "...", "action_hypothesis": "..."}},
  {{"scene_check_hypothesis": "...", "action_hypothesis": "..."}}
]}}
```

Return ONLY the JSON."""

    try:
        response = generate(prompt, model=model)
        parsed = extract_json(response)
        merged_pairs = parsed["merged_pairs"]

        if not merged_pairs or not isinstance(merged_pairs, list):
            log.warning("Merge returned empty/invalid pairs, keeping originals")
            return gates, statement_candidates

        new_gates = [p["scene_check_hypothesis"] for p in merged_pairs]
        new_stmts = [p["action_hypothesis"] for p in merged_pairs]

        n_merged = len(statement_candidates) - len(new_stmts)
        if n_merged > 0:
            log.info("Merged %d hypotheses → %d via LLM dedup (-%d)",
                     len(statement_candidates), len(new_stmts), n_merged)
        elif n_merged < 0:
            log.warning("LLM produced MORE pairs than input (%d → %d), using originals",
                        len(statement_candidates), len(new_stmts))
            return gates, statement_candidates

        return new_gates, new_stmts
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("Hypothesis merge failed, keeping all: %s", exc)
        return gates, statement_candidates


def summarize_triggers(
    character: str,
    gates: list[str],
    statement_candidates: list[str],
    model: str | None = None,
) -> tuple[list[str], list[str]]:
    """Summarize and compress hypothesis pairs into top 8 if needed."""
    paired_hypotheses = [
        {"scene_check_hypothesis": gate, "action_hypothesis": stmt} for gate, stmt in zip(gates, statement_candidates)
    ]

    if len(paired_hypotheses) > 8:
        boost_prompt = f"""
# Task: Summarize & Compress Scene–Action Hypothesis Pairs into Top 8

You are given a list of paired hypotheses. Each pair contains:
- "scene_check_hypothesis": a question about {character}'s next action
- "action_hypothesis": a general behavioral grounding statement about {character}

Input pairs:
{paired_hypotheses}

## Goal
Produce a rewritten, deduplicated, and compressed set of **exactly 8** pairs that capture the **most important** and **most general** behavioral grounding logic for {character}.

Rewriting is allowed and encouraged to increase:
- generality
- coverage across different subsets of scenes
- clarity
- non-assertiveness

## Selection Principles (prioritized)
1. **Coverage**: The 8 pairs should collectively cover the widest range of distinct behavioral patterns and distinct scene triggers.
2. **Centrality**: Prefer pairs that reflect recurring or core behaviors across many scene-action pairs.
3. **Specificity without overfitting**: Keep statements general; only keep a specific skill/ability if it appears repeatedly and broadly.
4. **Non-redundancy**: Each of the 8 pairs must represent a meaningfully different behavior/trigger from the others.
5. **Pair coherence**: The scene_check_hypothesis must plausibly test for the corresponding action_hypothesis (do not mismatch them).

## Dedup & Merge Rules
- You may merge multiple similar input pairs into one rewritten pair.
- If two candidate pairs overlap heavily in either the action or the scene question, combine them into a single more general pair.
- Do not preserve original wording when a clearer/general rewrite is possible.

## Constraints to Preserve
### scene_check_hypothesis
- Must be a **single, simple question**
- Must explicitly contain the exact phrase: "{character}'s next action"
- Must target **scene conditions** that could trigger that next action
- Must be applicable to a **subset** of scenes (not a universal always-true condition)

### action_hypothesis
- Must be a **single, concise sentence** of **at most 15 words**
- Must be **non-assertive** (use "may", "tends to", "often appears to", "is described as", "is observed as", etc.)
- Must be **specific enough to be FALSE in at least 30% of scenes** — avoid universal character descriptions that are always true
- Must reference a **specific behavioral trigger or context**, not a general personality trait
- Must not invent backstory or assume prior knowledge

## Output Format (JSON only)
Return exactly 8 pairs:

```json
{{
  "top8_pairs": [
    {{
      "scene_check_hypothesis": "...",
      "action_hypothesis": "..."
    }},
    {{
      "scene_check_hypothesis": "...",
      "action_hypothesis": "..."
    }}
  ]
}}
```

## Quality Checklist (must satisfy)
* Exactly 8 pairs.
* No two action_hypothesis items mean the same thing.
* No two scene_check_hypothesis questions ask the same trigger.
* Each scene_check_hypothesis clearly tests for {character}'s next action.
* Each action_hypothesis is general, grounded, and non-assertive.
"""
        try:
            response = generate(boost_prompt, model=model)
            parsed = extract_json(response)
            paired_hypotheses = parsed["top8_pairs"]
        except (ValueError, KeyError) as exc:
            log.warning("Summarize compression failed, using first 8: %s", exc)
            paired_hypotheses = paired_hypotheses[:8]

    return (
        [pair["scene_check_hypothesis"] for pair in paired_hypotheses],
        [pair["action_hypothesis"] for pair in paired_hypotheses],
    )
