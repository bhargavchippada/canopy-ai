"""LLM prompt templates and hypothesis generation for CDT construction."""

from __future__ import annotations

import logging

from canopy.llm import extract_json, generate, generate_many

log = logging.getLogger(__name__)


def make_hypothesis_prompt(
    cluster: list[dict],
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
    established_statement_verbalized = "\n".join(established_statements) if len(established_statements) > 0 else "N/A"
    gate_path_verbalized = "\n".join(gate_path) if len(gate_path) > 0 else "N/A"

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

3. Summarize {k} potential common points of the given scenes that trigger each behavior, **which should be different from already proposed common points.**
- The question should be simple, not ambiguous, and specific to a subset of scenes rather than always applicable.
- Focus on the **next action** when asking! Don't ask whether certain event is involved, instead ask whether the scene might trigger potential behavior for {character}'s **next action**.
- Directly include "{character}'s next action" in the question!

4. Output the hypothesized scene-action triggers in the following JSON format:
```json
{{
  "action_hypotheses": [],
  "scene_check_hypotheses": []
}}
```
Where action_hypotheses is a list of syntactically complete statements (always mentioning {character})
and scene_check_hypotheses is a list of syntactically complete questions to check the given scene (always mentioning {character}).
"""


def parse_hypothesis_response(response: str) -> tuple[list[str], list[str]]:
    """Parse a hypothesis LLM response into (action_hypotheses, scene_check_hypotheses)."""
    parsed = extract_json(response)
    action_hyps = parsed.get("action_hypotheses", [])
    scene_hyps = parsed.get("scene_check_hypotheses", [])
    min_len = min(len(action_hyps), len(scene_hyps))
    return action_hyps[:min_len], scene_hyps[:min_len]


def make_hypotheses_batch(
    clusters: list[list[dict]],
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
    responses = generate_many(prompts, model=model)
    statement_candidates: list[str] = []
    gates: list[str] = []
    for i, response in enumerate(responses):
        try:
            action_hyps, scene_hyps = parse_hypothesis_response(response)
            statement_candidates.extend(action_hyps)
            gates.extend(scene_hyps)
        except (ValueError, KeyError) as exc:
            log.warning("Cluster %d hypothesis parsing failed: %s", i, exc)
            continue
    return statement_candidates, gates


def summarize_triggers(
    character: str,
    gates: list[str],
    statement_candidates: list[str],
    model: str | None = None,
) -> tuple[list[str], list[str]]:
    """Summarize and compress hypothesis pairs into top 8 if needed."""
    paired_hypotheses = [
        {"scene_check_hypothesis": gate, "action_hypothesis": stmt}
        for gate, stmt in zip(gates, statement_candidates)
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

## Output Format (JSON only)
Return exactly 8 pairs:

```json
{{
  "top8_pairs": [
    {{
      "scene_check_hypothesis": "...",
      "action_hypothesis": "..."
    }}
  ]
}}
```

## Quality Checklist
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
