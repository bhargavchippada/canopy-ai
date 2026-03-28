# PERSONAMEM Evaluation Harness Design

**Date:** 2026-03-28
**Paper:** PERSONAMEM: A Benchmark for Dynamic User Profiling (arxiv 2504.14225)
**Goal:** Evaluate CDT-based grounding on dynamic user preference prediction

---

## 1. Task Format

PERSONAMEM uses **multiple-choice question (MCQ)** format — fundamentally different from CDT paper's NLI scoring.

### Structure
```
Given:
  - Conversation history (N prior conversations with a persona)
  - In-situ user query (a question about preferences within conversation N+1)

Task:
  - Select the correct personalized response from 4 options (A/B/C/D)
  - One option reflects the persona's actual preference
  - Three are plausible distractors

Scoring:
  - Accuracy (% correct) — chance = 25%
  - NOT NLI (entailed/neutral/contradicted)
```

### Key Differences from CDT Benchmark

| Aspect | CDT Paper (NLI) | PERSONAMEM (MCQ) |
|---|---|---|
| Task | Predict next action | Select correct preference |
| Scoring | NLI (100/50/0) | Accuracy (correct/incorrect) |
| Context | Scene (10 prior actions) | Full conversation history |
| Ground truth | Single action | 1 of 4 choices |
| Baseline | Vanilla ~55% | Frontier LLMs ~50% |
| Data type | Narrative/dialogue | Multi-turn conversation |

---

## 2. CDT Integration Approach

### Pipeline

```
Conversation history (before query)
    → Extract observations (each turn = BehavioralObservation)
    → Build CDT from observations (canopy pipeline)
    → In-situ query arrives
    → Traverse CDT with query context → grounding statements
    → Score each MCQ option against grounding
    → Select highest-scoring option
```

### Approach A: LLM Selection with CDT Grounding

```
System: You are evaluating which response best matches this user's preferences.

# Background Knowledge (CDT Grounding)
{traversed_statements}

# Conversation History
{recent_conversation_context}

# Question
{query}

# Options
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

Select the option that best matches the user's established preferences.
Output only the letter (A/B/C/D).
```

**Pros:** Leverages LLM reasoning + CDT grounding. Natural extension of CDT benchmark approach.
**Cons:** LLM may ignore grounding and use its own reasoning. Expensive per query.

### Approach B: Embedding Similarity

```
For each MCQ option:
    1. Embed the option text
    2. Embed each CDT grounding statement
    3. Score = max cosine similarity between option and any grounding statement
    4. Select option with highest aggregate score
```

**Pros:** No LLM call needed. Deterministic. Fast.
**Cons:** Semantic similarity ≠ preference alignment. May miss nuanced reasoning.

### Approach C: Hybrid (Recommended)

1. Use Approach B to pre-filter: rank options by embedding similarity to CDT statements
2. Use Approach A with the ranked options + CDT grounding for final selection
3. If Approach B gives high-confidence answer (top option >> others), skip LLM call

---

## 3. Baseline Comparison

### Published Results (from PERSONAMEM paper)

| Method | Accuracy | Note |
|---|---|---|
| Random | 25.0% | 4-way MCQ chance |
| GPT-4.5 | ~50% | Frontier model, no memory |
| o1 | ~50% | Reasoning model |
| Gemini | ~50% | Frontier model |
| RAG | ~55% | Retrieval-augmented |
| **CDT (ours, target)** | **>55%** | **Structured preference grounding** |

### Why CDT Should Beat Raw LLM

1. **Structured memory**: CDT encodes validated behavioral patterns, not raw conversation
2. **Situation-specific retrieval**: Gate traversal selects relevant statements per query
3. **Temporal awareness**: T-CDT can weight recent preferences over old ones
4. **Explicit validation**: CDT statements are NLI-validated, not hallucinated summaries

### Why CDT Might Struggle

1. **Domain shift**: CDT was designed for character RP, not user preference prediction
2. **Sparse data**: PERSONAMEM personas have limited conversation history
3. **Dynamic preferences**: Users change their minds — static CDT may miss this
4. **MCQ format**: CDT produces grounding statements, not option rankings

---

## 4. In-Sample Test Plan

### Phase 1: Data Exploration (1 persona)

```python
# Pick persona_id=0
# Load their conversations
# Examine: how many conversations? How many MCQ pairs? What topics?
persona = load_persona(0)
print(f"Conversations: {len(persona.conversations)}")
print(f"MCQ pairs: {len(persona.mcq_pairs)}")
```

### Phase 2: CDT Construction (first 10 conversations)

```python
# Convert conversations to BehavioralObservations
observations = []
for conv in persona.conversations[:10]:
    for turn in conv.turns:
        if turn.role == "user":
            obs = BehavioralObservation(
                scene=turn.context,      # prior assistant response + conversation state
                action=turn.message,     # user's actual response
                actor="user",
                timestamp=turn.timestamp,
            )
            observations.append(obs)

# Build CDT
topic2cdt, rel_topic2cdt = build_character_profile(
    observations, character="user", config=CDTConfig(max_depth=4)
)
```

### Phase 3: Manual Quality Check

For the CDT built from 10 conversations:
1. Print all statements — do they capture real preferences?
2. Are statements falsifiable (specific triggers, not universal)?
3. Do gate conditions match meaningful preference contexts?
4. Compare statement count to CDT benchmark: expect fewer (less data)

### Phase 4: MCQ Evaluation (5 pairs)

```python
# Test on 5 MCQ pairs from this persona
correct = 0
for mcq in persona.mcq_pairs[:5]:
    # Traverse CDT with query context
    grounding = traverse_cdt(topic2cdt, mcq.query_context)

    # Score each option (Approach A or B)
    selected = select_option(grounding, mcq.options, mcq.query)

    if selected == mcq.correct_answer:
        correct += 1

    # Manual inspection
    print(f"Query: {mcq.query}")
    print(f"Grounding: {grounding[:200]}")
    print(f"Selected: {selected}, Correct: {mcq.correct_answer}")
```

### Success Criteria

| Metric | Target | Rationale |
|---|---|---|
| CDT builds successfully | Yes | Proves domain-agnostic adapter works |
| Statements capture preferences | ≥50% relevant | Manual inspection |
| MCQ accuracy (5 pairs) | >25% (>chance) | CDT adds value over random |
| MCQ accuracy vs no-grounding | +5% improvement | CDT grounding helps selection |

---

## 5. Implementation Checklist

- [ ] PERSONAMEM data loader (cody building)
- [ ] BehavioralObservation adapter for conversation turns
- [ ] MCQ evaluation script with CDT traversal
- [ ] Approach A prompt template
- [ ] Approach B embedding scorer
- [ ] Baseline: LLM without CDT grounding (for comparison)
- [ ] Results logging with per-question details

---

## 6. In-Sample Test Results (2026-03-28)

**CDT alone: 1/5 correct (20%) — below random (25%).**

The task is factual memory recall, not behavioral prediction. CDT captures behavioral *patterns* but PersonaMem MCQs test recall of *specific facts* and *preference changes*.

### Per-Query-Type Analysis: CDT Relevance

| # | Query Type | Nature | LLM Baseline | CDT Helps? | Why |
|---|---|---|---|---|---|
| 1 | Recall user-shared facts | **Factual** | 60-70% | **No** | Requires retrieving specific mentioned items — CDT generalizes away specifics |
| 2 | Suggest new ideas | **Behavioral** | 30-40% | **Yes** | CDT knows preference patterns → can exclude known items, suggest aligned ones |
| 3 | Acknowledge latest preferences | **Temporal-factual** | 40-50% | **Partial** | T-CDT temporal weighting could help; static CDT misses recency |
| 4 | Track preference evolution | **Temporal-factual** | 60-70% | **Partial** | CDT captures current state but not evolution history |
| 5 | Revisit reasons behind updates | **Factual** | 40-50% | **No** | Requires recalling specific causal events — CDT doesn't store reasons |
| 6 | Preference-aligned recommendations | **Behavioral** | 30-50% | **Yes** | CDT grounding directly applicable — "user prefers X in context Y" |
| 7 | Generalize to new scenarios | **Behavioral** | 30-40% | **Yes** | CDT's gate structure maps preferences across contexts — core strength |

**CDT sweet spot: types 2, 6, 7** (behavioral, 30-40% LLM baseline — most room for improvement)
**CDT weakness: types 1, 4, 5** (factual recall — CDT abstracts away the specific facts)
**Hybrid needed: types 3, 4** (temporal — CDT captures patterns but RAG retrieves specific timeline)

### Recommended Hybrid Architecture

```
Query arrives
    ├─ Is it factual recall? (types 1, 5)
    │   └─ RAG retrieval from conversation history
    ├─ Is it behavioral/preference? (types 2, 6, 7)
    │   └─ CDT traversal for grounding
    └─ Is it temporal? (types 3, 4)
        └─ T-CDT temporal weighting + RAG for specific events
```

### Key Insight

CDT is a *compression* of behavioral patterns — it tells you "this user prefers X when Y" but NOT "this user mentioned Z on March 15th." PersonaMem tests both. The hybrid approach uses CDT for pattern-based questions and RAG for fact-based questions, routing by query type.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CDT too shallow with sparse data | High | Use lower min_cluster_size, d2 instead of d4 |
| Statements too generic for MCQ | Medium | Falsifiability constraints already in prompt |
| Domain mismatch (RP → preferences) | Medium | Test with 1 persona first, iterate |
| MCQ options too similar | Low | Use embedding similarity to find close calls |
| Temporal preference shift | High | T-CDT temporal weighting (future work) |
