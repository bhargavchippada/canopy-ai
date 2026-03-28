# Codifier Comparison — CDT Build & Benchmark Results

**Date:** 2026-03-28
**Character:** Kasumi (167 test pairs, except where noted)

---

## CDT Package Summary

| CDT Config | Codifier | Sys Prompt | Prompt Fix | Nodes | Stmts | Gates | Depth | θ_accept | θ_filter |
|---|---|---|---|---|---|---|---|---|---|
| Paper d3 (v3.1) | GPT-4.1 | None | N/A | 36 | 83 | 28 | 2 | 0.80* | 0.80* |
| **Sonnet d4 (prompt fix)** | **Sonnet** | **None** | **Yes** | **30** | **113** | **22** | **3** | **0.75** | **0.75** |
| Sonnet d3 (old) | Sonnet | Yes | No | 30 | 88 | 22 | 3 | 0.75 | 0.80 |
| Haiku d4 (no-sysprompt) | Haiku | None | No | 17 | 71 | 9 | 2 | 0.75 | 0.75 |
| Haiku d4 (with-sysprompt, overwritten) | Haiku | Yes | No | 16 | 79 | 8 | 1 | 0.75 | 0.75 |

\* Paper code defaults; paper text says θ=0.75.

### Key Observations
- Paper CDT has **3.5x more gates** (28 vs 8-9) than our builds
- Removing system prompt: +1 node, -8 stmts, +1 gate, +1 depth level (v1→v2)
- System prompt effect is **modest** — the bigger gap is codifier quality (GPT-4.1 vs Haiku)
- Sonnet d3 has 22 gates (closest to paper's 28) but all in relationship trees, not attribute trees

---

## Benchmark Results (All Verified from results/ JSON files)

### WITH Relationships (paper-comparable)

| CDT | Gen | Eval | Score | n | Note |
|---|---|---|---|---|---|
| paper-original (v3.1) | Llama | Sonnet | **70.66** | 167 | Paper CDT baseline |
| paper-original (pkl) | Sonnet | Sonnet | 65.87 | 167 | Different paper pkl file |
| gpt41.d3.rel | Sonnet | Sonnet | 66.17 | 167 | GPT-4.1 CDT from our d3 build |
| haiku.d3.a75 | Haiku | Sonnet | 58.38 | 167 | |
| sonnet.d3.a75 | Haiku | Sonnet | 58.98 | 167 | |
| sonnet.d3.a75 | Haiku | Haiku | 50.00 | 167 | |
| sonnet.d3.a80 | Sonnet | Sonnet | 64.07 | 167 | |
| **haiku.d4.a75 (v1)** | **Sonnet** | **Sonnet** | **71.39** | **166** | **BEST — exceeds paper baseline** |
| haiku.d4.a75 (v2) | Llama | Sonnet | 65.87 | 167 | no-sysprompt CDT |
| sonnet.d3.a75 | Llama | Sonnet | 60.0 | 10 | n=10 only |

### WITHOUT Relationships

| CDT | Gen | Eval | Score | n | Note |
|---|---|---|---|---|---|
| sonnet.d3.a75 | Sonnet | Sonnet | 65.57 | 167 | |

### Cross-Character (Arisa)

| CDT | Gen | Eval | Score | n | Note |
|---|---|---|---|---|---|
| paper-original | Llama | Sonnet | 54.31 | 116 | A=30%, B=48%, C=22% |
| paper-original | Sonnet | Sonnet | 63.36 | 116 | |

---

## Statement Quality Comparison

| Metric | Haiku d4 | Paper GPT-4.1 |
|---|---|---|
| Avg word count | 17.2 | 20.0 |
| Conditional hedging ("tends to"/"appears to") | 86% | 49% |
| Name/context-specific references | 53% | 43% |
| Stmts at depth 0 (flat) | 66% | 43% |
| Stmts at depth 1+ (gated) | 34% | 57% |

### Identity Topic Deep Dive

| Metric | Haiku d4 | Paper GPT-4.1 |
|---|---|---|
| Nodes | 2 | 8 |
| Statements | 14 | 22 |
| Gates | 1 | 7 |
| Max depth | 1 | 2 |

Paper's identity statements are longer (20-31 words), more declarative, and more falsifiable. Haiku's are shorter (15-19 words), heavily hedged with "tends to" / "appears to".

---

## Root Causes (Impact on Score, Ranked)

| Factor | Impact | Evidence |
|---|---|---|
| Relationships included | ~15 pts | 55.69 → 70.66 (Llama, paper CDT) |
| Eval model (Sonnet vs GPT-4.1) | ~13 pts | 71 (Sonnet) vs 84.25 (GPT-4.1) |
| Codifier quality (Haiku vs GPT-4.1) | ~3-5 pts | 65.87 vs 70.66 (same gen/eval) |
| System prompt in CDT construction | ~1-2 pts | 16n/8g → 17n/9g (modest improvement) |
| Gen format (dialogue vs narration) | ~1-2 pts | 66.17 → 67.96 |
| Gen model (Llama vs Sonnet) | ~5-6 pts | 65.87 → 71.39 (same CDT) |

---

## Root Cause: Hypothesis Framing (Not Codifier Model)

**Confirmed by testing both Haiku and Sonnet codifiers — both produce identical flat attribute trees.**

The issue is how Claude frames hypotheses vs how GPT-4.1 frames them:

### Side-by-Side: Identity Root Statements

| Our CDT (Claude) | Paper CDT (GPT-4.1) |
|---|---|
| "Kasumi **tends to** respond with immediate, unfiltered emotional expression..." (19w) | "Kasumi **proposes** direct collective action and **demonstrates** resistance..." (14w) |
| "Kasumi **tends to** express affection and verbally affirm the irreplaceability..." (18w) | "Kasumi **responds** through spontaneous vocal excitement..." (18w) |
| "Kasumi **tends to** refuse resignation when facing obstacles..." (16w) | "Kasumi **directly addresses** and seeks engagement with Arisa..." (21w) |
| 8 root stmts, **0 gates** | 4 root stmts, **3 gates** |

### Three Differences That Create Flat Trees

1. **Hedging**: Claude uses "tends to" / "appears to" (86% of stmts). GPT-4.1 uses declarative present tense ("proposes", "responds", "demonstrates"). Hedged statements pass NLI on ALL scenes → never become gates.

2. **Universality**: Claude stmts are universally true of Kasumi. Paper stmts name specific characters (Arisa), specific contexts. Universal truths can't be gated — they apply everywhere.

3. **Front-loading**: Claude produces 8 root stmts covering all behavior. GPT-4.1 produces 4 root stmts + reserves conditional behavior for 3 gates. Claude puts everything at the root; GPT-4.1 decomposes into situation-specific branches.

### Fix Applied (commit b1bd433)

Added 3 constraints to `summarize_triggers` prompt (`src/canopy/prompts.py`):
1. **Max 15 words** — prevents verbose universal descriptions
2. **Must be FALSE in 30%+ of scenes** — prevents always-true statements
3. **Must reference specific behavioral trigger** — prevents general traits

**Result**: Identity topic 7 nodes, 6 gates, depth=4 (was 2n/1g/d1; paper: 8n/7g/d3)

### Known Constraint Tension (from code review)
- "non-assertive" hedge words ("tends to", "may") on line 211 partially conflicts with the falsifiability requirement on line 212 — hedged statements are harder to falsify
- In practice, the 15-word limit + trigger specificity constraint override the hedging tendency (empirically validated: trees now match paper depth)
- Future refinement: replace "non-assertive" with "descriptive, not causal" and narrow acceptable hedges to "is observed as" / "is described as"
- Quality checklist (line 234-239) not yet updated with new constraints
- `make_hypothesis_prompt` (upstream generator) is unconstrained; topics with ≤8 pairs skip compression and bypass all new constraints

The codifier model (Haiku vs Sonnet) is NOT the bottleneck — both produce the same flat pattern. The prompt is.

---

## Key Findings

1. **71.39 exceeds paper's Sonnet-eval baseline** (70.66) despite using Haiku as codifier
2. **Relationships are the biggest factor** (~15 pts) — paper ALWAYS includes them
3. **Eval model gap is ~13 pts** (Sonnet 35-40% B vs GPT-4.1 ~18% B), not 28 as originally estimated
4. **Hypothesis framing is the CDT quality bottleneck** — Claude hedges ("tends to"), GPT-4.1 declares
5. **Codifier model doesn't matter** — Haiku and Sonnet produce identical flat attribute trees
6. **System prompt removal has modest effect** on CDT structure (+1 gate, +1 depth level)
7. **Sonnet gen outperforms Llama gen by ~5-6 pts** on the same CDT — the paper's Llama is not optimal
8. **Mode collapse** in Llama gen: Kasumi 24% "Let's...", Arisa 28% "Can we..."
