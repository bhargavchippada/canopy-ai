# PersonaMem Integration Feasibility Analysis

> Research output for canopy-ai CDT pipeline integration.
> Date: 2026-03-28

## 1. Dataset Overview

**PersonaMem** (COLM 2025) benchmarks LLMs on dynamic user profiling and personalized response generation across multi-session conversations.

| Property | Value |
|----------|-------|
| Paper | [arXiv 2504.14225](https://arxiv.org/abs/2504.14225) |
| Repo | [bowen-upenn/PersonaMem](https://github.com/bowen-upenn/PersonaMem) |
| HuggingFace | [bowen-upenn/PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem) |
| License | MIT |
| Personas | 20 (persona_id 0-19), each with expanded demographic profiles |
| Topics | 18 conversation domains (see below) |
| Context sizes | 32k (589 rows), 128k (2,730 rows), 1M (2,670 rows) |
| Total QA pairs | ~5,990 |
| Generation model | GPT-4o (synthetic conversations) |

### 1.1 Topics (18 domains)

bookRecommendation, coding, datingConsultation, email, familyRelations, financialConsultation, foodRecommendation, homeDecoration, legalConsultation, medicalConsultation, movieRecommendation, musicRecommendation, onlineShopping, sportsRecommendation, studyConsultation, therapy, travelPlanning, writing.

### 1.2 Query Types (7 categories)

| Type | Share | Description |
|------|-------|-------------|
| Acknowledge latest user preferences | 30.1% | Recognize current preference state |
| Suggest new ideas | 22.9% | Novel suggestions aligned with persona |
| Provide preference-aligned recommendations | 11.6% | Recommendations matching known prefs |
| Track full preference evolution | 11.0% | How preferences changed over time |
| Revisit reasons behind preference updates | 9.3% | Why user changed a preference |
| Generalize to new scenarios | 9.4% | Apply preferences in new contexts |
| Recall user-shared facts | 5.8% | Remember facts user mentioned |

### 1.3 Data Format

**Questions CSV** (`questions_[SIZE].csv`): Each row is a QA evaluation pair with `persona_id`, `question_id`, `question_type`, `topic`, `user_question_or_message`, `correct_answer` (a/b/c/d), `all_options` (JSON array of 4 MCQ choices), `shared_context_id`, and `end_index_in_shared_context`.

**Shared Contexts JSONL** (`shared_contexts_[SIZE].jsonl`): Each line is a JSON object keyed by `shared_context_id`, containing the full conversation history as an OpenAI-format message list (`[{"role": "user"|"assistant", "content": "..."}]`). The `end_index_in_shared_context` field slices this list to get the context visible at query time.

**Raw conversation structure** (pre-benchmark): Conversations are generated as lists of strings with `User:`, `Assistant:` (or `Patient:`, `Therapist:` for therapy), and `Side_Note:` prefixes. Side notes carry timestamps (MM/DD/YYYY) and annotate which personal history event each exchange references. These are stripped during benchmark assembly — the final shared_contexts contain only user/assistant turns.

### 1.4 PersonaMem-v2 (ImplicitPersona)

A larger follow-up dataset with 195 personas, 51.7k rows, multimodal support, and richer schema (expanded_persona, preference types, conversation scenarios, sensitivity flags). More complex but the same conceptual structure. Integration would follow the same pattern as v1.

## 2. Data Format Mapping to BehavioralObservation

### 2.1 BehavioralObservation Schema (canopy-ai)

```python
@dataclass(frozen=True)
class BehavioralObservation:
    scene: str        # Context (preceding actions, environment)
    action: str       # The observed behavior/response
    actor: str        # Who performed the action
    participants: tuple[str, ...]  # Other actors present
    metadata: dict[str, Any]       # Domain-specific metadata
```

### 2.2 Mapping Strategy

PersonaMem conversations are multi-turn user-assistant dialogues. Each user message + preceding context constitutes an observation of user behavior.

| BehavioralObservation field | PersonaMem source |
|----------------------------|-------------------|
| `scene` | Previous N assistant+user turns (sliding window context) |
| `action` | The user's message in that turn |
| `actor` | `"user_{persona_id}"` (e.g., `"user_0"`) |
| `participants` | `("assistant",)` — single chatbot interlocutor |
| `metadata.topic` | Topic from the conversation block (e.g., `"musicRecommendation"`) |
| `metadata.persona_id` | Integer persona identifier |
| `metadata.timestamp` | Extracted from Side_Note if using raw data; absent in benchmark format |
| `metadata.session_index` | Position in the multi-session sequence |
| `metadata.question_type` | From QA CSV (for evaluation observations only) |

### 2.3 Extraction Approaches

**Approach A — From shared_contexts (benchmark format):**
Parse the OpenAI-format message list. For each user message at index `i`, construct:
- `scene` = concatenation of messages `[max(0, i-k) : i]` (k = context window, e.g. 4-6 turns)
- `action` = `messages[i]["content"]`
- `actor` = `"user_{persona_id}"` (looked up from questions CSV via shared_context_id)

Pros: Clean, ready-to-use, no raw data needed.
Cons: No timestamps, no Side_Note annotations, no preference labels.

**Approach B — From raw generated data (data/output/):**
Parse the original conversation lists with Side_Note annotations intact. Each Side_Note carries a timestamp and links to a specific personal history event (like/dislike, with optional change reasons).

Pros: Rich temporal annotations, preference change markers, event provenance.
Cons: Requires running the generation pipeline or obtaining pre-generated raw data (not included in the HuggingFace release — only the assembled benchmark is distributed).

**Recommendation:** Start with Approach A (benchmark shared_contexts). It provides sufficient conversational data for CDT construction. Approach B adds temporal grounding but requires regenerating data locally.

### 2.4 Key Difference from CDT Paper's RP Domain

The CDT paper (arxiv 2601.10080) profiles **fictional characters** from roleplay scenes. PersonaMem profiles **user preferences** from chatbot conversations. This shifts the CDT's purpose:

| Dimension | CDT Paper (RP) | PersonaMem (User Profiling) |
|-----------|---------------|----------------------------|
| Actor | Fictional character (Kasumi, Arisa) | Synthetic user persona |
| Observation | RP scene + character action | Conversation turn + user message |
| What's profiled | Character personality traits | User preferences, habits, evolving tastes |
| Temporal dynamics | Static (character doesn't change) | Dynamic (preferences evolve over weeks/months/years) |
| Relationship CDTs | Character-to-character | User-to-topic or user-to-chatbot |
| Ground truth | Scene-action consistency | MCQ evaluation (correct personalized response) |

This is a natural fit for canopy-ai's domain-agnostic design. BehavioralObservation already supports arbitrary metadata. The temporal evolution in PersonaMem is exactly what T-CDT (Phase 5) is designed to capture.

## 3. Proposed CDT Topics for User Preferences

The CDT paper uses 4 attribute topics (identity, personality, habits, goals) and relationship CDTs per character pair. For user profiling, the topics should reflect preference dimensions:

### 3.1 Attribute Topics (per user)

| Topic | What it captures | Example hypotheses |
|-------|------------------|--------------------|
| `taste_preferences` | Likes, dislikes, aesthetic preferences | "User prefers jazz fusion over classical" |
| `lifestyle_habits` | Daily routines, consumption patterns | "User cooks at home on weekdays" |
| `decision_style` | How user makes choices (analytical, impulsive, value-driven) | "User prioritizes reviews over price" |
| `knowledge_expertise` | What user knows about, expertise areas | "User has deep knowledge of wine regions" |
| `emotional_needs` | What user seeks from interactions (validation, exploration, comfort) | "User wants reassurance before trying new things" |
| `values_priorities` | Core values driving preferences | "User values sustainability over convenience" |

### 3.2 Relationship Topics (per user x topic-domain)

Instead of character-to-character relationships, build user-to-domain CDTs:

| Relationship | What it captures |
|-------------|-----------------|
| `user x musicRecommendation` | Music-specific preferences, listening habits |
| `user x foodRecommendation` | Food preferences, dietary restrictions, exploration vs. comfort |
| `user x therapy` | Communication needs, emotional patterns |
| `user x travelPlanning` | Travel style, budget sensitivity, adventure level |

This maps directly to canopy-ai's existing `build_character_profile()` where `other_characters` becomes `topic_domains`, and `rel_topic2cdt` captures domain-specific preference trees.

### 3.3 Temporal CDT Opportunity

PersonaMem explicitly models preference evolution across 4 time periods (init, next_week, next_month, next_year). This is the ideal testbed for T-CDT:
- Build CDT at time T0 (init conversations)
- Update CDT at T1 (next_week) — some preferences have changed
- Track which hypotheses are invalidated by new evidence
- Measure whether updated CDT produces better personalized responses

## 4. Integration Complexity Estimate

### 4.1 Data Loader (SIMPLE — 1-2 files)

Write a `PersonaMemLoader` that:
1. Downloads the HuggingFace dataset (or reads local CSVs/JSONL)
2. Parses shared_contexts into `list[BehavioralObservation]` per persona
3. Groups observations by topic and time period

Estimated effort: 1 file (~150 lines), plus tests.

### 4.2 Topic Mapping (SIMPLE — config only)

Define the 6 attribute topics and topic-domain relationships in CDTConfig or a separate config. No code changes to canopy core — the topics are just string labels passed to `build_cdt()`.

### 4.3 Evaluation Harness (MEDIUM — 3-5 files)

Use PersonaMem's MCQ evaluation:
1. Build CDT from conversation history up to `end_index_in_shared_context`
2. Use CDT to generate/select the personalized response
3. Compare against `correct_answer`
4. Measure accuracy by question_type, topic, context_length

This requires:
- A traversal function that uses CDT hypotheses to score/rank response options
- An evaluation script that runs across all QA pairs
- Result aggregation by question_type and topic

Estimated effort: 3 files (~400 lines), plus tests.

### 4.4 Prompt Adaptation (SIMPLE — modify existing)

The `summarize_triggers` prompt in canopy needs to generate preference-oriented hypotheses rather than character-trait hypotheses. This is a prompt-level change, not a code change. May need a `domain="user_profiling"` config option to select the appropriate prompt variant.

### 4.5 Total Estimate

| Component | Complexity | Files | Lines |
|-----------|-----------|-------|-------|
| Data loader | SIMPLE | 1 + test | ~150 |
| Topic config | SIMPLE | config | ~30 |
| Evaluation harness | MEDIUM | 3 + tests | ~400 |
| Prompt variant | SIMPLE | 1 edit | ~50 |
| **Total** | **MEDIUM** | **~5 new + 1 edit** | **~630** |

## 5. Open Questions

1. **Raw data availability.** The HuggingFace release contains only the assembled benchmark (shared_contexts + QA pairs). The raw conversation data with Side_Note annotations and personal history JSON is not distributed. Do we need to run the generation pipeline ourselves to get temporal annotations? Or is the benchmark format sufficient for initial CDT construction?

2. **Observation granularity.** Each user message is one observation. But some messages are short ("yes", "sounds good") while others contain rich preference signals. Should we filter or weight observations by information density? Or let the CDT's NLI validation handle this naturally (sparse messages produce hypotheses that fail validation)?

3. **CDT topic selection.** The 6 proposed attribute topics are speculative. Should we use the PersonaMem topic domains (musicRecommendation, etc.) directly as CDT topics instead? Or use a hybrid: attribute topics for cross-domain profiling + domain topics for domain-specific preferences?

4. **Evaluation methodology.** PersonaMem evaluates via MCQ (pick the best response from 4 options). CDT traversal produces a character description, not a response selector. The integration point is: given a CDT profile, can an LLM use it as context to select the correct MCQ answer? This adds an LLM call to the evaluation loop. Is this the right evaluation design, or should we measure CDT quality independently (e.g., preference prediction accuracy)?

5. **Baseline comparison.** PersonaMem's leaderboard shows GPT-4.5 at ~52% on MCQ. If CDT-augmented responses score higher, that demonstrates value. But the CDT itself is built from the same conversation history the LLM already sees. The hypothesis: CDTs compress the preference signal, making it more accessible than raw conversation history. Is this a fair comparison?

6. **v1 vs v2.** PersonaMem-v2 (ImplicitPersona) has 195 personas and richer annotations but is more complex. Should we start with v1 (20 personas, simpler format) for proof-of-concept, then scale to v2?

7. **Participant modeling.** In PersonaMem, the "assistant" is a generic chatbot — there's no meaningful variation in assistant identity. Relationship CDTs would be `user x domain` rather than `user x other_character`. Does the relationship CDT mechanism still add value, or should we focus on attribute CDTs only?

## 6. Recommendation

**Start with PersonaMem v1 (32k context split, 589 QA pairs, 20 personas).** This is small enough for rapid iteration and covers all 7 query types and 18 topics. The 32k context fits comfortably in modern LLMs for evaluation.

**Phase 1:** Build data loader + CDT construction for 2-3 personas across all topics. Validate that CDTs capture meaningful preference structure.

**Phase 2:** Build evaluation harness using the MCQ format. Compare CDT-augmented LLM responses against vanilla LLM baseline.

**Phase 3:** Add temporal dynamics (T-CDT) using the preference evolution signal. Measure whether CDT updates improve accuracy on "track preference evolution" and "revisit reasons" query types.

This positions canopy-ai as the first CDT-based approach to dynamic user profiling — a novel contribution beyond the RP domain the CDT paper covers.
