# LLM Adapter Tradeoffs

## Adapter Comparison

| Adapter | Auth | Cost | Latency | Rate Limits | Batch Support |
|---------|------|------|---------|-------------|---------------|
| **ClaudeCodeAdapter** (claude-agent-sdk) | Max subscription | Included in sub | ~3-5s/call (subprocess) | Max sub limits | No |
| **AnthropicAPIAdapter** (anthropic SDK) | API key | Per-token | ~1-2s/call | Tier-based | Yes (batches API) |
| **OpenAIAdapter** (openai SDK) | API key | Per-token | ~1-2s/call | Tier-based | Yes |
| **SubprocessAdapter** (claude CLI) | Max subscription | Included | ~3-5s/call | Max sub limits | No |

## Current Choice: ClaudeCodeAdapter

**Why:** We have Claude Max subscription. No separate API key needed. The claude-agent-sdk wraps the Claude Code CLI, routing through the Max subscription.

**Tradeoffs:**
- (+) No API costs beyond subscription
- (+) Uses same auth as Claude Code terminal
- (-) Higher latency per call (subprocess spawn)
- (-) No batch processing
- (-) No fine-grained temperature/token control

**Acceptable for Phase 0-1:** CDT construction makes ~30-40 LLM calls per character. At ~4s each, that's ~2-3 minutes of LLM time per character. Tolerable for reproduction.

## Future: AnthropicAPIAdapter

For production or batch processing, switch to the `anthropic` SDK directly:
- Lower latency
- Batch API support (50% cost reduction)
- Fine-grained parameters (temperature, max_tokens, stop sequences)
- Requires separate API key

## Model IDs

| Provider | Model | ID |
|----------|-------|----|
| Claude Code SDK | Sonnet 4.6 | `claude-sonnet-4-6` |
| Claude Code SDK | Opus 4.6 | `claude-opus-4-6` |
| Claude Code SDK | Haiku 4.5 | `claude-haiku-4-5` |
| Anthropic API | Sonnet 4.6 | `claude-sonnet-4-6-20250514` |
| OpenAI | GPT-4.1 | `gpt-4.1` |
