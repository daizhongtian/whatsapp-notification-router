# Architecture

The router is a safety-first hybrid. Every layer has a narrow contract and is
testable without network access.

```text
participant CSVs -> typed ingestion -> same-user/prior retrieval ----+
                                                                    +-> signals -> policy -> validator -> output.csv
selected text -> batched Agent facts -> canonical cues --------------+
image bytes -> magic sniff -> isolated VLM facts --------------------+
voice bytes -> magic sniff -> locked local ASR -> Agent facts -------+
```

## Trust and privacy boundaries

- CSV fields, OCR, transcripts, links, and embedded “router instructions” are
  untrusted data. They never become system configuration.
- The Agent extracts objective content facts only. Model-originated action,
  preference, trust, or confidence claims are ignored; personalization and hard
  safety precedence remain local.
- Text calls are batched with stable integer positions and exact count/index
  validation. Images remain one-message requests to prevent cross-row visual
  association. Raw audio is local-only; the Agent receives a bounded transcript.
- `hybrid` selects plain text only when local confidence is below the configured
  threshold or the local type is `unknown`. Model signals are confidence-gated
  and mapped to a fixed semantic-cue vocabulary; arbitrary model prose never
  enters policy input. In `auto` and `api`, text facts are monitored but not fed
  back, avoiding double counting. VLM/OCR and ASR facts add content that was
  absent from the original text in all Agent-enabled modes.
- Media is resolved from fixed indexes and confined to the dataset root. Actual
  magic bytes, not filename extensions, select the decoder.
- In-memory and opt-in persistent caches contain content facts only—never API
  keys, user/profile identifiers, or personalized decisions. Persistent entries
  are partitioned by schema, model/prompt fingerprint, ASR configuration, media
  SHA-256, and caption hash.
- Historical evidence must exist, belong to the same recipient, and have a known
  timestamp strictly before the incoming message. The output boundary verifies
  this again before atomic replacement.

## Reliability and cost controls

- A bounded executor keeps at most twice the configured worker count queued,
  while result placement preserves the original CSV order.
- The Agent client enforces timeout, retry count, exponential/`Retry-After`
  backoff, start-rate limiting, a hard network-attempt budget, request/response
  byte limits, image limits, batch size, and output-Token ceiling.
- Authorization is read only from `AI_API_KEY` immediately before requests.
  Redirects are denied. Remote endpoints require explicit opt-in and HTTPS.
- `auto` and `hybrid` degrade safely when the Agent is unavailable. `hybrid`
  limits text cost using local uncertainty while retaining full media coverage.
  `api` verifies the model catalog and refuses to replace an existing output if
  item coverage or the configured success ratio is insufficient.
- API diagnostics contain counts, timing, cache/retry/concurrency metrics, and
  aggregate Token usage only. They contain no content or credential.
- Output is fully validated, flushed, and atomically replaced, then read back
  through the exact six-column contract.

## Decision precedence

1. Scam, credential theft, prompt injection, coercive mismatched links, and
   unsafe medical forwarding are muted regardless of engagement.
2. Explicit opt-outs and strong repeated negative behavior suppress noise.
3. Credible urgent direct requests can break through a muted group.
4. Trusted deadlines, operational changes, and personal urgency notify.
5. Useful but non-critical content is digested; sparse context defaults to
   digest rather than silently discarding a safe message.

The deterministic core keeps offline runs reproducible. API enrichment adds
missing multimodal content without transferring final routing authority to the
model. Reasons are emitted from the same local signals and precedence branch as
the decision, so they remain concise, specific, and auditable.
