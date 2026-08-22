# Message Notification Router

This package routes each incoming WhatsApp message to `notify`, `digest`, or
`mute`. It combines deterministic text and metadata signals, personalized history
retrieval, safety rules, and optional media enrichment. The default `offline` mode
uses only participant-visible files and the Python standard library.

See [ARCHITECTURE.md](ARCHITECTURE.md) for trust boundaries, decision precedence,
and the multimodal data flow. The implementation is MIT licensed.

## Requirements and setup

- Python 3.11 or newer (Python 3.12 is the tested optional-media runtime)
- The challenge `dataset/` directory, including its `media/` subtree
- Optional: `faster-whisper` for local voice-note transcription
- Optional: Pillow plus `pillow-avif-plugin` for AVIF-to-PNG conversion
- Optional: an OpenAI-compatible Agent Gateway for API media enrichment

From the repository root:

```powershell
python -m venv code\.venv
code\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\code
```

On macOS or Linux, activate with `source code/.venv/bin/activate`. The core
installation has no third-party runtime dependencies. For a fully pinned local
transcription/VLM environment:

```text
python -m pip install -r code/requirements-media.lock
```

The shorter editable-install alternatives are `python -m pip install -e
"./code[media]"` and `python -m pip install -e "./code[vision]"`. The lock file is
preferred for reproducibility. Native packages such as NumPy must match the
Python ABI; do not reuse a CPython 3.12 target directory from Python 3.14.

`requirements.txt` is intentionally dependency-free for the core runtime.

## Run

The explicit command is:

```text
python code/main.py route --dataset dataset --output dataset/output.csv --mode offline
```

`route` may be omitted for convenience:

```text
python code/main.py --dataset dataset --output dataset/output.csv --mode offline
```

Installed environments may use `message-router` instead of `python code/main.py`.
Paths may be absolute or relative to the current directory.

Routing modes:

- `offline` is deterministic, makes no network requests, and needs no secret.
- `auto` analyzes every item when the Agent API is configured and otherwise
  retains the safe offline behavior.
- `hybrid` first runs the deterministic policy, sends only low-confidence or
  `unknown` plain-text items to the Agent, and still analyzes every image and
  voice note. Model text is reduced to a confidence-gated canonical cue
  vocabulary before it can influence the local policy. If the Agent is not
  configured, the mode degrades to the same safe local behavior as `offline`.
- `api` requires the Agent API, validates the selected model, analyzes every
  incoming item, and refuses to replace an existing output when coverage or the
  configured success-ratio gate is not met.

When text is selected for Agent analysis, items are batched to amortize fixed
context cost. Images remain isolated VLM requests. Voice notes are transcribed
locally and the bounded transcript is sent for objective fact extraction; raw
audio is not sent. In `auto` and `api`, plain-text Agent results are monitored
but are not appended to the already-lossless source text, avoiding duplicate
urgency/promotion signals. Only `hybrid` feeds recognized canonical semantic
cues back into policy evaluation.

The process exits nonzero on invalid input, routing failure, or an output-contract
violation. The final CSV is validated first and then replaced atomically, so a
failed run does not leave a partial submission.

To save a content-free operational report (request counts, retries, aggregate
Token usage, coverage, cache hits, concurrency, and elapsed time), add
`--diagnostics-json run-diagnostics.json`.

## Python API

The same public pipeline is available in process:

```python
from pathlib import Path

from router.pipeline import run_pipeline

predictions = run_pipeline(
    dataset_dir=Path("dataset"),
    output_path=Path("dataset/output.csv"),
    mode="offline",
)
```

`run_pipeline(...)` returns the prediction objects after writing the requested
output. The lower-level output boundary is also reusable:

```python
from router.output import validate_output_file, validate_predictions, write_output
```

`write_output(predictions, messages, message_history, output_path)` checks every
invariant and writes rows in incoming-message order. `validate_output_file(...)`
additionally requires the CSV header to have the exact official column order.

## Output contract

The output header is exactly:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

The validator enforces:

- exactly one row for every incoming `message_id`, with no extras or duplicates;
- `action` and `message_type` values from the challenge enums;
- finite confidence in the inclusive range `0` to `1`;
- nonempty human-readable reasons;
- `none` or canonical semicolon-separated evidence IDs (no commas, spaces,
  empty segments, or duplicates); and
- evidence that exists in `message_history.csv`, belongs to the same receiving
  user, and was created strictly before the incoming message.

## Configuration and secrets

Offline mode requires no environment variables. `code/.env.example` documents the
optional settings:

| Variable | Purpose |
| --- | --- |
| `AI_API_KEY` | Sole credential source; leave unset for offline use |
| `AI_API_BASE_URL` | API base URL; default `http://127.0.0.1:4310/v1` |
| `AI_API_MODEL` | Model ID, verified through `/v1/models` |
| `AI_API_ALLOW_REMOTE` | Default `0`; remote APIs require `1` and HTTPS |
| `AI_API_TIMEOUT_SECONDS` | Per-attempt timeout, default 60 seconds |
| `AI_API_MAX_RETRIES` | Retry count for transient failures, default 2 |
| `AI_API_RETRY_BACKOFF_SECONDS` | Bounded exponential-backoff base |
| `AI_API_CONCURRENCY` | Bounded worker count, default 4 |
| `AI_API_BATCH_SIZE` | Plain-text items per request, default 8 |
| `AI_API_REQUESTS_PER_SECOND` | Request-start rate, default 4 per second |
| `AI_API_MAX_NETWORK_REQUESTS` | Hard run-level request-attempt budget |
| `AI_API_MAX_OUTPUT_TOKENS` | Per-response output ceiling |
| `AI_API_MIN_SUCCESS_RATIO` | API-mode quality gate, default 0.95 |
| `AI_API_HYBRID_CONFIDENCE_THRESHOLD` | In `hybrid`, select text below this local confidence; default 0.68 |
| `AI_API_CACHE_TTL_SECONDS` | In-memory content-fact cache TTL |
| `ROUTER_CACHE_DIR` | Opt-in persistent media-fact cache directory |
| `ROUTER_WHISPER_MODEL` | Local faster-whisper model name or path |
| `ROUTER_WHISPER_DOWNLOAD_ROOT` | Optional model storage directory |
| `ROUTER_ASR_DEVICE` | ASR device, normally `cpu` |
| `ROUTER_ASR_COMPUTE_TYPE` | ASR compute type, normally `int8` |
| `ROUTER_ENABLE_LOCAL_ASR` | Set `0` to disable local transcription |
| `ROUTER_WHISPER_LOCAL_FILES_ONLY` | Default `1`; prevents implicit model downloads |

Set values in the process environment or through your own secret manager. The
stdlib application does not load `.env` files. `AI_API_KEY` is read immediately
before each request and is never placed in configuration, cache keys, diagnostics,
or exceptions. Redirects are rejected so authorization cannot cross origins.
Never put a credential in source, CSV files, prompts, logs, shell history, or the
submission archive.

Persistent caching is disabled unless `ROUTER_CACHE_DIR` is set. Cache documents
contain bounded OCR/transcript content and hashes, but no key, message/user ID,
profile, or personalized decision. Treat that directory as private user data and
delete it when no longer needed.

## Leakage-safe evaluation

Run the solved examples in offline mode from the repository root:

```text
python code/evaluation/main.py --dataset dataset --labels dataset/sample_messages.csv --mode offline
```

Optionally persist the JSON report:

```text
python code/evaluation/main.py --dataset dataset --labels dataset/sample_messages.csv --mode offline --json-output evaluation-report.json
```

The evaluator does not point the router at the labeled file. It creates a temporary
dataset, writes only the 11 official input columns to `messages.csv`, allowlists
context/media files, omits `sample_messages.csv`, and invokes the public CLI in a
separate subprocess. Labels remain in evaluator memory and are joined by
`message_id` only after prediction. Offline evaluation also removes credential-like
environment variables from the subprocess.

The JSON report includes action accuracy, message-type accuracy, joint accuracy,
both confusion matrices, validated evidence-reference rates, evidence exact/Jaccard
and micro overlap scores, and confidence Brier score/ECE bins. Confidence is
calibrated against joint action-and-type correctness.

For a read-only audit of all 13 CSVs, referential integrity, exact output
coverage, every indexed media file, magic bytes, and optional deep image/audio
decode, run:

```text
python code/evaluation/audit_dataset.py --dataset dataset --json-output dataset-audit.json
```

On the 30 provided labeled examples, the final multimodal configuration passes
all 30 action/type pairs. This is a regression result for the public examples,
not a claim about hidden-set performance.

## Tests

The focused suite uses only the standard library and workspace-safe fixtures:

```text
python -m unittest discover -s code/tests -p "test_*.py" -v
```

The tests cover exact columns/order, atomic replacement, ID coverage, enums,
confidence bounds, historical evidence constraints, semicolon syntax, CLI
isolation, label stripping, metric joins, evidence overlap, calibration, prompt
injection, path confinement, MIME sniffing, batch parsing, concurrency, timeouts,
retry behavior, request budgets, cache isolation, and failure fallback.

## Submission package

Before packaging, run the tests and create `dataset/output.csv` with the desired
mode. Then use the deterministic packager, which allowlists source files, rejects
credential-like values, excludes caches/environments/artifacts, and reads the ZIP
back before reporting success:

```text
python code/package.py --output submission/code.zip
python -m zipfile -l submission/code.zip
```

Submit `code.zip` and `dataset/output.csv` as separate artifacts. Also provide the
required chat transcript through the challenge submission flow. Inspect the archive
listing before upload and confirm it contains this README, `main.py`, `router/`,
`evaluation/`, `prompts/`, `pyproject.toml`, `requirements.txt`, both
`requirements-*.lock` files, and `.env.example`.
