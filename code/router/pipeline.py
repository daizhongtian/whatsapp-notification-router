"""End-to-end orchestration for loading, enriching, routing, and validating.

The model-facing stage extracts objective content facts only. Personalized
notification decisions remain in the auditable policy layer, so the same poster
can be routed differently for two users and unsafe content can never promote
itself by embedding instructions for the router.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .data import load_dataset
from .gateway import AgentGatewayClient, GatewayConfig, GatewayError
from .media import MediaFacts, MediaResolver
from .models import MediaType, MessageType, Prediction
from .output import validate_output_file, write_output
from .policy import RouterPolicy


RoutingMode = Literal["offline", "auto", "hybrid", "api"]


class PipelineError(RuntimeError):
    """Raised when the public routing pipeline cannot complete safely."""


@dataclass(frozen=True)
class PipelineRun:
    """Predictions plus content-free operational diagnostics."""

    predictions: tuple[Prediction, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _Enrichment:
    content: str = ""
    available: bool = False
    media_kind: str = "none"
    source: str = "none"
    error: str = ""
    cache_hit: bool = False
    attempted: bool = False
    selected_for_agent: bool = False
    is_media_message: bool = False


_HYBRID_CUE_RULES: tuple[tuple[str, tuple[str, ...], float], ...] = (
    (
        "explicit deadline or near-term time constraint",
        ("deadline", "time-sensitive", "time sensitive", "expires", "expiry"),
        0.60,
    ),
    (
        "urgent or immediate action request",
        ("urgent", "immediate", "emergency", "act now"),
        0.60,
    ),
    (
        "schedule, appointment, pickup, event, or access change",
        ("schedule", "rescheduled", "appointment", "pickup", "event date", "access change"),
        0.60,
    ),
    (
        "payment, invoice, fee, transfer, or refund request",
        ("payment", "invoice", "fee", "transfer", "refund"),
        0.60,
    ),
    (
        "request to share an OTP, verification code, password, PIN, or account number",
        (
            "credential",
            "otp",
            "one-time",
            "verification code",
            "password",
            "pin request",
            "account number",
        ),
        0.80,
    ),
    (
        "suspicious, shortened, or mismatched link or domain",
        ("suspicious link", "shortened link", "short url", "domain mismatch", "mismatched domain"),
        0.80,
    ),
    (
        "promotion, sale, discount, or marketing offer",
        ("promotion", "promotional", "sale", "discount", "marketing offer"),
        0.60,
    ),
    (
        "request to forward this chain message to everyone",
        ("forward", "chain message", "share with everyone"),
        0.80,
    ),
    (
        "impersonated account-security verification request",
        ("impersonation", "impersonates"),
        0.85,
    ),
    (
        "unsafe advice to stop medication",
        ("unsafe advice", "unsafe medical"),
        0.85,
    ),
    (
        "explicit threat or coercion",
        ("threat", "coercion", "coercive"),
        0.80,
    ),
    (
        "explicitly non-urgent wording",
        ("non-urgent", "non urgent", "not urgent", "no rush", "whenever convenient"),
        0.60,
    ),
)

_HYBRID_NONURGENT_MARKERS = (
    "non-urgent",
    "non urgent",
    "not urgent",
    "no rush",
    "whenever convenient",
)


def _hybrid_semantic_cues(facts: dict[str, Any]) -> str:
    """Map model facts to a small canonical vocabulary before policy use.

    Arbitrary model text is never appended to the source message.  This keeps
    prompt-injected or hallucinated prose out of the deterministic policy while
    still allowing uncertain multilingual text to gain objective semantic cues.
    """

    if facts.get("available") is not True:
        return ""
    raw_confidence = facts.get("confidence", 0.0)
    if (
        isinstance(raw_confidence, bool)
        or not isinstance(raw_confidence, (int, float))
        or float(raw_confidence) < 0.60
    ):
        return ""
    raw_signals = facts.get("signals", [])
    if not isinstance(raw_signals, list):
        return ""
    signal_text = " ".join(
        str(value)[:240].casefold()
        for value in raw_signals[:24]
        if isinstance(value, str)
    )
    confidence = float(raw_confidence)
    explicitly_nonurgent = any(
        marker in signal_text for marker in _HYBRID_NONURGENT_MARKERS
    )
    canonical = []
    for cue, markers, minimum_confidence in _HYBRID_CUE_RULES:
        if confidence < minimum_confidence:
            continue
        if cue == "urgent or immediate action request" and explicitly_nonurgent:
            continue
        if any(marker in signal_text for marker in markers):
            canonical.append(cue)
    if not canonical:
        return ""
    return "Objective semantic cues: " + "; ".join(canonical)


def _message_media_input(message: object) -> dict[str, object]:
    """Return only bounded content fields accepted by the media resolver."""

    values = asdict(message)  # MessageRecord is a frozen dataclass.
    media_type = values.get("media_type")
    return {
        "media_id": values.get("media_id") or "",
        "media_type": getattr(media_type, "value", media_type) or "",
        "message_text": str(values.get("message_text") or "")[:16_000],
    }


def _gateway_for_mode(mode: RoutingMode) -> AgentGatewayClient | None:
    if mode == "offline":
        return None
    try:
        client = AgentGatewayClient(GatewayConfig.from_env())
    except (GatewayError, ValueError) as exc:
        if mode == "api":
            raise PipelineError(f"invalid Agent API configuration: {exc}") from exc
        return None
    if not client.available():
        if mode == "api":
            raise PipelineError("api mode requires AI_API_KEY in the process environment")
        return None
    try:
        models = client.list_models(refresh=True)
        model_ids = {str(model.get("id", "")) for model in models}
        if client.config.model not in model_ids:
            if mode == "api":
                raise PipelineError(
                    "configured Agent API model is not present in the model catalog"
                )
            return None
    except GatewayError as exc:
        if mode == "api":
            raise PipelineError(f"Agent API is unavailable: {exc}") from exc
        return None
    return client


def _enrich_text_batch(
    messages: tuple[Any, ...],
    client: AgentGatewayClient,
    *,
    include_semantic_cues: bool = False,
) -> tuple[_Enrichment, ...]:
    facts_values = client.extract_content_facts_batch(
        [
            {
                "media_kind": "text",
                "message_text": str(message.message_text or "")[:16_000],
            }
            for message in messages
        ]
    )
    results: list[_Enrichment] = []
    for facts in facts_values:
        available = facts.get("available") is True
        results.append(
            _Enrichment(
                # Plain text is already lossless input to the deterministic
                # policy. Feeding its model summary back would count urgency and
                # promotion terms twice. The Agent result is still validated,
                # monitored, cached, and quality-gated; content overrides are
                # reserved for OCR/VLM and ASR facts that add missing modalities.
                content=_hybrid_semantic_cues(facts) if include_semantic_cues else "",
                available=available,
                media_kind="none",
                source="gateway" if available else "none",
                error=str(facts.get("error", ""))[:100] if not available else "",
                attempted=True,
                selected_for_agent=True,
            )
        )
    return tuple(results)


def _enrich_media(
    message: Any,
    resolver: MediaResolver,
    client: AgentGatewayClient | None,
) -> _Enrichment:
    facts: MediaFacts = resolver.analyze_message(
        _message_media_input(message), client=client
    )
    return _Enrichment(
        content=facts.as_text() if facts.available else "",
        available=facts.available,
        media_kind=facts.media_kind,
        source=facts.source,
        error=facts.error,
        cache_hit=facts.cache_hit,
        attempted=True,
        selected_for_agent=client is not None,
        is_media_message=True,
    )


def _content_enrichment(
    dataset_dir: Path,
    data: Any,
    *,
    mode: RoutingMode,
    client: AgentGatewayClient | None,
    cache_dir: str | os.PathLike[str] | None,
) -> tuple[dict[str, str], tuple[_Enrichment, ...]]:
    resolver = MediaResolver(dataset_dir, cache_dir=cache_dir)

    if client is not None:
        enrichments: list[_Enrichment | None] = [None] * len(data.messages)
        selected_text_ids: set[str] | None = None
        if mode == "hybrid":
            baseline = RouterPolicy(data).route_all()
            selected_text_ids = {
                prediction.message_id
                for message, prediction in zip(data.messages, baseline, strict=True)
                if message.media_type is MediaType.NONE
                and (
                    prediction.confidence < client.config.hybrid_confidence_threshold
                    or prediction.message_type is MessageType.UNKNOWN
                )
            }

        def job_descriptors():
            text_indices: list[int] = []
            for index, message in enumerate(data.messages):
                if message.media_type is MediaType.NONE:
                    if selected_text_ids is not None and message.message_id not in selected_text_ids:
                        enrichments[index] = _Enrichment(source="local")
                        continue
                    text_indices.append(index)
                    if len(text_indices) >= client.config.batch_size:
                        indices = tuple(text_indices)
                        yield ("text", indices, tuple(data.messages[i] for i in indices))
                        text_indices.clear()
                else:
                    yield ("media", (index,), message)
            if text_indices:
                indices = tuple(text_indices)
                yield ("text", indices, tuple(data.messages[i] for i in indices))

        def store_result(kind: str, indices: tuple[int, ...], value: Any) -> None:
            if kind == "text":
                for index, item in zip(indices, value, strict=True):
                    enrichments[index] = item
            else:
                enrichments[indices[0]] = value

        with ThreadPoolExecutor(
            max_workers=client.config.concurrency,
            thread_name_prefix="router-content",
        ) as executor:
            pending: dict[Any, tuple[str, tuple[int, ...]]] = {}
            max_pending = client.config.concurrency * 2
            for kind, indices, value in job_descriptors():
                if kind == "text":
                    future = executor.submit(
                        _enrich_text_batch,
                        value,
                        client,
                        include_semantic_cues=mode == "hybrid",
                    )
                else:
                    future = executor.submit(_enrich_media, value, resolver, client)
                pending[future] = (kind, indices)
                if len(pending) >= max_pending:
                    completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for item in completed:
                        completed_kind, completed_indices = pending.pop(item)
                        store_result(completed_kind, completed_indices, item.result())
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for item in completed:
                    completed_kind, completed_indices = pending.pop(item)
                    store_result(completed_kind, completed_indices, item.result())
        final_enrichments = tuple(
            item if item is not None else _Enrichment(error="analysis_missing")
            for item in enrichments
        )
    else:
        final_enrichments = tuple(
            _Enrichment()
            if message.media_type is MediaType.NONE
            else _enrich_media(message, resolver, None)
            for message in data.messages
        )

    content = {
        message.message_id: enrichment.content
        for message, enrichment in zip(data.messages, final_enrichments, strict=True)
        if enrichment.content
    }
    return content, final_enrichments


def _build_report(
    *,
    mode: RoutingMode,
    elapsed_seconds: float,
    messages: int,
    enrichments: tuple[_Enrichment, ...],
    client: AgentGatewayClient | None,
) -> dict[str, Any]:
    failures = Counter(item.error for item in enrichments if item.error)
    media = tuple(item for item in enrichments if item.is_media_message)
    attempted = tuple(item for item in enrichments if item.attempted)
    selected_for_agent = tuple(item for item in enrichments if item.selected_for_agent)
    selected_text = tuple(
        item for item in selected_for_agent if not item.is_media_message
    )
    selected_media = tuple(
        item for item in selected_for_agent if item.is_media_message
    )
    gateway_metrics = client.metrics_snapshot() if client is not None else {}
    api_calls = int(gateway_metrics.get("content_calls", 0))
    api_successes = int(gateway_metrics.get("content_successes", 0))
    reusable_media_hits = sum(item.cache_hit for item in media)
    covered_items = api_calls + reusable_media_hits
    successful_items = api_successes + reusable_media_hits
    return {
        "mode": mode,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "messages": messages,
        "content": {
            "attempted": len(attempted),
            "available": sum(item.available for item in attempted),
            "unavailable": sum(not item.available for item in attempted),
            "skipped": messages - len(attempted),
            "failure_codes": dict(sorted(failures.items())),
        },
        "media": {
            "messages": len(media),
            "available": sum(item.available for item in media),
            "cache_hits": sum(item.cache_hit for item in media),
            "sources": dict(sorted(Counter(item.source for item in media).items())),
        },
        "api": {
            "enabled": client is not None,
            "model": client.config.model if client is not None else "",
            "configured_concurrency": client.config.concurrency if client is not None else 0,
            "configured_requests_per_second": (
                client.config.requests_per_second if client is not None else 0.0
            ),
            "selection_strategy": (
                "selective" if mode == "hybrid" and client is not None
                else "all" if client is not None
                else "none"
            ),
            "selected_items": len(selected_for_agent),
            "selected_text_messages": len(selected_text),
            "selected_media_messages": len(selected_media),
            "skipped_text_messages": (
                messages - len(media) - len(selected_text)
            ),
            "covered_items": covered_items,
            "successful_items": successful_items,
            "content_success_ratio": (
                round(successful_items / covered_items, 6) if covered_items else 0.0
            ),
            "metrics": gateway_metrics,
        },
    }


def run_pipeline_with_report(
    dataset_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    mode: RoutingMode = "auto",
    cache_dir: str | os.PathLike[str] | None = None,
) -> PipelineRun:
    """Run the complete router and return sanitized operational diagnostics."""

    if mode not in {"offline", "auto", "hybrid", "api"}:
        raise PipelineError(f"unsupported routing mode: {mode!r}")

    started = time.perf_counter()
    root = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    data = load_dataset(root)
    client = _gateway_for_mode(mode)

    content_by_id, enrichments = _content_enrichment(
        root, data, mode=mode, client=client, cache_dir=cache_dir
    )
    report = _build_report(
        mode=mode,
        elapsed_seconds=time.perf_counter() - started,
        messages=len(data.messages),
        enrichments=enrichments,
        client=client,
    )

    if mode == "api":
        api = report["api"]
        covered_items = int(api.get("covered_items", 0))
        if covered_items != len(data.messages):
            raise PipelineError(
                "api mode could not submit every incoming message for Agent analysis"
            )
        ratio = float(api["content_success_ratio"])
        assert client is not None
        if ratio < client.config.min_success_ratio:
            raise PipelineError(
                "Agent API success ratio is below AI_API_MIN_SUCCESS_RATIO "
                f"({ratio:.3f} < {client.config.min_success_ratio:.3f})"
            )

    predictions = tuple(
        RouterPolicy(data).route_all(content_by_message_id=content_by_id)
    )
    write_output(predictions, data.messages, data.message_history, destination)
    validate_output_file(destination, root / "messages.csv", root / "message_history.csv")

    final_report = dict(report)
    final_report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return PipelineRun(predictions=predictions, report=final_report)


def run_pipeline(
    dataset_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    mode: RoutingMode = "auto",
    cache_dir: str | os.PathLike[str] | None = None,
) -> list[Prediction]:
    """Compatibility wrapper returning only the ordered predictions."""

    return list(
        run_pipeline_with_report(
            dataset_dir, output_path, mode=mode, cache_dir=cache_dir
        ).predictions
    )


__all__ = [
    "PipelineError",
    "PipelineRun",
    "RoutingMode",
    "run_pipeline",
    "run_pipeline_with_report",
]
