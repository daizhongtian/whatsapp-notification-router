"""Validation and atomic CSV output for message-routing predictions.

This module is deliberately dependency-free.  It is the final trust boundary before
predictions become a submission: every row is checked against the incoming message
set and every evidence reference is checked against historical data.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias


OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

ALLOWED_ACTIONS = frozenset({"notify", "digest", "mute"})
ALLOWED_MESSAGE_TYPES = frozenset(
    {
        "personal",
        "urgent",
        "event",
        "payment",
        "business_update",
        "promotion",
        "greeting",
        "forward",
        "spam",
        "scam",
        "unknown",
    }
)
MAX_OUTPUT_FILE_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_ROWS = 2_000_000
MAX_OUTPUT_FIELD_CHARACTERS = 250_000

Row: TypeAlias = Mapping[str, Any]
RowSource: TypeAlias = str | os.PathLike[str] | Iterable[Row]


class OutputValidationError(ValueError):
    """Raised when predictions cannot form a contract-compliant output file."""


def _csv_rows(path: Path, *, exact_columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OutputValidationError(f"CSV is not a regular file: {path}")
        if path.stat().st_size > MAX_OUTPUT_FILE_BYTES:
            raise OutputValidationError(f"CSV exceeds the byte limit: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise OutputValidationError(f"CSV has no header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise OutputValidationError(f"CSV has duplicate headers: {path}")
            if exact_columns is not None and tuple(reader.fieldnames) != tuple(exact_columns):
                raise OutputValidationError(
                    f"CSV columns must be exactly {list(exact_columns)!r}; "
                    f"got {reader.fieldnames!r} in {path}"
                )
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if row_number > MAX_OUTPUT_ROWS + 1:
                    raise OutputValidationError(f"CSV exceeds the row limit: {path}")
                if None in row:
                    raise OutputValidationError(
                        f"CSV row {row_number} has unexpected extra fields: {path}"
                    )
                rows.append(dict(row))
            return rows
    except OSError as exc:
        raise OutputValidationError(f"Could not read CSV {path}: {exc}") from exc


def _as_mapping(value: Any) -> Row:
    if isinstance(value, Mapping):
        return value
    for method_name in ("to_row", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, Mapping):
                return converted
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Domain objects may carry audit-only fields that do not belong in output.csv.
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    fields = {
        name: getattr(value, name)
        for name in OUTPUT_COLUMNS
        if hasattr(value, name)
    }
    if fields:
        return fields
    raise OutputValidationError(
        "Prediction rows must be mappings, dataclass instances, or objects with output fields"
    )


def _load_rows(source: RowSource, *, exact_columns: Sequence[str] | None = None) -> list[Row]:
    if isinstance(source, (str, os.PathLike)):
        return _csv_rows(Path(source), exact_columns=exact_columns)
    try:
        return [_as_mapping(row) for row in source]
    except TypeError as exc:
        raise OutputValidationError("Expected a CSV path or an iterable of rows") from exc


def _text(value: Any, field: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        raise OutputValidationError(f"{field} cannot be null")
    result = str(value)
    if not result or result != result.strip():
        raise OutputValidationError(f"{field} must be non-empty and have no surrounding whitespace")
    if "\x00" in result:
        raise OutputValidationError(f"{field} cannot contain a NUL character")
    if len(result) > MAX_OUTPUT_FIELD_CHARACTERS:
        raise OutputValidationError(
            f"{field} exceeds {MAX_OUTPUT_FIELD_CHARACTERS} characters"
        )
    return result


def _confidence(value: Any, message_id: str) -> str:
    if isinstance(value, bool):
        raise OutputValidationError(f"confidence for {message_id} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OutputValidationError(f"confidence for {message_id} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise OutputValidationError(
            f"confidence for {message_id} must be finite and between 0 and 1; got {value!r}"
        )
    # Ten significant digits are ample for a confidence while avoiding binary-float noise.
    return format(number, ".10g")


def parse_evidence_ids(value: Any) -> tuple[str, ...]:
    """Parse the evidence field and enforce its canonical semicolon syntax.

    ``none`` represents an empty evidence set.  Otherwise IDs must be separated by a
    single semicolon, with no spaces, commas, empty segments, or duplicates.  A
    sequence is accepted from in-process predictors and serialized canonically.
    """

    if value is None:
        return ()
    if isinstance(value, str):
        if value == "none":
            return ()
        if not value or value != value.strip():
            raise OutputValidationError(
                "evidence_message_ids must be 'none' or a semicolon-separated ID list without whitespace"
            )
        if "," in value:
            raise OutputValidationError(
                "evidence_message_ids must use semicolons, never commas"
            )
        values = value.split(";")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
        if not values:
            return ()
    else:
        raise OutputValidationError(
            "evidence_message_ids must be 'none', a semicolon-separated string, or an ID sequence"
        )

    ids: list[str] = []
    for raw_id in values:
        evidence_id = _text(raw_id, "evidence message ID")
        if evidence_id == "none":
            raise OutputValidationError("'none' cannot be combined with evidence message IDs")
        if ";" in evidence_id or "," in evidence_id or any(ch.isspace() for ch in evidence_id):
            raise OutputValidationError(
                "evidence IDs cannot contain separators or whitespace"
            )
        ids.append(evidence_id)
    if len(ids) != len(set(ids)):
        raise OutputValidationError("evidence_message_ids cannot contain duplicate IDs")
    return tuple(ids)


def serialize_evidence_ids(value: Any) -> str:
    """Return the canonical CSV representation of an evidence value."""

    evidence_ids = parse_evidence_ids(value)
    return ";".join(evidence_ids) if evidence_ids else "none"


def _timestamp(value: Any, *, context: str) -> datetime:
    text = _text(value, context)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise OutputValidationError(f"Invalid ISO timestamp for {context}: {value!r}") from exc


def _strictly_before(historical: datetime, incoming: datetime, *, context: str) -> bool:
    historical_aware = historical.utcoffset() is not None
    incoming_aware = incoming.utcoffset() is not None
    if historical_aware != incoming_aware:
        raise OutputValidationError(
            f"Cannot compare timezone-aware and timezone-naive timestamps for {context}"
        )
    if historical_aware:
        historical = historical.astimezone(timezone.utc)
        incoming = incoming.astimezone(timezone.utc)
    return historical < incoming


def _index_rows(rows: Iterable[Row], *, source_name: str) -> tuple[list[str], dict[str, Row]]:
    order: list[str] = []
    index: dict[str, Row] = {}
    for number, row in enumerate(rows, start=2):
        if "message_id" not in row:
            raise OutputValidationError(f"{source_name} row {number} has no message_id")
        message_id = _text(row["message_id"], f"{source_name} message_id")
        if message_id in index:
            raise OutputValidationError(f"Duplicate message_id {message_id!r} in {source_name}")
        order.append(message_id)
        index[message_id] = row
    return order, index


def validate_predictions(
    predictions: RowSource,
    messages: RowSource,
    message_history: RowSource,
) -> list[dict[str, str]]:
    """Validate predictions and return canonical rows in incoming-message order.

    Validation covers the exact output fields, allowed enums, finite confidence,
    one prediction per input ID, and historical evidence ownership/chronology.
    """

    message_rows = _load_rows(messages)
    history_rows = _load_rows(message_history)
    prediction_rows = _load_rows(predictions)

    message_order, message_index = _index_rows(message_rows, source_name="messages")
    _, history_index = _index_rows(history_rows, source_name="message_history")

    prediction_index: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(prediction_rows, start=2):
        if set(row) != set(OUTPUT_COLUMNS):
            missing = sorted(set(OUTPUT_COLUMNS) - set(row))
            extra = sorted(set(row) - set(OUTPUT_COLUMNS))
            raise OutputValidationError(
                f"Prediction row {row_number} must contain exactly {list(OUTPUT_COLUMNS)!r}; "
                f"missing={missing!r}, extra={extra!r}"
            )

        message_id = _text(row["message_id"], "prediction message_id")
        if message_id in prediction_index:
            raise OutputValidationError(f"Duplicate prediction for message_id {message_id!r}")

        action = _text(row["action"], f"action for {message_id}")
        if action not in ALLOWED_ACTIONS:
            raise OutputValidationError(
                f"Invalid action for {message_id}: {action!r}; allowed={sorted(ALLOWED_ACTIONS)!r}"
            )
        message_type = _text(row["message_type"], f"message_type for {message_id}")
        if message_type not in ALLOWED_MESSAGE_TYPES:
            raise OutputValidationError(
                f"Invalid message_type for {message_id}: {message_type!r}; "
                f"allowed={sorted(ALLOWED_MESSAGE_TYPES)!r}"
            )
        reason = _text(row["reason"], f"reason for {message_id}")
        evidence_ids = parse_evidence_ids(row["evidence_message_ids"])

        prediction_index[message_id] = {
            "message_id": message_id,
            "action": action,
            "message_type": message_type,
            "reason": reason,
            "confidence": _confidence(row["confidence"], message_id),
            "evidence_message_ids": ";".join(evidence_ids) if evidence_ids else "none",
        }

    missing_predictions = [mid for mid in message_order if mid not in prediction_index]
    unexpected_predictions = sorted(set(prediction_index) - set(message_index))
    if missing_predictions or unexpected_predictions:
        raise OutputValidationError(
            "Prediction IDs must match messages.csv exactly; "
            f"missing={missing_predictions!r}, unexpected={unexpected_predictions!r}"
        )

    for message_id in message_order:
        incoming = message_index[message_id]
        incoming_user = _text(incoming.get("user_id"), f"user_id for incoming {message_id}")
        incoming_time: datetime | None = None
        evidence_ids = parse_evidence_ids(
            prediction_index[message_id]["evidence_message_ids"]
        )
        for evidence_id in evidence_ids:
            historical = history_index.get(evidence_id)
            if historical is None:
                raise OutputValidationError(
                    f"Evidence {evidence_id!r} for {message_id} is not in message_history.csv"
                )
            historical_user = _text(
                historical.get("user_id"), f"user_id for historical {evidence_id}"
            )
            if historical_user != incoming_user:
                raise OutputValidationError(
                    f"Evidence {evidence_id!r} for {message_id} belongs to user "
                    f"{historical_user!r}, not {incoming_user!r}"
                )
            if incoming_time is None:
                incoming_time = _timestamp(
                    incoming.get("created_at"), context=f"incoming {message_id} created_at"
                )
            historical_time = _timestamp(
                historical.get("created_at"), context=f"historical {evidence_id} created_at"
            )
            if not _strictly_before(
                historical_time,
                incoming_time,
                context=f"evidence {evidence_id} for {message_id}",
            ):
                raise OutputValidationError(
                    f"Evidence {evidence_id!r} for {message_id} is not strictly historical"
                )

    return [prediction_index[message_id] for message_id in message_order]


def validate_output_file(
    output_path: str | os.PathLike[str],
    messages: RowSource,
    message_history: RowSource,
) -> list[dict[str, str]]:
    """Validate an existing output CSV, including exact header order."""

    rows = _csv_rows(Path(output_path), exact_columns=OUTPUT_COLUMNS)
    return validate_predictions(rows, messages, message_history)


def write_output(
    predictions: RowSource,
    messages: RowSource,
    message_history: RowSource,
    output_path: str | os.PathLike[str],
) -> Path:
    """Atomically write a fully validated ``output.csv``.

    The temporary file is created beside the destination so ``os.replace`` remains
    atomic on a single filesystem.  Existing output is untouched if validation or
    serialization fails.
    """

    canonical_rows = validate_predictions(predictions, messages, message_history)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS), extrasaction="raise")
            writer.writeheader()
            writer.writerows(canonical_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return destination


__all__ = [
    "ALLOWED_ACTIONS",
    "ALLOWED_MESSAGE_TYPES",
    "OUTPUT_COLUMNS",
    "OutputValidationError",
    "parse_evidence_ids",
    "serialize_evidence_ids",
    "validate_output_file",
    "validate_predictions",
    "write_output",
]
