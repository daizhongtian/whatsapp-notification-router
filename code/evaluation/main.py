"""Leakage-safe evaluation harness for the public message-router CLI.

The labeled CSV never becomes visible to the routing process.  Evaluation writes a
temporary dataset containing only the official 11 input columns in ``messages.csv``,
copies participant-visible context/media, invokes ``code/main.py`` in a subprocess,
and joins predictions with labels only after the subprocess exits.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from router.output import (  # noqa: E402 - code root is added for direct-script use
    ALLOWED_ACTIONS,
    ALLOWED_MESSAGE_TYPES,
    OUTPUT_COLUMNS,
    OutputValidationError,
    parse_evidence_ids,
    validate_output_file,
)


INPUT_COLUMNS = (
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
)

LABEL_COLUMNS = (
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

CONTEXT_FILES = (
    "users.csv",
    "groups.csv",
    "group_members.csv",
    "business_accounts.csv",
    "user_business_history.csv",
    "message_history.csv",
    "message_events.csv",
    "images.csv",
    "voice_notes.csv",
    "daily_notification_summary.csv",
)
MAX_EVALUATION_MEDIA_FILES = 10_000
MAX_EVALUATION_MEDIA_BYTES = 2_000_000_000


class EvaluationError(RuntimeError):
    """Raised when the isolated evaluation cannot be completed safely."""


@contextlib.contextmanager
def _evaluation_workspace() -> Iterable[Path]:
    """Create a writable evaluation workspace without Windows 3.14 ACL traps.

    ``tempfile.TemporaryDirectory`` creates an owner-only ACL on recent Windows
    Python builds that becomes unusable in some managed runners and OneDrive
    workspaces.  A UUID-named directory plus guaranteed recursive cleanup keeps
    the same isolation without that platform-specific failure.
    """

    configured = os.environ.get("ROUTER_TEMP_DIR", "").strip()
    parent = (
        Path(configured).expanduser().resolve()
        if configured
        else (REPOSITORY_ROOT / ".test-tmp").resolve()
    )
    parent.mkdir(parents=True, exist_ok=True)
    workspace = parent / f"message-router-eval-{uuid.uuid4().hex}"
    workspace.mkdir()
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@dataclass(frozen=True)
class EvaluationReport:
    """Machine-readable evaluation results."""

    rows: int
    accuracy: dict[str, float]
    confusion: dict[str, dict[str, dict[str, int]]]
    evidence: dict[str, float | int]
    calibration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "accuracy": self.accuracy,
            "confusion": self.confusion,
            "evidence": self.evidence,
            "calibration": self.calibration,
        }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EvaluationError(f"CSV has no header: {path}")
            return list(reader.fieldnames), [dict(row) for row in reader]
    except OSError as exc:
        raise EvaluationError(f"Could not read {path}: {exc}") from exc


def read_labeled_rows(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    """Read and minimally validate a labeled evaluation CSV."""

    csv_path = Path(path)
    fieldnames, rows = _read_csv(csv_path)
    required = set(INPUT_COLUMNS) | set(LABEL_COLUMNS)
    missing = sorted(required - set(fieldnames))
    if missing:
        raise EvaluationError(f"Labeled CSV is missing columns {missing!r}: {csv_path}")
    if not rows:
        raise EvaluationError(f"Labeled CSV contains no examples: {csv_path}")

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        message_id = row.get("message_id", "")
        if not message_id:
            raise EvaluationError(f"Labeled CSV row {row_number} has no message_id")
        if message_id in seen:
            raise EvaluationError(f"Duplicate labeled message_id {message_id!r}")
        seen.add(message_id)
        if row.get("action") not in ALLOWED_ACTIONS:
            raise EvaluationError(
                f"Invalid gold action {row.get('action')!r} for {message_id}"
            )
        if row.get("message_type") not in ALLOWED_MESSAGE_TYPES:
            raise EvaluationError(
                f"Invalid gold message_type {row.get('message_type')!r} for {message_id}"
            )
    return rows


def _copy_media_tree_safely(source: Path, target: Path) -> None:
    """Copy participant media without following links outside the dataset."""

    if source.is_symlink():
        raise EvaluationError("dataset/media must not be a symbolic link")
    source_root = source.resolve(strict=True)
    target.mkdir(parents=True, exist_ok=False)
    file_count = 0
    total_bytes = 0
    for item in sorted(source.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise EvaluationError("dataset/media contains a symbolic link")
        try:
            resolved = item.resolve(strict=True)
            relative = resolved.relative_to(source_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise EvaluationError("dataset/media contains an unsafe path") from exc
        destination = target / relative
        if resolved.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if not resolved.is_file():
            raise EvaluationError("dataset/media contains a non-regular file")
        file_count += 1
        total_bytes += resolved.stat().st_size
        if (
            file_count > MAX_EVALUATION_MEDIA_FILES
            or total_bytes > MAX_EVALUATION_MEDIA_BYTES
        ):
            raise EvaluationError("dataset/media exceeds the evaluation copy limits")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, destination)


def build_sanitized_dataset(
    source_dataset: str | os.PathLike[str],
    labeled_rows: Sequence[Mapping[str, Any]],
    destination: str | os.PathLike[str],
) -> Path:
    """Build the only dataset visible to the evaluated routing subprocess.

    The function uses an allowlist.  It never copies ``sample_messages.csv``, an
    existing ``messages.csv``, or ``output.csv`` from the source dataset.
    """

    source = Path(source_dataset).resolve()
    target = Path(destination).resolve()
    if not source.is_dir():
        raise EvaluationError(f"Dataset directory does not exist: {source}")
    if source == target:
        raise EvaluationError("Sanitized evaluation dataset must differ from the source dataset")

    target.mkdir(parents=True, exist_ok=False)
    for filename in CONTEXT_FILES:
        source_file = source / filename
        if source_file.is_file():
            shutil.copy2(source_file, target / filename)

    if not (target / "message_history.csv").is_file():
        raise EvaluationError("Evaluation requires dataset/message_history.csv")

    source_media = source / "media"
    if source_media.is_dir():
        _copy_media_tree_safely(source_media, target / "media")

    messages_path = target / "messages.csv"
    try:
        with messages_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(INPUT_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row_number, row in enumerate(labeled_rows, start=2):
                missing = [column for column in INPUT_COLUMNS if column not in row]
                if missing:
                    raise EvaluationError(
                        f"Labeled row {row_number} is missing input columns {missing!r}"
                    )
                writer.writerow({column: row[column] for column in INPUT_COLUMNS})
    except OSError as exc:
        raise EvaluationError(f"Could not create sanitized messages.csv: {exc}") from exc

    return target


def _subprocess_environment(mode: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["ROUTER_EVALUATION"] = "1"
    if mode == "offline":
        # Keep accidental credentials outside the offline evaluation boundary.
        for name in tuple(environment):
            upper_name = name.upper()
            if (
                upper_name.endswith("_API_KEY")
                or upper_name.endswith("_TOKEN")
                or upper_name.endswith("_SECRET")
            ):
                environment.pop(name, None)
    return environment


def run_public_pipeline(
    main_script: str | os.PathLike[str],
    dataset_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    mode: str = "offline",
    timeout_seconds: float = 600.0,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess[str]:
    """Invoke the documented public CLI across the leakage boundary."""

    if mode not in {"offline", "auto", "hybrid", "api"}:
        raise EvaluationError(f"Unsupported routing mode: {mode!r}")
    script = Path(main_script).resolve()
    if not script.is_file():
        raise EvaluationError(f"Router CLI does not exist: {script}")

    command = [
        python_executable,
        str(script),
        "route",
        "--dataset",
        str(Path(dataset_dir).resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--mode",
        mode,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=script.parent,
            env=_subprocess_environment(mode),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluationError(f"Could not run router CLI: {exc}") from exc
    if completed.returncode != 0:
        raise EvaluationError(
            "Router CLI failed with exit code "
            f"{completed.returncode}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not Path(output_path).is_file():
        raise EvaluationError("Router CLI completed without creating its requested output CSV")
    return completed


def _confusion_matrix(
    gold: Iterable[str],
    predicted: Iterable[str],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for expected, actual in zip(gold, predicted, strict=True):
        row = matrix.setdefault(expected, {})
        row[actual] = row.get(actual, 0) + 1
    return {key: dict(sorted(value.items())) for key, value in sorted(matrix.items())}


def _divide(numerator: float, denominator: float, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _calibration(confidences: Sequence[float], outcomes: Sequence[int]) -> dict[str, Any]:
    count = len(confidences)
    if not count:
        return {
            "target": "joint_action_and_message_type_correct",
            "mean_confidence": 0.0,
            "brier_score": 0.0,
            "expected_calibration_error": 0.0,
            "bins": [],
        }

    bins: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == 9 and confidence == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(confidences[index] for index in members) / len(members)
        empirical_accuracy = sum(outcomes[index] for index in members) / len(members)
        gap = abs(mean_confidence - empirical_accuracy)
        weighted_gap += len(members) * gap
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_gap": gap,
            }
        )

    return {
        "target": "joint_action_and_message_type_correct",
        "mean_confidence": sum(confidences) / count,
        "brier_score": sum(
            (confidence - outcome) ** 2
            for confidence, outcome in zip(confidences, outcomes, strict=True)
        )
        / count,
        "expected_calibration_error": weighted_gap / count,
        "bins": bins,
    }


def score_predictions(
    labeled_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> EvaluationReport:
    """Join labels after prediction and calculate routing-quality metrics."""

    gold_by_id = {str(row["message_id"]): row for row in labeled_rows}
    prediction_by_id = {str(row["message_id"]): row for row in prediction_rows}
    if set(gold_by_id) != set(prediction_by_id):
        raise EvaluationError(
            "Prediction IDs do not match labeled IDs; "
            f"missing={sorted(set(gold_by_id) - set(prediction_by_id))!r}, "
            f"unexpected={sorted(set(prediction_by_id) - set(gold_by_id))!r}"
        )

    ordered_ids = [str(row["message_id"]) for row in labeled_rows]
    gold_actions = [str(gold_by_id[mid]["action"]) for mid in ordered_ids]
    predicted_actions = [str(prediction_by_id[mid]["action"]) for mid in ordered_ids]
    gold_types = [str(gold_by_id[mid]["message_type"]) for mid in ordered_ids]
    predicted_types = [str(prediction_by_id[mid]["message_type"]) for mid in ordered_ids]
    action_outcomes = [
        int(expected == actual)
        for expected, actual in zip(gold_actions, predicted_actions, strict=True)
    ]
    type_outcomes = [
        int(expected == actual)
        for expected, actual in zip(gold_types, predicted_types, strict=True)
    ]
    joint_outcomes = [
        action_ok * type_ok
        for action_ok, type_ok in zip(action_outcomes, type_outcomes, strict=True)
    ]

    evidence_exact = 0
    evidence_jaccard = 0.0
    overlap_count = 0
    predicted_reference_count = 0
    gold_reference_count = 0
    for message_id in ordered_ids:
        predicted_evidence = set(
            parse_evidence_ids(prediction_by_id[message_id]["evidence_message_ids"])
        )
        gold_evidence = set(parse_evidence_ids(gold_by_id[message_id]["evidence_message_ids"]))
        intersection = len(predicted_evidence & gold_evidence)
        union = len(predicted_evidence | gold_evidence)
        evidence_exact += int(predicted_evidence == gold_evidence)
        evidence_jaccard += _divide(intersection, union, empty=1.0)
        overlap_count += intersection
        predicted_reference_count += len(predicted_evidence)
        gold_reference_count += len(gold_evidence)

    evidence_precision = _divide(
        overlap_count,
        predicted_reference_count,
        empty=1.0 if gold_reference_count == 0 else 0.0,
    )
    evidence_recall = _divide(
        overlap_count,
        gold_reference_count,
        empty=1.0 if predicted_reference_count == 0 else 0.0,
    )
    evidence_f1 = _divide(
        2 * evidence_precision * evidence_recall,
        evidence_precision + evidence_recall,
        empty=0.0,
    )

    confidences = [float(prediction_by_id[mid]["confidence"]) for mid in ordered_ids]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidences):
        raise EvaluationError("Predicted confidence must be finite and between 0 and 1")

    row_count = len(ordered_ids)
    return EvaluationReport(
        rows=row_count,
        accuracy={
            "action": _divide(sum(action_outcomes), row_count),
            "message_type": _divide(sum(type_outcomes), row_count),
            "joint": _divide(sum(joint_outcomes), row_count),
        },
        confusion={
            "action": _confusion_matrix(gold_actions, predicted_actions),
            "message_type": _confusion_matrix(gold_types, predicted_types),
        },
        evidence={
            # The output validator verifies existence, ownership, chronology, and syntax.
            "valid_row_rate": 1.0,
            "valid_reference_rate": 1.0,
            "predicted_reference_count": predicted_reference_count,
            "gold_reference_count": gold_reference_count,
            "overlap_reference_count": overlap_count,
            "exact_match_rate": _divide(evidence_exact, row_count),
            "macro_jaccard": _divide(evidence_jaccard, row_count),
            "micro_precision": evidence_precision,
            "micro_recall": evidence_recall,
            "micro_f1": evidence_f1,
        },
        calibration=_calibration(confidences, joint_outcomes),
    )


def evaluate(
    dataset_dir: str | os.PathLike[str],
    labeled_csv: str | os.PathLike[str],
    *,
    main_script: str | os.PathLike[str] = CODE_ROOT / "main.py",
    mode: str = "offline",
    timeout_seconds: float = 600.0,
    python_executable: str = sys.executable,
) -> EvaluationReport:
    """Run an end-to-end leakage-safe evaluation using the public CLI."""

    # Labels are held only in this process.  They are not written to the subprocess dataset.
    labeled_rows = read_labeled_rows(labeled_csv)
    with _evaluation_workspace() as temporary_root:
        safe_dataset = build_sanitized_dataset(
            dataset_dir,
            labeled_rows,
            temporary_root / "dataset",
        )
        output_path = temporary_root / "predictions.csv"
        run_public_pipeline(
            main_script,
            safe_dataset,
            output_path,
            mode=mode,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        try:
            predictions = validate_output_file(
                output_path,
                safe_dataset / "messages.csv",
                safe_dataset / "message_history.csv",
            )
        except OutputValidationError as exc:
            raise EvaluationError(f"Router produced invalid output: {exc}") from exc

    # The label/prediction join happens only after routing has completed.
    return score_predictions(labeled_rows, predictions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the router without exposing labeled columns to it."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "dataset",
        help="Source participant dataset directory (default: repository dataset/)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPOSITORY_ROOT / "dataset" / "sample_messages.csv",
        help="Labeled CSV used only by the evaluator",
    )
    parser.add_argument(
        "--main",
        dest="main_script",
        type=Path,
        default=CODE_ROOT / "main.py",
        help="Public router CLI script",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "auto", "hybrid", "api"),
        default="offline",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="CLI timeout in seconds")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Also save the JSON report to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate(
            args.dataset,
            args.labels,
            main_script=args.main_script,
            mode=args.mode,
            timeout_seconds=args.timeout,
        )
    except (EvaluationError, OutputValidationError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
