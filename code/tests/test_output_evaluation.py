from __future__ import annotations

import csv
import contextlib
import os
import shutil
import sys
import textwrap
import unittest
import uuid
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.main import INPUT_COLUMNS, evaluate
from router.output import (
    OUTPUT_COLUMNS,
    OutputValidationError,
    validate_output_file,
    validate_predictions,
    write_output,
)
from router.models import Action, MessageType, Prediction


@contextlib.contextmanager
def workspace_tempdir():
    """Use a normal workspace directory on Windows/Python 3.14."""

    path = CODE_ROOT / (".output-eval-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _prediction(
    message_id: str,
    *,
    action: str = "notify",
    message_type: str = "urgent",
    confidence: object = 0.8,
    evidence: object = "none",
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "action": action,
        "message_type": message_type,
        "reason": "A concise routing explanation.",
        "confidence": confidence,
        "evidence_message_ids": evidence,
    }


class OutputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [
            {"message_id": "incoming_2", "user_id": "u2", "created_at": "2026-07-02 10:00"},
            {"message_id": "incoming_1", "user_id": "u1", "created_at": "2026-07-02 10:00"},
        ]
        self.history = [
            {"message_id": "history_u1", "user_id": "u1", "created_at": "2026-07-01 10:00"},
            {"message_id": "history_u2", "user_id": "u2", "created_at": "2026-07-01 10:00"},
            {"message_id": "future_u1", "user_id": "u1", "created_at": "2026-07-03 10:00"},
        ]

    def test_writer_uses_exact_columns_input_order_and_canonical_evidence(self) -> None:
        predictions = [
            _prediction("incoming_1", evidence=["history_u1"]),
            _prediction("incoming_2", action="digest", message_type="event", confidence="0.70"),
        ]
        with workspace_tempdir() as temporary:
            destination = Path(temporary) / "nested" / "output.csv"
            returned = write_output(predictions, self.messages, self.history, destination)
            self.assertEqual(returned, destination)

            with destination.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), OUTPUT_COLUMNS)
                rows = list(reader)

        self.assertEqual([row["message_id"] for row in rows], ["incoming_2", "incoming_1"])
        self.assertEqual(rows[0]["confidence"], "0.7")
        self.assertEqual(rows[0]["evidence_message_ids"], "none")
        self.assertEqual(rows[1]["evidence_message_ids"], "history_u1")

    def test_invalid_predictions_leave_existing_destination_untouched(self) -> None:
        invalid = [_prediction("incoming_1")]
        with workspace_tempdir() as temporary:
            destination = Path(temporary) / "output.csv"
            destination.write_text("existing submission\n", encoding="utf-8")
            with self.assertRaises(OutputValidationError):
                write_output(invalid, self.messages, self.history, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "existing submission\n")
            self.assertEqual(list(Path(temporary).glob(".output.csv.*.tmp")), [])

    def test_rejects_invalid_enum_confidence_and_evidence(self) -> None:
        valid = [
            _prediction("incoming_2", evidence="history_u2"),
            _prediction("incoming_1", evidence="history_u1"),
        ]
        mutations = (
            (0, "action", "later"),
            (0, "message_type", "advert"),
            (0, "confidence", float("nan")),
            (0, "confidence", 1.01),
            (0, "evidence_message_ids", "history_u2,history_u1"),
            (0, "evidence_message_ids", "history_u2; history_u1"),
            (0, "evidence_message_ids", "history_u1"),  # wrong user
            (1, "evidence_message_ids", "future_u1"),
        )
        for row_index, field, value in mutations:
            with self.subTest(field=field, value=value):
                candidate = [dict(row) for row in valid]
                candidate[row_index][field] = value
                with self.assertRaises(OutputValidationError):
                    validate_predictions(candidate, self.messages, self.history)

    def test_rejects_noncanonical_header_order(self) -> None:
        with workspace_tempdir() as temporary:
            output_path = Path(temporary) / "output.csv"
            swapped = list(OUTPUT_COLUMNS)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=swapped).writeheader()
            with self.assertRaises(OutputValidationError):
                validate_output_file(output_path, self.messages, self.history)

    def test_accepts_domain_prediction_with_audit_only_fields(self) -> None:
        predictions = [
            Prediction(
                message_id="incoming_2",
                action=Action.DIGEST,
                message_type=MessageType.EVENT,
                reason="Useful later.",
                confidence=0.7,
            ),
            Prediction(
                message_id="incoming_1",
                action=Action.NOTIFY,
                message_type=MessageType.URGENT,
                reason="Needs attention.",
                confidence=0.9,
                evidence_message_ids=("history_u1",),
            ),
        ]
        rows = validate_predictions(predictions, self.messages, self.history)
        self.assertEqual([row["message_id"] for row in rows], ["incoming_2", "incoming_1"])


class LeakageSafeEvaluationTests(unittest.TestCase):
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_evaluator_hides_labels_calls_cli_and_reports_metrics(self) -> None:
        with workspace_tempdir() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            self._write_csv(
                dataset / "message_history.csv",
                ["message_id", "user_id", "created_at"],
                [
                    {
                        "message_id": "hist_1",
                        "user_id": "user_1",
                        "created_at": "2026-07-01 08:00",
                    }
                ],
            )
            # This sentinel labeled file must never cross into the temporary dataset.
            (dataset / "sample_messages.csv").write_text(
                "organizer_only_label,DO_NOT_EXPOSE\n", encoding="utf-8"
            )
            (dataset / "media").mkdir()
            (dataset / "media" / "sentinel.txt").write_text("participant media", encoding="utf-8")

            labeled_rows: list[dict[str, object]] = []
            for message_id, user_id, action, message_type, evidence in (
                ("eval_1", "user_1", "notify", "urgent", "hist_1"),
                ("eval_2", "user_2", "digest", "promotion", "none"),
            ):
                row = {column: "" for column in INPUT_COLUMNS}
                row.update(
                    {
                        "message_id": message_id,
                        "user_id": user_id,
                        "conversation_type": "personal",
                        "sender_user_id": "sender",
                        "created_at": "2026-07-02 08:00",
                        "message_text": "routing input",
                        "forwarded_count": "0",
                        "action": action,
                        "message_type": message_type,
                        "reason": "gold label",
                        "confidence": "0.9",
                        "evidence_message_ids": evidence,
                    }
                )
                labeled_rows.append(row)
            labels = root / "labels.csv"
            self._write_csv(
                labels,
                list(INPUT_COLUMNS) + [
                    "action",
                    "message_type",
                    "reason",
                    "confidence",
                    "evidence_message_ids",
                ],
                labeled_rows,
            )

            stub_main = root / "stub_main.py"
            stub_main.write_text(
                textwrap.dedent(
                    f"""
                    import argparse
                    import csv
                    import os
                    from pathlib import Path

                    input_columns = {list(INPUT_COLUMNS)!r}
                    output_columns = {list(OUTPUT_COLUMNS)!r}

                    parser = argparse.ArgumentParser()
                    parser.add_argument('command')
                    parser.add_argument('--dataset', required=True)
                    parser.add_argument('--output', required=True)
                    parser.add_argument('--mode', required=True)
                    args = parser.parse_args()
                    assert args.command == 'route'
                    assert args.mode == 'offline'
                    assert os.environ.get('SENTINEL_API_KEY') is None
                    dataset = Path(args.dataset)
                    assert not (dataset / 'sample_messages.csv').exists()
                    assert (dataset / 'media' / 'sentinel.txt').is_file()
                    with (dataset / 'messages.csv').open(encoding='utf-8', newline='') as handle:
                        reader = csv.DictReader(handle)
                        assert reader.fieldnames == input_columns
                        rows = list(reader)
                    assert all(not (set(row) & {{'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids'}}) for row in rows)
                    predictions = [
                        {{'message_id': 'eval_1', 'action': 'notify', 'message_type': 'urgent', 'reason': 'correct', 'confidence': '0.9', 'evidence_message_ids': 'hist_1'}},
                        {{'message_id': 'eval_2', 'action': 'mute', 'message_type': 'promotion', 'reason': 'partly correct', 'confidence': '0.8', 'evidence_message_ids': 'none'}},
                    ]
                    with Path(args.output).open('w', encoding='utf-8', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=output_columns)
                        writer.writeheader()
                        writer.writerows(predictions)
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            previous = os.environ.get("SENTINEL_API_KEY")
            os.environ["SENTINEL_API_KEY"] = "must-not-cross-offline-boundary"
            try:
                report = evaluate(dataset, labels, main_script=stub_main, mode="offline")
            finally:
                if previous is None:
                    os.environ.pop("SENTINEL_API_KEY", None)
                else:
                    os.environ["SENTINEL_API_KEY"] = previous

        result = report.to_dict()
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["accuracy"], {"action": 0.5, "message_type": 1.0, "joint": 0.5})
        self.assertEqual(result["confusion"]["action"]["digest"]["mute"], 1)
        self.assertEqual(result["confusion"]["action"]["notify"]["notify"], 1)
        self.assertEqual(result["evidence"]["valid_reference_rate"], 1.0)
        self.assertEqual(result["evidence"]["exact_match_rate"], 1.0)
        self.assertAlmostEqual(result["calibration"]["brier_score"], 0.325)


if __name__ == "__main__":
    unittest.main()
