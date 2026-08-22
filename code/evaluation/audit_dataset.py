"""Read-only structural and media-integrity audit for the challenge dataset.

This tool is intentionally separate from routing. It reads every provided CSV,
including the labeled sample and current output, only to validate the corpus and
never exports their content to the router or an API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from router.media import sniff_magic  # noqa: E402
from router.output import ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, OUTPUT_COLUMNS  # noqa: E402


MESSAGE_COLUMNS = (
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type", "media_id",
    "forwarded_count",
)
EXPECTED_COLUMNS = {
    "messages.csv": MESSAGE_COLUMNS,
    "sample_messages.csv": MESSAGE_COLUMNS + tuple(OUTPUT_COLUMNS[1:]),
    "output.csv": OUTPUT_COLUMNS,
    "users.csv": ("user_id", "do_not_disturb_window", "messages_opened_30d", "messages_replied_30d", "notifications_dismissed_30d", "messages_reported_30d"),
    "groups.csv": ("group_id", "group_name", "group_type", "member_count", "admin_count", "created_at", "messages_30d"),
    "group_members.csv": ("group_id", "user_id", "role", "joined_at", "messages_sent_30d", "messages_read_30d", "replies_sent_30d", "notifications_dismissed_30d", "group_muted_by_user"),
    "business_accounts.csv": ("business_id", "display_name", "brand_name", "category", "verified", "official_domain", "domain_used_by_sender", "account_age_days", "messages_sent_30d", "user_reports_30d", "domain_used_by_sender_age_days"),
    "user_business_history.csv": ("user_id", "business_id", "why_user_knows_account", "last_activity_at", "allows_promotions", "promotions_opted_out_at", "activity_count_180d", "messages_opened_30d", "messages_dismissed_30d", "messages_replied_30d", "last_reply_at"),
    "message_history.csv": MESSAGE_COLUMNS,
    "message_events.csv": ("user_id", "message_id", "message_opened", "message_replied", "reaction_time_minutes", "notification_dismissed", "muted_after_message", "message_reported"),
    "images.csv": ("image_id", "file_path"),
    "voice_notes.csv": ("voice_note_id", "file_path"),
    "daily_notification_summary.csv": ("user_id", "date", "notifications_sent", "notifications_dismissed"),
}
UNIQUE_KEYS = {
    "messages.csv": ("message_id",),
    "sample_messages.csv": ("message_id",),
    "output.csv": ("message_id",),
    "users.csv": ("user_id",),
    "groups.csv": ("group_id",),
    "group_members.csv": ("group_id", "user_id"),
    "business_accounts.csv": ("business_id",),
    "user_business_history.csv": ("user_id", "business_id"),
    "message_history.csv": ("message_id",),
    "message_events.csv": ("user_id", "message_id"),
    "images.csv": ("image_id",),
    "voice_notes.csv": ("voice_note_id",),
    "daily_notification_summary.csv": ("user_id", "date"),
}


def _read_csv(path: Path, expected: Sequence[str], issues: list[str]) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        issues.append(f"missing_or_unsafe_csv:{path.name}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected):
            issues.append(f"invalid_header:{path.name}")
        rows = list(reader)
    if any(None in row for row in rows):
        issues.append(f"extra_fields:{path.name}")
    return rows


def _duplicates(rows: Sequence[dict[str, str]], fields: Sequence[str]) -> int:
    keys = [tuple(row.get(field, "") for field in fields) for row in rows]
    return len(keys) - len(set(keys))


def _safe_media(root: Path, raw: str) -> Path | None:
    candidate = Path(raw.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    unresolved = root / candidate
    if unresolved.is_symlink():
        return None
    current = root
    for part in candidate.parts[:-1]:
        current = current / part
        if current.is_symlink():
            return None
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() and not resolved.is_symlink() else None


def _audit_media(root: Path, tables: dict[str, list[dict[str, str]]], issues: list[str]) -> dict[str, Any]:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        Image = None
    try:
        import av  # type: ignore[import-not-found]
    except Exception:
        av = None

    formats: Counter[str] = Counter()
    total_bytes = 0
    decoded_images = 0
    decoded_audio = 0
    audio_seconds = 0.0
    hashes: list[str] = []
    entries = [
        ("image", row.get("image_id", ""), row.get("file_path", ""))
        for row in tables["images.csv"]
    ] + [
        ("voice", row.get("voice_note_id", ""), row.get("file_path", ""))
        for row in tables["voice_notes.csv"]
    ]
    for declared_kind, media_id, raw_path in entries:
        path = _safe_media(root, raw_path)
        if path is None:
            issues.append(f"unsafe_or_missing_media:{media_id}")
            continue
        data = path.read_bytes()
        total_bytes += len(data)
        hashes.append(hashlib.sha256(data).hexdigest())
        actual_kind, _mime, format_name = sniff_magic(data[:64])
        formats[format_name or "unknown"] += 1
        if actual_kind != declared_kind:
            issues.append(f"media_kind_mismatch:{media_id}")
            continue
        try:
            if declared_kind == "image" and Image is not None:
                with Image.open(path) as image:
                    image.load()
                    if image.width <= 0 or image.height <= 0:
                        raise ValueError("invalid dimensions")
                decoded_images += 1
            elif declared_kind == "voice" and av is not None:
                with av.open(str(path)) as container:
                    samples = 0
                    rate = 0
                    for frame in container.decode(audio=0):
                        samples += int(frame.samples)
                        rate = int(frame.sample_rate or rate)
                    if samples <= 0 or rate <= 0:
                        raise ValueError("no decoded audio")
                    audio_seconds += samples / rate
                decoded_audio += 1
        except Exception:
            issues.append(f"media_decode_failed:{media_id}")
    return {
        "files": len(entries),
        "bytes": total_bytes,
        "formats": dict(sorted(formats.items())),
        "unique_sha256": len(set(hashes)),
        "duplicate_content_files": len(hashes) - len(set(hashes)),
        "deep_decoders": {"pillow": Image is not None, "pyav": av is not None},
        "decoded_images": decoded_images,
        "decoded_audio": decoded_audio,
        "audio_seconds": round(audio_seconds, 3),
    }


def audit(dataset_dir: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    issues: list[str] = []
    tables = {
        name: _read_csv(root / name, columns, issues)
        for name, columns in EXPECTED_COLUMNS.items()
    }
    for name, fields in UNIQUE_KEYS.items():
        count = _duplicates(tables[name], fields)
        if count:
            issues.append(f"duplicate_keys:{name}:{count}")

    users = {row["user_id"] for row in tables["users.csv"]}
    groups = {row["group_id"] for row in tables["groups.csv"]}
    businesses = {row["business_id"] for row in tables["business_accounts.csv"]}
    history = {(row["user_id"], row["message_id"]) for row in tables["message_history.csv"]}
    image_ids = {row["image_id"] for row in tables["images.csv"]}
    voice_ids = {row["voice_note_id"] for row in tables["voice_notes.csv"]}
    dangling: Counter[str] = Counter()
    for row in tables["messages.csv"]:
        dangling["message_user"] += row["user_id"] not in users
        dangling["message_group"] += bool(row["group_id"]) and row["group_id"] not in groups
        dangling["message_business"] += bool(row["business_id"]) and row["business_id"] not in businesses
        dangling["message_image"] += row["media_type"] == "image" and row["media_id"] not in image_ids
        dangling["message_voice"] += row["media_type"] == "voice" and row["media_id"] not in voice_ids
    for row in tables["group_members.csv"]:
        dangling["membership_user"] += row["user_id"] not in users
        dangling["membership_group"] += row["group_id"] not in groups
    for row in tables["user_business_history.csv"]:
        dangling["business_history_user"] += row["user_id"] not in users
        dangling["business_history_business"] += row["business_id"] not in businesses
    for row in tables["message_events.csv"]:
        dangling["event_history"] += (row["user_id"], row["message_id"]) not in history
    for row in tables["daily_notification_summary.csv"]:
        dangling["daily_user"] += row["user_id"] not in users
    dangling = Counter({key: value for key, value in dangling.items() if value})
    if dangling:
        issues.extend(f"dangling:{key}:{value}" for key, value in sorted(dangling.items()))

    message_ids = [row["message_id"] for row in tables["messages.csv"]]
    output_ids = [row["message_id"] for row in tables["output.csv"]]
    if output_ids and output_ids != message_ids:
        issues.append("output_message_order_or_coverage_mismatch")
    for row in tables["output.csv"] + tables["sample_messages.csv"]:
        if row.get("action") not in ALLOWED_ACTIONS:
            issues.append("invalid_labeled_action")
            break
        if row.get("message_type") not in ALLOWED_MESSAGE_TYPES:
            issues.append("invalid_labeled_message_type")
            break

    media = _audit_media(root, tables, issues)
    return {
        "ok": not issues,
        "dataset": str(root),
        "csv": {
            name: {
                "rows": len(rows),
                "bytes": (root / name).stat().st_size if (root / name).is_file() else 0,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()
                if (root / name).is_file() else "",
            }
            for name, rows in sorted(tables.items())
        },
        "media": media,
        "issues": issues,
    }


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPOSITORY_ROOT / "dataset")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.dataset)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"dataset audit error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        _write_json_atomic(args.json_output, report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
