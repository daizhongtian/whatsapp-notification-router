"""Defensive, dependency-free ingestion for participant-facing CSV data.

Only a fixed allow-list of context files is opened.  Delimiter sniffing and
dynamic filenames are intentionally avoided because the dataset is input, not
trusted configuration.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, TypeVar

from .models import (
    BusinessAccount,
    ConversationType,
    DailyNotificationSummary,
    DatasetBundle,
    GroupMembership,
    GroupProfile,
    IncomingMessage,
    MediaAsset,
    MediaType,
    MessageContext,
    MessageEvent,
    MessageRecord,
    UserBusinessHistory,
    UserProfile,
)


MAX_DATA_FILE_BYTES = 128 * 1024 * 1024
MAX_FIELD_CHARACTERS = 250_000
MAX_ROWS_PER_FILE = 2_000_000


class DataValidationError(ValueError):
    """Raised when an input file cannot safely satisfy its declared schema."""


_MESSAGE_COLUMNS = (
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

_USER_COLUMNS = (
    "user_id",
    "do_not_disturb_window",
    "messages_opened_30d",
    "messages_replied_30d",
    "notifications_dismissed_30d",
    "messages_reported_30d",
)

_GROUP_COLUMNS = (
    "group_id",
    "group_name",
    "group_type",
    "member_count",
    "admin_count",
    "created_at",
    "messages_30d",
)

_MEMBERSHIP_COLUMNS = (
    "group_id",
    "user_id",
    "role",
    "joined_at",
    "messages_sent_30d",
    "messages_read_30d",
    "replies_sent_30d",
    "notifications_dismissed_30d",
    "group_muted_by_user",
)

_BUSINESS_COLUMNS = (
    "business_id",
    "display_name",
    "brand_name",
    "category",
    "verified",
    "official_domain",
    "domain_used_by_sender",
    "account_age_days",
    "messages_sent_30d",
    "user_reports_30d",
    "domain_used_by_sender_age_days",
)

_BUSINESS_HISTORY_COLUMNS = (
    "user_id",
    "business_id",
    "why_user_knows_account",
    "last_activity_at",
    "allows_promotions",
    "promotions_opted_out_at",
    "activity_count_180d",
    "messages_opened_30d",
    "messages_dismissed_30d",
    "messages_replied_30d",
    "last_reply_at",
)

_EVENT_COLUMNS = (
    "user_id",
    "message_id",
    "message_opened",
    "message_replied",
    "reaction_time_minutes",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
)

_DAILY_COLUMNS = (
    "user_id",
    "date",
    "notifications_sent",
    "notifications_dismissed",
)


def _clean_text(value: str | None, *, field_name: str) -> str:
    """Remove control characters that can confuse logs while preserving text."""

    if value is None:
        return ""
    if len(value) > MAX_FIELD_CHARACTERS:
        raise DataValidationError(
            f"field {field_name!r} exceeds {MAX_FIELD_CHARACTERS} characters"
        )
    # Keep line breaks and tabs, which are meaningful in message bodies.  NUL and
    # other C0 controls are never meaningful in these CSV schemas.
    return "".join(
        character
        for character in value
        if character in "\n\r\t" or ord(character) >= 32
    ).strip()


def _required_id(row: Mapping[str, str], field_name: str) -> str:
    value = _clean_text(row.get(field_name), field_name=field_name)
    if not value:
        raise DataValidationError(f"required identifier {field_name!r} is empty")
    if len(value) > 512:
        raise DataValidationError(f"identifier {field_name!r} is too long")
    return value


def _optional_text(row: Mapping[str, str], field_name: str) -> str | None:
    return _clean_text(row.get(field_name), field_name=field_name) or None


def _parse_nonnegative_int(
    value: str | None, *, field_name: str, optional: bool = False
) -> int | None:
    cleaned = _clean_text(value, field_name=field_name)
    if not cleaned:
        return None if optional else 0
    try:
        parsed = int(cleaned)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field_name!r} must be an integer") from exc
    if parsed < 0:
        raise DataValidationError(f"{field_name!r} must be non-negative")
    return min(parsed, 2_147_483_647)


def _parse_bool(value: str | None, *, field_name: str) -> bool:
    cleaned = _clean_text(value, field_name=field_name).casefold()
    if cleaned in {"1", "true", "yes", "y"}:
        return True
    if cleaned in {"", "0", "false", "no", "n"}:
        return False
    raise DataValidationError(f"{field_name!r} must be a boolean or 0/1")


def _parse_datetime(value: str | None, *, field_name: str) -> datetime | None:
    cleaned = _clean_text(value, field_name=field_name)
    if not cleaned:
        return None
    normalized = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DataValidationError(
            f"{field_name!r} is not a valid ISO date or datetime"
        ) from exc


def _parse_date(value: str | None, *, field_name: str) -> date | None:
    cleaned = _clean_text(value, field_name=field_name)
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError as exc:
        raise DataValidationError(f"{field_name!r} is not a valid date") from exc


def _safe_relative_path(value: str | None, *, field_name: str) -> str:
    cleaned = _clean_text(value, field_name=field_name)
    if not cleaned:
        raise DataValidationError(f"{field_name!r} must not be empty")
    path = Path(cleaned.replace("\\", "/"))
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise DataValidationError(f"{field_name!r} must remain inside the dataset")
    return path.as_posix()


def _read_rows(
    dataset_dir: Path,
    filename: str,
    required_columns: Iterable[str],
    *,
    optional: bool,
) -> Iterator[dict[str, str]]:
    path = dataset_dir / filename
    if not path.exists():
        if optional:
            return
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file():
        raise DataValidationError(f"expected a regular CSV file: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(dataset_dir.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise DataValidationError(f"CSV path escapes the dataset directory: {filename}") from exc
    if resolved.stat().st_size > MAX_DATA_FILE_BYTES:
        raise DataValidationError(f"CSV file is unexpectedly large: {path.name}")

    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=",")
        headers = reader.fieldnames
        if not headers:
            raise DataValidationError(f"CSV file has no header: {path.name}")
        if len(headers) != len(set(headers)):
            raise DataValidationError(f"CSV file has duplicate headers: {path.name}")
        missing = tuple(column for column in required_columns if column not in headers)
        if missing:
            raise DataValidationError(
                f"{path.name} is missing required columns: {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            if row_number > MAX_ROWS_PER_FILE + 1:
                raise DataValidationError(f"too many rows in {path.name}")
            if None in row:
                raise DataValidationError(
                    f"unexpected extra fields in {path.name} row {row_number}"
                )
            yield {
                header: _clean_text(row.get(header), field_name=header)
                for header in headers
            }


def _conversation_type(value: str) -> ConversationType:
    try:
        return ConversationType(value.casefold())
    except ValueError:
        return ConversationType.UNKNOWN


def _media_type(value: str) -> MediaType:
    try:
        return MediaType(value.casefold())
    except ValueError as exc:
        raise DataValidationError(f"unknown media_type {value!r}") from exc


def _message_from_row(row: Mapping[str, str]) -> MessageRecord:
    return MessageRecord(
        message_id=_required_id(row, "message_id"),
        user_id=_required_id(row, "user_id"),
        conversation_type=_conversation_type(row.get("conversation_type", "")),
        group_id=_optional_text(row, "group_id"),
        business_id=_optional_text(row, "business_id"),
        sender_user_id=_optional_text(row, "sender_user_id"),
        created_at=_parse_datetime(row.get("created_at"), field_name="created_at"),
        message_text=_clean_text(row.get("message_text"), field_name="message_text"),
        media_type=_media_type(row.get("media_type", "")),
        media_id=_optional_text(row, "media_id"),
        forwarded_count=int(
            _parse_nonnegative_int(
                row.get("forwarded_count"), field_name="forwarded_count"
            )
            or 0
        ),
    )


T = TypeVar("T")
K = TypeVar("K")


def _unique_map(
    values: Iterable[T], key: Callable[[T], K], *, source_name: str
) -> dict[K, T]:
    result: dict[K, T] = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise DataValidationError(
                f"duplicate key {item_key!r} in {source_name}"
            )
        result[item_key] = value
    return result


def load_dataset(dataset_dir: str | Path) -> DatasetBundle:
    """Load the official dataset and context without consulting label examples.

    ``messages.csv`` is required.  Context files are optional so the router can
    still operate conservatively in unit tests or partial offline deployments.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    messages = tuple(
        _message_from_row(row)
        for row in _read_rows(
            root, "messages.csv", _MESSAGE_COLUMNS, optional=False
        )
    )
    _unique_map(messages, lambda message: message.message_id, source_name="messages.csv")

    users = _unique_map(
        (
            UserProfile(
                user_id=_required_id(row, "user_id"),
                do_not_disturb_window=_optional_text(row, "do_not_disturb_window"),
                messages_opened_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_opened_30d"),
                        field_name="messages_opened_30d",
                    )
                    or 0
                ),
                messages_replied_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_replied_30d"),
                        field_name="messages_replied_30d",
                    )
                    or 0
                ),
                notifications_dismissed_30d=int(
                    _parse_nonnegative_int(
                        row.get("notifications_dismissed_30d"),
                        field_name="notifications_dismissed_30d",
                    )
                    or 0
                ),
                messages_reported_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_reported_30d"),
                        field_name="messages_reported_30d",
                    )
                    or 0
                ),
            )
            for row in _read_rows(root, "users.csv", _USER_COLUMNS, optional=True)
        ),
        lambda user: user.user_id,
        source_name="users.csv",
    )

    groups = _unique_map(
        (
            GroupProfile(
                group_id=_required_id(row, "group_id"),
                group_name=_clean_text(row.get("group_name"), field_name="group_name"),
                group_type=_clean_text(row.get("group_type"), field_name="group_type"),
                member_count=int(
                    _parse_nonnegative_int(
                        row.get("member_count"), field_name="member_count"
                    )
                    or 0
                ),
                admin_count=int(
                    _parse_nonnegative_int(
                        row.get("admin_count"), field_name="admin_count"
                    )
                    or 0
                ),
                created_at=_parse_datetime(
                    row.get("created_at"), field_name="created_at"
                ),
                messages_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_30d"), field_name="messages_30d"
                    )
                    or 0
                ),
            )
            for row in _read_rows(root, "groups.csv", _GROUP_COLUMNS, optional=True)
        ),
        lambda group: group.group_id,
        source_name="groups.csv",
    )

    memberships = _unique_map(
        (
            GroupMembership(
                group_id=_required_id(row, "group_id"),
                user_id=_required_id(row, "user_id"),
                role=_clean_text(row.get("role"), field_name="role"),
                joined_at=_parse_datetime(
                    row.get("joined_at"), field_name="joined_at"
                ),
                messages_sent_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_sent_30d"),
                        field_name="messages_sent_30d",
                    )
                    or 0
                ),
                messages_read_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_read_30d"),
                        field_name="messages_read_30d",
                    )
                    or 0
                ),
                replies_sent_30d=int(
                    _parse_nonnegative_int(
                        row.get("replies_sent_30d"),
                        field_name="replies_sent_30d",
                    )
                    or 0
                ),
                notifications_dismissed_30d=int(
                    _parse_nonnegative_int(
                        row.get("notifications_dismissed_30d"),
                        field_name="notifications_dismissed_30d",
                    )
                    or 0
                ),
                group_muted_by_user=_parse_bool(
                    row.get("group_muted_by_user"),
                    field_name="group_muted_by_user",
                ),
            )
            for row in _read_rows(
                root, "group_members.csv", _MEMBERSHIP_COLUMNS, optional=True
            )
        ),
        lambda membership: (membership.group_id, membership.user_id),
        source_name="group_members.csv",
    )

    businesses = _unique_map(
        (
            BusinessAccount(
                business_id=_required_id(row, "business_id"),
                display_name=_clean_text(
                    row.get("display_name"), field_name="display_name"
                ),
                brand_name=_clean_text(
                    row.get("brand_name"), field_name="brand_name"
                ),
                category=_clean_text(row.get("category"), field_name="category"),
                verified=_parse_bool(row.get("verified"), field_name="verified"),
                official_domain=_optional_text(row, "official_domain"),
                domain_used_by_sender=_optional_text(row, "domain_used_by_sender"),
                account_age_days=int(
                    _parse_nonnegative_int(
                        row.get("account_age_days"), field_name="account_age_days"
                    )
                    or 0
                ),
                messages_sent_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_sent_30d"),
                        field_name="messages_sent_30d",
                    )
                    or 0
                ),
                user_reports_30d=int(
                    _parse_nonnegative_int(
                        row.get("user_reports_30d"),
                        field_name="user_reports_30d",
                    )
                    or 0
                ),
                domain_used_by_sender_age_days=int(
                    _parse_nonnegative_int(
                        row.get("domain_used_by_sender_age_days"),
                        field_name="domain_used_by_sender_age_days",
                    )
                    or 0
                ),
            )
            for row in _read_rows(
                root, "business_accounts.csv", _BUSINESS_COLUMNS, optional=True
            )
        ),
        lambda business: business.business_id,
        source_name="business_accounts.csv",
    )

    business_history = _unique_map(
        (
            UserBusinessHistory(
                user_id=_required_id(row, "user_id"),
                business_id=_required_id(row, "business_id"),
                why_user_knows_account=_clean_text(
                    row.get("why_user_knows_account"),
                    field_name="why_user_knows_account",
                ),
                last_activity_at=_parse_datetime(
                    row.get("last_activity_at"), field_name="last_activity_at"
                ),
                allows_promotions=_parse_bool(
                    row.get("allows_promotions"), field_name="allows_promotions"
                ),
                promotions_opted_out_at=_parse_datetime(
                    row.get("promotions_opted_out_at"),
                    field_name="promotions_opted_out_at",
                ),
                activity_count_180d=int(
                    _parse_nonnegative_int(
                        row.get("activity_count_180d"),
                        field_name="activity_count_180d",
                    )
                    or 0
                ),
                messages_opened_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_opened_30d"),
                        field_name="messages_opened_30d",
                    )
                    or 0
                ),
                messages_dismissed_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_dismissed_30d"),
                        field_name="messages_dismissed_30d",
                    )
                    or 0
                ),
                messages_replied_30d=int(
                    _parse_nonnegative_int(
                        row.get("messages_replied_30d"),
                        field_name="messages_replied_30d",
                    )
                    or 0
                ),
                last_reply_at=_parse_datetime(
                    row.get("last_reply_at"), field_name="last_reply_at"
                ),
            )
            for row in _read_rows(
                root,
                "user_business_history.csv",
                _BUSINESS_HISTORY_COLUMNS,
                optional=True,
            )
        ),
        lambda relationship: (relationship.user_id, relationship.business_id),
        source_name="user_business_history.csv",
    )

    historical_messages = tuple(
        _message_from_row(row)
        for row in _read_rows(
            root, "message_history.csv", _MESSAGE_COLUMNS, optional=True
        )
    )
    _unique_map(
        historical_messages,
        lambda message: message.message_id,
        source_name="message_history.csv",
    )

    events = _unique_map(
        (
            MessageEvent(
                user_id=_required_id(row, "user_id"),
                message_id=_required_id(row, "message_id"),
                message_opened=_parse_bool(
                    row.get("message_opened"), field_name="message_opened"
                ),
                message_replied=_parse_bool(
                    row.get("message_replied"), field_name="message_replied"
                ),
                reaction_time_minutes=_parse_nonnegative_int(
                    row.get("reaction_time_minutes"),
                    field_name="reaction_time_minutes",
                    optional=True,
                ),
                notification_dismissed=_parse_bool(
                    row.get("notification_dismissed"),
                    field_name="notification_dismissed",
                ),
                muted_after_message=_parse_bool(
                    row.get("muted_after_message"),
                    field_name="muted_after_message",
                ),
                message_reported=_parse_bool(
                    row.get("message_reported"), field_name="message_reported"
                ),
            )
            for row in _read_rows(
                root, "message_events.csv", _EVENT_COLUMNS, optional=True
            )
        ),
        lambda event: (event.user_id, event.message_id),
        source_name="message_events.csv",
    )

    media_values: list[MediaAsset] = []
    for row in _read_rows(
        root, "images.csv", ("image_id", "file_path"), optional=True
    ):
        media_values.append(
            MediaAsset(
                media_id=_required_id(row, "image_id"),
                media_type=MediaType.IMAGE,
                file_path=_safe_relative_path(
                    row.get("file_path"), field_name="file_path"
                ),
            )
        )
    for row in _read_rows(
        root, "voice_notes.csv", ("voice_note_id", "file_path"), optional=True
    ):
        media_values.append(
            MediaAsset(
                media_id=_required_id(row, "voice_note_id"),
                media_type=MediaType.VOICE,
                file_path=_safe_relative_path(
                    row.get("file_path"), field_name="file_path"
                ),
            )
        )
    media = _unique_map(
        media_values,
        lambda asset: (asset.media_type, asset.media_id),
        source_name="media indexes",
    )

    daily_by_user: defaultdict[str, list[DailyNotificationSummary]] = defaultdict(list)
    for row in _read_rows(
        root,
        "daily_notification_summary.csv",
        _DAILY_COLUMNS,
        optional=True,
    ):
        summary = DailyNotificationSummary(
            user_id=_required_id(row, "user_id"),
            day=_parse_date(row.get("date"), field_name="date"),
            notifications_sent=int(
                _parse_nonnegative_int(
                    row.get("notifications_sent"), field_name="notifications_sent"
                )
                or 0
            ),
            notifications_dismissed=int(
                _parse_nonnegative_int(
                    row.get("notifications_dismissed"),
                    field_name="notifications_dismissed",
                )
                or 0
            ),
        )
        daily_by_user[summary.user_id].append(summary)
    daily_summaries = {
        user_id: tuple(
            sorted(
                summaries,
                key=lambda summary: (summary.day or date.min),
            )
        )
        for user_id, summaries in daily_by_user.items()
    }

    return DatasetBundle(
        dataset_dir=str(root),
        messages=messages,
        users=users,
        groups=groups,
        group_memberships=memberships,
        businesses=businesses,
        user_business_history=business_history,
        message_history=historical_messages,
        message_events=events,
        media=media,
        daily_summaries=daily_summaries,
    )


def build_context(message: IncomingMessage, data: DatasetBundle) -> MessageContext:
    """Join a message to its recipient-specific context with safe fallbacks."""

    group = data.groups.get(message.group_id) if message.group_id else None
    membership = (
        data.group_memberships.get((message.group_id, message.user_id))
        if message.group_id
        else None
    )
    business = (
        data.businesses.get(message.business_id) if message.business_id else None
    )
    business_history = (
        data.user_business_history.get((message.user_id, message.business_id))
        if message.business_id
        else None
    )
    media_asset = (
        data.media.get((message.media_type, message.media_id))
        if message.media_id
        else None
    )
    summaries = data.daily_summaries.get(message.user_id, ())
    return MessageContext(
        message=message,
        user=data.users.get(message.user_id),
        group=group,
        group_membership=membership,
        business=business,
        business_history=business_history,
        media_asset=media_asset,
        daily_summaries=summaries,
    )


def resolve_media_path(data: DatasetBundle, asset: MediaAsset) -> Path:
    """Resolve a validated media asset while enforcing dataset containment."""

    root = Path(data.dataset_dir).resolve()
    resolved = (root / asset.file_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DataValidationError("media path escapes the dataset directory") from exc
    return resolved


# Familiar alternative used by some entry points.
load_data = load_dataset
