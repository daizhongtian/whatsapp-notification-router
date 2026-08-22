"""Typed domain models for the offline notification router.

The objects in this module deliberately contain data only.  In particular, text
from a message is never interpreted as configuration or executable policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Mapping


class Action(StrEnum):
    """Allowed output actions from the challenge contract."""

    NOTIFY = "notify"
    DIGEST = "digest"
    MUTE = "mute"


class MessageType(StrEnum):
    """Allowed output categories from the challenge contract."""

    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"


class ConversationType(StrEnum):
    PERSONAL = "personal"
    GROUP = "group"
    BUSINESS = "business"
    UNKNOWN = "unknown"


class MediaType(StrEnum):
    NONE = ""
    IMAGE = "image"
    VOICE = "voice"


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """An incoming or historical message in the official input schema."""

    message_id: str
    user_id: str
    conversation_type: ConversationType
    group_id: str | None
    business_id: str | None
    sender_user_id: str | None
    created_at: datetime | None
    message_text: str
    media_type: MediaType
    media_id: str | None
    forwarded_count: int = 0

    @property
    def conversation_id(self) -> str | None:
        """Return the entity that best identifies the conversation."""

        if self.conversation_type is ConversationType.GROUP:
            return self.group_id
        if self.conversation_type is ConversationType.BUSINESS:
            return self.business_id
        if self.conversation_type is ConversationType.PERSONAL:
            return self.sender_user_id
        return self.group_id or self.business_id or self.sender_user_id


# Semantic aliases keep call sites readable while preserving a single schema.
IncomingMessage = MessageRecord
HistoricalMessage = MessageRecord
Message = MessageRecord


@dataclass(frozen=True, slots=True)
class UserProfile:
    user_id: str
    do_not_disturb_window: str | None
    messages_opened_30d: int = 0
    messages_replied_30d: int = 0
    notifications_dismissed_30d: int = 0
    messages_reported_30d: int = 0


@dataclass(frozen=True, slots=True)
class GroupProfile:
    group_id: str
    group_name: str
    group_type: str
    member_count: int = 0
    admin_count: int = 0
    created_at: datetime | None = None
    messages_30d: int = 0


@dataclass(frozen=True, slots=True)
class GroupMembership:
    group_id: str
    user_id: str
    role: str
    joined_at: datetime | None = None
    messages_sent_30d: int = 0
    messages_read_30d: int = 0
    replies_sent_30d: int = 0
    notifications_dismissed_30d: int = 0
    group_muted_by_user: bool = False


@dataclass(frozen=True, slots=True)
class BusinessAccount:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool = False
    official_domain: str | None = None
    domain_used_by_sender: str | None = None
    account_age_days: int = 0
    messages_sent_30d: int = 0
    user_reports_30d: int = 0
    domain_used_by_sender_age_days: int = 0


@dataclass(frozen=True, slots=True)
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: datetime | None = None
    allows_promotions: bool = False
    promotions_opted_out_at: datetime | None = None
    activity_count_180d: int = 0
    messages_opened_30d: int = 0
    messages_dismissed_30d: int = 0
    messages_replied_30d: int = 0
    last_reply_at: datetime | None = None


# Shorter name for integrations that do not mirror the CSV filename.
BusinessRelationship = UserBusinessHistory


@dataclass(frozen=True, slots=True)
class MessageEvent:
    user_id: str
    message_id: str
    message_opened: bool = False
    message_replied: bool = False
    reaction_time_minutes: int | None = None
    notification_dismissed: bool = False
    muted_after_message: bool = False
    message_reported: bool = False

    @property
    def positive_engagement(self) -> bool:
        return self.message_replied or self.message_opened

    @property
    def negative_engagement(self) -> bool:
        return (
            self.notification_dismissed
            or self.muted_after_message
            or self.message_reported
        )


@dataclass(frozen=True, slots=True)
class MediaAsset:
    media_id: str
    media_type: MediaType
    """The media path, relative to the dataset root and already validated."""

    file_path: str


@dataclass(frozen=True, slots=True)
class DailyNotificationSummary:
    user_id: str
    day: date | None
    notifications_sent: int = 0
    notifications_dismissed: int = 0


@dataclass(frozen=True, slots=True)
class Evidence:
    """A same-user, prior historical message returned by retrieval."""

    message: HistoricalMessage
    score: float
    event: MessageEvent | None = None

    @property
    def message_id(self) -> str:
        return self.message.message_id


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """All participant-facing context needed by the deterministic router."""

    dataset_dir: str
    messages: tuple[IncomingMessage, ...]
    users: Mapping[str, UserProfile] = field(default_factory=dict)
    groups: Mapping[str, GroupProfile] = field(default_factory=dict)
    group_memberships: Mapping[tuple[str, str], GroupMembership] = field(
        default_factory=dict
    )
    businesses: Mapping[str, BusinessAccount] = field(default_factory=dict)
    user_business_history: Mapping[
        tuple[str, str], UserBusinessHistory
    ] = field(default_factory=dict)
    message_history: tuple[HistoricalMessage, ...] = ()
    message_events: Mapping[tuple[str, str], MessageEvent] = field(
        default_factory=dict
    )
    media: Mapping[tuple[MediaType, str], MediaAsset] = field(default_factory=dict)
    daily_summaries: Mapping[str, tuple[DailyNotificationSummary, ...]] = field(
        default_factory=dict
    )


# Friendly compatibility alias.
RouterData = DatasetBundle


@dataclass(frozen=True, slots=True)
class MessageContext:
    message: IncomingMessage
    user: UserProfile | None = None
    group: GroupProfile | None = None
    group_membership: GroupMembership | None = None
    business: BusinessAccount | None = None
    business_history: UserBusinessHistory | None = None
    media_asset: MediaAsset | None = None
    daily_summaries: tuple[DailyNotificationSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class SafetySignals:
    risk_score: float = 0.0
    prompt_injection: bool = False
    asks_for_secret: bool = False
    suspicious_link: bool = False
    domain_mismatch: bool = False
    impersonation: bool = False
    coercive_urgency: bool = False
    unsafe_advice: bool = False
    chain_message: bool = False
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    """Structured, inspectable features consumed by :class:`RouterPolicy`."""

    message_type: MessageType
    category_scores: Mapping[MessageType, float]
    urgency: float
    relevance: float
    engagement: float
    repetition: float
    quiet_hours: bool
    direct_mention: bool
    explicit_time_constraint: bool
    trusted_context: bool
    muted_context: bool
    promotions_opted_out: bool
    high_notification_load: bool
    safety: SafetySignals
    evidence: tuple[Evidence, ...] = ()
    notes: tuple[str, ...] = ()

    def category_score(self, message_type: MessageType) -> float:
        return float(self.category_scores.get(message_type, 0.0))


@dataclass(frozen=True, slots=True)
class Prediction:
    """A contract-valid routing prediction plus optional audit signals."""

    message_id: str
    action: Action
    message_type: MessageType
    reason: str
    confidence: float
    evidence_message_ids: tuple[str, ...] = ()
    signals: RoutingSignals | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("prediction message_id must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prediction confidence must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("prediction reason must not be empty")
        if any(not message_id for message_id in self.evidence_message_ids):
            raise ValueError("evidence message IDs must not be empty")

    def to_row(self) -> dict[str, str | float]:
        """Return the exact six-column output row required by the challenge."""

        return {
            "message_id": self.message_id,
            "action": self.action.value,
            "message_type": self.message_type.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "evidence_message_ids": (
                ";".join(self.evidence_message_ids)
                if self.evidence_message_ids
                else "none"
            ),
        }

    # Common serialization spelling used by lightweight integrations.
    as_dict = to_row


OUTPUT_COLUMNS: tuple[str, ...] = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

