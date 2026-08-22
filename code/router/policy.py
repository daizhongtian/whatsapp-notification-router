"""Personalized deterministic routing policy with safety-first precedence."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from .data import build_context
from .models import (
    Action,
    DatasetBundle,
    Evidence,
    IncomingMessage,
    MessageContext,
    MessageType,
    Prediction,
    RoutingSignals,
)
from .retrieval import HistoryRetriever, ensure_same_user_evidence
from .signals import analyze_signals


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _select_evidence(signals: RoutingSignals, *, limit: int = 2) -> tuple[str, ...]:
    """Expose only evidence that materially contributed to the decision."""

    selected: list[str] = []
    for item in signals.evidence:
        event_is_strong = bool(
            item.event
            and (
                item.event.message_replied
                or item.event.notification_dismissed
                or item.event.muted_after_message
                or item.event.message_reported
            )
        )
        minimum_score = 0.32 if event_is_strong else 0.36
        # ASR/OCR wording is inherently noisier than source text.  Permit one
        # opened historical message at the weaker threshold only when media is
        # the message's sole semantic content; do not broaden ordinary text
        # evidence or append multiple weak references.
        if (
            not selected
            and "media_content_used" in signals.notes
            and item.event is not None
            and item.event.message_opened
        ):
            minimum_score = min(minimum_score, 0.32)
        if item.score < minimum_score:
            continue
        if item.message_id not in selected:
            selected.append(item.message_id)
        if len(selected) >= max(0, limit):
            break
    return tuple(selected)


def _safety_reason(signals: RoutingSignals) -> str:
    safety = signals.safety
    if safety.prompt_injection and safety.asks_for_secret:
        return "Muted because it manipulates routing and requests sensitive credentials."
    if safety.domain_mismatch and safety.asks_for_secret:
        return "Muted because a mismatched domain asks for sensitive credentials."
    if safety.asks_for_secret:
        return "Muted because it asks the recipient to disclose a sensitive code or account detail."
    if safety.domain_mismatch:
        return "Muted because the linked domain does not match the claimed business."
    if safety.unsafe_advice:
        return "Muted because the forwarded advice could cause unsafe health decisions."
    if safety.suspicious_link and safety.coercive_urgency:
        return "Muted because a suspicious link is paired with coercive urgency."
    if safety.impersonation:
        return "Muted because the account or payment claim has strong impersonation signals."
    if safety.prompt_injection:
        return "Muted because embedded router instructions are untrusted message content."
    if safety.chain_message:
        return "Muted as a chain message that pressures recipients to forward it."
    return "Muted because multiple safety signals indicate a suspicious message."


def _normal_reason(action: Action, signals: RoutingSignals) -> str:
    message_type = signals.message_type
    notes = frozenset(signals.notes)
    if action is Action.NOTIFY:
        if signals.direct_mention:
            return "A direct mention carries a credible time-sensitive request, even in a muted conversation."
        if "immediate_health_call" in notes:
            return "An immediate health-related callback request warrants attention now."
        if "lost_item_deadline" in notes:
            return "A time-limited lost-item or identity-document update requires prompt action."
        if "school_action_notice" in notes:
            return "A school schedule or transport change requires timely action from the recipient."
        if "operational_alert" in notes:
            return "A near-term operational disruption affects access or essential service and needs attention now."
        if "work_urgent" in notes:
            return "A direct work request has a credible near-term deadline and needs a prompt response."
        if "trusted_payment_notice" in notes:
            return "A payment deadline from a trusted administrator or known account needs timely attention."
        if "business_schedule_change" in notes:
            return "A trusted booking or service schedule changed and may require immediate adjustment."
        if "upcoming_event_context" in notes:
            return "A relevant upcoming event has a concrete schedule or access change requiring action."
        if message_type is MessageType.URGENT:
            return "The message has a credible near-term deadline or action requirement."
        if message_type is MessageType.PAYMENT:
            return "A relevant, trusted payment deadline needs timely attention."
        if message_type is MessageType.EVENT:
            return "A relevant schedule or access change requires timely action."
        if message_type is MessageType.BUSINESS_UPDATE:
            return "A trusted transactional update needs timely attention."
        return "A relevant personal request is time-sensitive enough to notify now."

    if action is Action.MUTE:
        if message_type is MessageType.PROMOTION and signals.promotions_opted_out:
            return "Muted because the recipient explicitly opted out of promotions from this sender."
        if "poor_sender_reputation" in notes:
            return "Muted because the sender has strong report or reputation signals indicating unwanted content."
        if "similar_history_was_rejected" in notes:
            return "Muted because closely related messages were previously dismissed, muted, or reported."
        if signals.repetition >= 0.60 and signals.engagement < 0.48:
            return "Muted because repeated similar messages received little engagement and frequent dismissal."
        if signals.muted_context and signals.urgency < 0.55:
            return "Muted because the recipient muted this conversation and no credible urgency overrides it."
        if message_type in {MessageType.SPAM, MessageType.FORWARD}:
            return "Muted as low-value forwarded or repetitive content with no actionable personal relevance."
        return "Muted because low relevance and prior behavior indicate unwanted noise."

    if signals.quiet_hours:
        return "The message is useful but non-critical during the recipient's quiet hours, so it can wait."
    if "explicitly_nonurgent" in notes:
        return "The sender explicitly indicates that no immediate response is needed, so this can wait."
    if "vague_plan" in notes:
        return "The tentative personal plan has no confirmed time or immediate action requirement."
    if signals.high_notification_load:
        return "The message is relevant but non-urgent, and batching avoids adding to today's high notification load."
    if message_type is MessageType.PROMOTION:
        return "The recipient permits this promotion, but the offer does not justify an interruption."
    if message_type is MessageType.GREETING:
        return "This friendly greeting is personally relevant but has no action or time requirement."
    if message_type is MessageType.BUSINESS_UPDATE:
        return "This appears to be a legitimate informational update without an immediate service change."
    if message_type is MessageType.PAYMENT:
        return "The payment information is relevant, but no credible immediate deadline requires interruption."
    if message_type is MessageType.EVENT:
        return "The event information is useful, but no immediate schedule change requires interruption."
    if message_type is MessageType.PERSONAL:
        return "The personal message is relevant but has no credible near-term deadline or urgent request."
    if message_type is MessageType.UNKNOWN:
        return "No reliable urgency or safety signal was found, so the message can be reviewed later."
    return "The message is relevant and safe but has no immediate action requirement."


def _action_scores(signals: RoutingSignals) -> dict[Action, float]:
    """Produce inspectable scores after safety hard gates have been checked."""

    urgency = signals.urgency
    relevance = signals.relevance
    engagement = signals.engagement
    repetition = signals.repetition
    risk = signals.safety.risk_score

    notify = 0.08 + 0.70 * urgency + 0.28 * relevance + 0.13 * engagement
    notify += 0.17 if signals.direct_mention else 0.0
    notify += 0.10 if signals.trusted_context else 0.0
    notify -= 0.20 if signals.quiet_hours else 0.0
    notify -= 0.11 if signals.high_notification_load else 0.0
    notify -= 0.20 * repetition + 0.75 * risk

    digest = 0.40 + 0.25 * relevance + 0.12 * engagement
    digest += 0.13 if signals.quiet_hours else 0.0
    digest += 0.08 if signals.high_notification_load else 0.0
    digest -= 0.18 * urgency + 0.25 * risk + 0.10 * repetition

    mute = 0.08 + 0.82 * risk + 0.44 * repetition
    mute += 0.20 * (1.0 - relevance) + 0.14 * (1.0 - engagement)
    mute += 0.16 if signals.muted_context and urgency < 0.55 else 0.0
    mute -= 0.16 * urgency + 0.08 * relevance

    if signals.message_type is MessageType.PROMOTION:
        digest += 0.12
        mute += 0.16 if engagement < 0.42 else 0.0
    if signals.message_type is MessageType.GREETING:
        digest += 0.12
    if signals.message_type in {MessageType.SPAM, MessageType.FORWARD}:
        mute += 0.20

    return {
        Action.NOTIFY: notify,
        Action.DIGEST: digest,
        Action.MUTE: mute,
    }


def _score_confidence(
    action: Action,
    scores: Mapping[Action, float],
    signals: RoutingSignals,
    context: MessageContext,
) -> float:
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    signal_strength = max(
        signals.urgency,
        signals.relevance,
        signals.repetition,
        signals.safety.risk_score,
    )
    confidence = 0.54 + 0.24 * math.tanh(max(0.0, margin)) + 0.14 * signal_strength
    if context.user is None:
        confidence -= 0.035
    if action is Action.DIGEST and signals.message_type is MessageType.UNKNOWN:
        confidence -= 0.04
    return round(_clamp(confidence, 0.51, 0.94), 3)


class RouterPolicy:
    """Offline policy that combines retrieval, safety, and personalization."""

    def __init__(
        self,
        data: DatasetBundle,
        *,
        retriever: HistoryRetriever | None = None,
        evidence_limit: int = 2,
    ):
        self.data = data
        self.retriever = retriever or HistoryRetriever(data)
        self.evidence_limit = max(0, min(int(evidence_limit), 10))

    def signals_for(
        self,
        message: IncomingMessage,
        *,
        context: MessageContext | None = None,
        evidence: Iterable[Evidence] | None = None,
        content_override: str | None = None,
    ) -> RoutingSignals:
        """Return structured signals for observability or policy integration."""

        message_context = context or build_context(message, self.data)
        if evidence is None:
            safe_evidence = self.retriever.search(
                message,
                limit=self.evidence_limit,
                content_override=content_override,
            )
        else:
            safe_evidence = ensure_same_user_evidence(message, evidence)
        return analyze_signals(
            message,
            message_context,
            safe_evidence,
            content_override=content_override,
        )

    def predict(
        self,
        message: IncomingMessage,
        *,
        context: MessageContext | None = None,
        evidence: Iterable[Evidence] | None = None,
        content_override: str | None = None,
    ) -> Prediction:
        """Route one message while preserving safety and evidence invariants."""

        message_context = context or build_context(message, self.data)
        signals = self.signals_for(
            message,
            context=message_context,
            evidence=evidence,
            content_override=content_override,
        )
        safety = signals.safety
        evidence_ids = _select_evidence(signals, limit=self.evidence_limit)

        # Safety is a hard boundary, not another preference feature.  This is
        # what prevents prior engagement or injected "trust" claims from
        # promoting credential theft into a notification.
        safety_block = (
            safety.risk_score >= 0.68
            or safety.asks_for_secret
            or safety.domain_mismatch
            or safety.unsafe_advice
            or safety.prompt_injection
            or safety.chain_message
            or (safety.suspicious_link and safety.risk_score >= 0.45)
            or (safety.impersonation and safety.coercive_urgency)
            or (
                signals.message_type is MessageType.SCAM
                and safety.risk_score >= 0.45
            )
        )
        if safety_block:
            if (
                safety.asks_for_secret
                or safety.suspicious_link
                or safety.domain_mismatch
                or safety.impersonation
            ):
                message_type = MessageType.SCAM
            elif safety.prompt_injection or safety.unsafe_advice:
                message_type = MessageType.SPAM
            elif safety.chain_message and signals.message_type in {
                MessageType.GREETING,
                MessageType.FORWARD,
            }:
                message_type = signals.message_type
            else:
                message_type = MessageType.SPAM
            confidence = _clamp(0.80 + 0.17 * max(safety.risk_score, 0.45), 0.82, 0.99)
            return Prediction(
                message_id=message.message_id,
                action=Action.MUTE,
                message_type=message_type,
                reason=_safety_reason(signals),
                confidence=round(confidence, 3),
                evidence_message_ids=evidence_ids,
                signals=signals,
            )

        if (
            signals.message_type is MessageType.PROMOTION
            and signals.promotions_opted_out
        ):
            return Prediction(
                message_id=message.message_id,
                action=Action.MUTE,
                message_type=MessageType.PROMOTION,
                reason=_normal_reason(Action.MUTE, signals),
                confidence=0.94,
                evidence_message_ids=evidence_ids,
                signals=signals,
            )

        if signals.repetition >= 0.68 and signals.engagement < 0.40:
            return Prediction(
                message_id=message.message_id,
                action=Action.MUTE,
                message_type=signals.message_type,
                reason=_normal_reason(Action.MUTE, signals),
                confidence=round(_clamp(0.72 + 0.22 * signals.repetition, 0.72, 0.93), 3),
                evidence_message_ids=evidence_ids,
                signals=signals,
            )

        scores = _action_scores(signals)
        action = max(
            (Action.NOTIFY, Action.DIGEST, Action.MUTE),
            key=lambda candidate: (scores[candidate], -list(Action).index(candidate)),
        )

        semantic_notes = set(signals.notes)

        # Promotions may be timely without being interrupt-worthy.  Explicit
        # opt-outs and negative history have already been handled above; all
        # remaining promotions are digest-or-mute, never notify.
        if signals.message_type is MessageType.PROMOTION and action is Action.NOTIFY:
            action = Action.DIGEST
        if (
            signals.message_type is MessageType.PROMOTION
            and action is Action.MUTE
            and not signals.muted_context
            and signals.repetition < 0.68
        ):
            action = Action.DIGEST
        if (
            signals.message_type is MessageType.SPAM
            or "poor_sender_reputation" in semantic_notes
        ):
            action = Action.MUTE

        elif "explicitly_nonurgent" in semantic_notes or "vague_plan" in semantic_notes:
            action = Action.DIGEST
        elif (
            "operational_update" in semantic_notes
            and action is Action.MUTE
            and signals.relevance >= 0.35
            and signals.engagement >= 0.45
        ):
            action = Action.DIGEST
        elif (
            "operational_alert" in semantic_notes
            and signals.urgency >= 0.50
            and (not signals.quiet_hours or signals.urgency >= 0.82)
        ):
            action = Action.NOTIFY
        elif (
            "school_action_notice" in semantic_notes
            and signals.relevance >= 0.42
            and not signals.quiet_hours
        ):
            action = Action.NOTIFY
        elif (
            "upcoming_event_context" in semantic_notes
            and signals.urgency >= 0.54
            and not signals.quiet_hours
        ):
            action = Action.NOTIFY
        elif (
            semantic_notes.intersection(
                {
                    "direct_personal_request",
                    "work_urgent",
                    "lost_item_deadline",
                    "immediate_health_call",
                }
            )
            and signals.relevance >= 0.42
            and (not signals.quiet_hours or signals.urgency >= 0.82)
        ):
            action = Action.NOTIFY
        elif (
            "trusted_payment_notice" in semantic_notes
            and signals.relevance >= 0.34
            and not signals.quiet_hours
        ):
            action = Action.NOTIFY

        # Direct, credible urgent requests can break through a muted group.  No
        # such exception exists above the safety block.
        if signals.message_type is not MessageType.PROMOTION and (
            signals.urgency >= 0.64
            and (signals.direct_mention or signals.relevance >= 0.54)
            and (not signals.quiet_hours or signals.urgency >= 0.82)
        ):
            action = Action.NOTIFY
        elif signals.quiet_hours and signals.urgency < 0.72 and action is Action.NOTIFY:
            action = Action.DIGEST

        # A mute score should win only with a concrete low-value signal; sparse
        # context defaults safely to digest rather than silently discarding.
        if action is Action.MUTE and not (
            signals.muted_context
            or signals.repetition >= 0.45
            or signals.message_type in {MessageType.SPAM, MessageType.FORWARD}
            or signals.relevance < 0.18
        ):
            action = Action.DIGEST

        confidence = _score_confidence(action, scores, signals, message_context)
        return Prediction(
            message_id=message.message_id,
            action=action,
            message_type=signals.message_type,
            reason=_normal_reason(action, signals),
            confidence=confidence,
            evidence_message_ids=evidence_ids,
            signals=signals,
        )

    # Familiar method name for orchestration pipelines.
    route = predict

    def route_all(
        self, *, content_by_message_id: Mapping[str, str] | None = None
    ) -> tuple[Prediction, ...]:
        """Route incoming messages in source order with deterministic output."""

        media_content = content_by_message_id or {}
        return tuple(
            self.predict(
                message,
                content_override=media_content.get(message.message_id),
            )
            for message in self.data.messages
        )


def route_message(
    message: IncomingMessage,
    data: DatasetBundle,
    *,
    content_override: str | None = None,
    context: MessageContext | None = None,
    evidence: Iterable[Evidence] | None = None,
) -> Prediction:
    """Convenience entry point for routing a single message."""

    return RouterPolicy(data).predict(
        message,
        content_override=content_override,
        context=context,
        evidence=evidence,
    )


def route_all(
    data: DatasetBundle, *, content_by_message_id: Mapping[str, str] | None = None
) -> tuple[Prediction, ...]:
    """Convenience entry point for routing an entire loaded dataset."""

    return RouterPolicy(data).route_all(content_by_message_id=content_by_message_id)
