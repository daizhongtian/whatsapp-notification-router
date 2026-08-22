"""Deterministic content, safety, and personalization signal extraction.

All message and media-derived text is untrusted.  It is normalized, bounded,
and matched only against fixed local patterns; embedded instructions never
alter configuration, labels, thresholds, or control flow.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, time, timezone
from urllib.parse import urlsplit

from .models import (
    ConversationType,
    Evidence,
    IncomingMessage,
    MediaType,
    MessageContext,
    MessageType,
    RoutingSignals,
    SafetySignals,
)
from .retrieval import ensure_same_user_evidence


MAX_ANALYSIS_CHARACTERS = 100_000
_SPACE_RE = re.compile(r"\s+")


def _patterns(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value, flags=re.IGNORECASE | re.UNICODE) for value in values)


_PROMPT_INJECTION_PATTERNS = _patterns(
    r"\b(?:ignore|disregard|override|bypass)\b.{0,60}\b(?:instruction|prompt|policy|rule|router|system)\b",
    r"\b(?:system|assistant|router)\s+(?:instruction|message|note|metadata|override)\b",
    r"\b(?:set|use|force)\s+(?:the\s+)?(?:action|label|confidence|message[_ ]?type)\s*[=:]",
    r"\b(?:mark|classify)\b.{0,40}\b(?:notify|digest|mute|urgent|trusted)\b",
    r"\b(?:action|confidence|verified_business|user_priority)\s*=\s*[a-z0-9_.-]+",
)

_SECRET_REQUEST_PATTERNS = _patterns(
    r"\b(?:send(?:ing)?|shar(?:e|ing)|reply(?:\s+with)?|enter(?:ing)?|submit(?:ting)?|confirm(?:ing)?|tell(?:ing)?|provid(?:e|ing)|batao|daal\s+do|bhejo)\b[^.!?\n]{0,90}\b(?:otp|pin|password|passcode|cvv|verification\s+code|login\s+code|one[ -]?time\s+password|bank\s+details|card\s+details|account\s+number)\b",
    r"\b(?:otp|pin|password|passcode|cvv|verification\s+code|login\s+code|bank\s+details|card\s+details|account\s+number)\b[^.!?\n]{0,90}\b(?:send|share|reply|enter|submit|confirm|tell|provide|batao|daal\s+do|bhejo)\b",
    r"\bfill\b[^.!?\n]{0,60}\b(?:bank|card|wallet)\s+(?:detail|information)s?\b",
)

_COERCIVE_PATTERNS = _patterns(
    r"\b(?:act|pay|verify|confirm|open|click|respond|reply|send|share|scan)\s+(?:right\s+)?now\b",
    r"\b(?:immediately|final\s+(?:warning|reminder)|limited\s+window|last\s+chance)\b",
    r"\b(?:account|profile|wallet|access|service|card)\b.{0,70}\b(?:block(?:ed)?|suspend(?:ed)?|restrict(?:ed)?|close[sd]?|expire[sd]?|lock(?:ed)?)\b",
    r"\b(?:warna|jaldi|abhi)\b.{0,70}\b(?:block|band|hold|lock)\b",
)

_UNSAFE_ADVICE_PATTERNS = _patterns(
    r"\b(?:stop|skip|discontinue|quit)\b.{0,45}\b(?:medicine|medication|tablet|prescription|insulin|antibiotic)s?\b",
    r"\b(?:replace|instead\s+of)\b.{0,45}\b(?:doctor|treatment|medicine|medication|prescription)\b.{0,60}\b(?:herbal|home\s+remedy|miracle|secret)\b",
    r"\bdoctors?\s+(?:do\s+not|don't|dont)\s+(?:tell|want\s+you\s+to\s+know)\b",
)

_CHAIN_PATTERNS = _patterns(
    r"\b(?:forward|share|send)\b.{0,45}\b(?:to\s+)?(?:all|everyone|everybody|\d+|ten)\b",
    r"\b(?:do\s+not|don't|dont)\s+(?:ignore|break\s+the\s+chain)\b",
    r"\b(?:good\s+luck|blessing|blessings|positive\s+energy)\b.{0,70}\b(?:forward|share|send)\b",
    r"\b(?:fwd|forwarded)\s+(?:as\s+received|health\s+tip|message)\b",
)

_IMPERSONATION_PATTERNS = _patterns(
    r"\b(?:account|profile|wallet|bank|card|workspace|support|service|access)\b.{0,70}\b(?:security|verification|login|access|block(?:ed)?|restrict(?:ed)?|suspend(?:ed)?|expire[sd]?|stop[sp]?|deactivat(?:e|ed))\b",
    r"\b(?:security|support|verification)\s+(?:alert|team|desk|check|required)\b",
    r"\b(?:refund|reward|benefit|loan|payout)\b.{0,65}\b(?:release|claim|approved|pending|verification)\b",
)

_ADVANCE_FEE_PATTERNS = _patterns(
    r"\b(?:loan|reward|benefit|refund|prize|payout)\b.{0,80}\b(?:pay|fee|charge|deposit|token)\b",
    r"\b(?:pay|send)\b.{0,45}\b(?:processing|release|reactivation|clearance|reattempt)\s+(?:fee|charge|amount)\b",
    r"\b(?:processing|release|reactivation|reattempt)\s+(?:fee|charge)\b.{0,65}\b(?:release|claim|approve|deliver)\b",
)

_PAYMENT_REQUEST_PATTERNS = _patterns(
    r"\b(?:pay|payment|fee|charge|token|deposit|clearance\s+amount|processing\s+fee)\b",
    r"\b(?:scan|qr|upi|wallet|bank\s+details|card\s+details)\b",
)

_URGENCY_PATTERNS = _patterns(
    r"\b(?:urgent|urgently|immediately|asap|right\s+now|call\s+me\s+now)\b",
    r"\b(?:please\s+)?call\s+(?:me\s+)?now\b",
    r"\b(?:leav(?:e|es|ing)|closes?|locks?|expires?|ends?|due)\b.{0,35}\b(?:today|tonight|soon|in\s+\d+\s+min|before|at\s+\d)\b",
    r"\b(?:next|within)\s+\d+\s+(?:minute|minutes|min|mins|hour|hours)\b",
    r"\b(?:today|tonight|tomorrow)\b.{0,45}\b(?:deadline|close[sd]?|leav(?:e|es|ing)|appointment|meeting|pickup|payment|submit|confirm)\b",
    r"\b(?:moved|changed|failed|failing|outside|stuck|alert\s+threshold|rollback)\b",
    r"\b(?:jaldi|abhi|turant|aaj)\b",
)

_NON_URGENT_PATTERNS = _patterns(
    r"\b(?:no|nothing)\s+(?:urgent|blocking)\b",
    r"\bno\s+(?:rush|urgency|pressure|need\s+to\s+respond)\b",
    r"\b(?:not|isn't|isnt)\s+time[- ]sensitive\b",
    r"\bwhenever\s+(?:convenient|you\s+can|free)\b",
    r"\b(?:read|check|reply|call)\s+(?:it\s+)?later\b",
    r"\bwe\s+can\s+(?:talk|discuss)\s+tomorrow\b",
    r"\b(?:no\s+need|don't\s+need|do\s+not\s+need)\s+to\s+(?:reply|respond|act|call)\b",
)

_TIME_CONSTRAINT_PATTERNS = _patterns(
    r"\b(?:before|by|till|until|at|after)\s+(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight|today|tonight|tomorrow)\b",
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    r"\b(?:today|tonight|tomorrow|this\s+(?:morning|evening))\b",
    r"\b(?:in|within|next)\s+\d+\s+(?:minute|minutes|min|mins|hour|hours)\b",
    r"\b(?:deadline|time[- ]sensitive|closes?|leav(?:e|es|ing)|window)\b",
    r"\bbefore\s+(?:the\s+)?(?:scheduled\s+time|appointment|pickup|deadline)\b",
    r"\b\d{1,2}\s*(?:minute|minutes|min|mins)\s+me\b",
    r"\b(?:avant|before)\s+\d{1,2}(?::\d{2}|h\d{0,2})?\b",
)

_EVENT_PATTERNS = _patterns(
    r"\b(?:appointment|meeting|standup|sync|practice|workshop|class|trip|event|potluck|booking|reservation)\b",
    r"\b(?:schedule|scheduled|moved|venue|studio|gate|registration|consent|circular|form|portal)\b",
    r"\b(?:maintenance|repair|water|tanker|elevator|lift|fire\s+alarm|pickup)\b",
)

_PAYMENT_PATTERNS = _patterns(
    r"\b(?:payment|pay|paid|amount\s+due|statement|invoice|receipt|refund|charge|fee|upi|wallet|card)\b",
    r"(?:rs\.?|inr|usd|eur|\$|\u20b9)\s*\d",
)

_BUSINESS_UPDATE_PATTERNS = _patterns(
    r"\b(?:order|delivery|parcel|shipment|pickup|driver|ride|route|arrival|return|refund|booking)\b",
    r"\b(?:account|statement|appointment|prescription|claim|status|update|packed|scheduled|ready)\b",
    r"\b(?:official|registered)\s+app\b",
)

_PROMOTION_PATTERNS = _patterns(
    r"\b(?:sale|offer|discount|coupon|promo|cashback|deal|limited[- ]time|shop|buy|off)\b",
    r"\b(?:subscribe|unsubscribe|marketing|selected\s+products|saved\s+items|launch\s+price)\b",
    r"\b\d{1,3}\s*%\s*off\b",
    r"\b(?:itinerary|travel\s+package|per\s+person|first\s+order|launch\s+price)\b",
    r"\b(?:promotion|promotional|advertisement|marketing\s+offer)\b",
)

_MARKETPLACE_PATTERNS = _patterns(
    r"\b(?:selling|for\s+sale|buyer|price\s+final|asking\s+price)\b",
    r"\b(?:dm|message)\s+(?:me\s+)?if\s+(?:interested|serious)\b",
    r"\b(?:photos?|pics?|pictures?)\b.{0,80}\battached\b.{0,80}\bpickup\b",
    r"\b(?:barely|gently)\s+used\b",
)

_MARKETPLACE_CONTINUATION_PATTERNS = _patterns(
    r"\b(?:kept|reserved|set)\b.{0,35}\b(?:aside|for\s+you)\b.{0,120}\b(?:still\s+want|confirm|someone\s+else|other\s+(?:people|buyers?)|others?\s+(?:are\s+)?asking)\b",
    r"\b(?:still\s+want|confirm)\b.{0,100}\b(?:offered?|show|release)\b.{0,40}\b(?:someone|anyone|buyer)\s+else\b",
)

_PERSONAL_CONVERSATION_PATTERNS = _patterns(
    r"\b(?:can|could|will|would)\s+you\s+(?:call|come|join|confirm|tell|check|pick|collect)\b",
    r"\b(?:call|message|ping)\s+me\b",
    # Voice ASR can insert or substitute a short connector (for example,
    # "call when free" -> "call went free").  Keep the pattern bounded to
    # the same clause so casual callback requests remain personal without
    # turning unrelated uses of "free" into a routing signal.
    r"\b(?:call|message|ping)\b[^.!?\n]{0,40}\b(?:free|available|convenient)\b",
    r"\bjust\s+checking\s+(?:if|whether)\b",
    r"\banyone\s+(?:watching|joining|coming|interested)\b",
    r"\b(?:we|i)\s+(?:can|might|may|will)\s+(?:talk|discuss|start|meet|join)\b",
    r"\bcall\s+(?:me\s+)?when\s+(?:you(?:'re|\s+are)\s+)?(?:free|available|convenient)\b",
)

_IMMEDIATE_HEALTH_CALL_PATTERNS = _patterns(
    r"\b(?:please\s+)?call\s+(?:me\s+)?now\b.{0,120}\b(?:unwell|sick|clinic|hospital|doctor|medical)\b",
    r"\b(?:unwell|sick|clinic|hospital|doctor|medical)\b.{0,120}\b(?:please\s+)?call\s+(?:me\s+)?now\b",
)

_VAGUE_PLAN_PATTERNS = _patterns(
    r"\b(?:might|maybe|perhaps|shayad)\b.{0,80}\b(?:plan|meet|thread|poll|discuss|confirm|watch)\b",
    r"\b(?:after\s+dinner|later|raat\s+ko|baad\s+me)\b.{0,70}\b(?:thread|poll|plan|confirm|discuss)\b",
    r"\bno\s+pressure\b",
    r"\b(?:baad|raat)\b.{0,70}\b(?:plan|poll|discuss|dekh|confirm)\b",
)

_FIRST_CONTACT_PATTERNS = _patterns(
    r"\b(?:found|got)\s+(?:your|this)\s+(?:number|contact)\b",
    r"\bis\s+this\s+[a-z][\w'-]*\b",
    r"\bthis\s+is\s+[a-z][\w'-]*\s+from\b",
)

_LOST_ITEM_PATTERNS = _patterns(
    r"\b(?:found|left|lost)\b.{0,65}\b(?:passport|wallet|phone|keys?|bottle|bag|id\s+card|document)\b",
    r"\b(?:passport|wallet|phone|keys?|bottle|bag|id\s+card|document)\b.{0,80}\b(?:front\s+desk|reception|collect|retrieve|pick\s+up)\b",
    r"\b(?:passeport|portefeuille|telephone|téléphone|cles|clés|document)\b.{0,100}\b(?:trouve|trouvé|retrouver|recuperer|récupérer|reception|réception)\b",
    r"\b(?:trouve|trouvé)\b.{0,80}\b(?:passeport|portefeuille|telephone|téléphone|cles|clés|document)\b",
)

_WORK_URGENCY_PATTERNS = _patterns(
    r"\b(?:prod|production|build|deployment|rollback|queue|job|worker|incident|client|standup|dashboard)\b",
    r"\b(?:retry|retries|alert\s+threshold|escalation|edge\s+case|failed[- ]payment|refund\s+edge\s+case)\b",
)

_OPERATIONAL_ALERT_PATTERNS = _patterns(
    r"\b(?:tanker|water\s+supply|motor\s+room\s+valve)\b",
    r"\b(?:main\s+gate|access\s+gate|driveway|repair\s+truck)\b.{0,90}\b(?:close|closing|blocked|move|tow|entry)\b",
    r"\b(?:lift|elevator)\b.{0,80}\b(?:maintenance|repair|closed|stopped|service\s+lift)\b",
    r"\b(?:tank|tanker)\s+aa\s+gaya\b",
    r"\bgate\b.{0,90}\b(?:band|hata\s+do|repair\s+truck|shift)\b",
)

_NEAR_TERM_PATTERNS = _patterns(
    r"\b(?:now|immediately|right\s+away)\b",
    r"\b(?:in|within|next|maybe)\s+\d+\s*(?:minute|minutes|min|mins)\b",
    r"\b\d+\s*(?:minute|minutes|min|mins)\s+(?:max|early|left|remaining)\b",
    r"\b(?:leaving|closes?|closing)\b.{0,30}\b(?:soon|now|in\s+\d+|today)\b",
    r"\b\d+\s*(?:minute|minutes|min|mins)\s+me\b",
)

_SCHOOL_EVENT_PATTERNS = _patterns(
    r"\b(?:school|teacher|student|child|children|parents?)\b.{0,100}\b(?:bus|trip|consent|circular|timing|list)\b",
    r"\b(?:school\s+circular|field\s+trip|consent\s+note|bus\s+list)\b",
    r"\bschool\s+transport\b|\bschool\b.{0,90}\b(?:pickup|pick[- ]?up|gate|departure|route)\b",
)

_TRUSTED_PAYMENT_NOTICE_PATTERNS = _patterns(
    r"\b(?:payment|maintenance\s+(?:payment|fee)|amount\s+due|fee)\b.{0,100}\b(?:admin|office|official\s+app|society\s+app|receipts?\s+(?:will\s+be\s+)?(?:matched|reconciled))\b",
    r"\b(?:admin|office|official\s+app|society\s+app)\b.{0,100}\b(?:payment|paid|amount\s+due|fee|receipt)\b",
    r"\bmaintenance\b.{0,80}\b(?:closes?|due|deadline)\b.{0,100}\b(?:society\s+app|office\s+qr|receipts?)\b",
)

_BUSINESS_SCHEDULE_CHANGE_PATTERNS = _patterns(
    r"\b(?:pickup|pick[- ]?up|route|driver|arrival|booking|reservation)\b.{0,100}\b(?:moved|changed|rescheduled|instead|new\s+time)\b",
    r"\b(?:moved|changed|rescheduled|instead|new\s+time)\b.{0,100}\b(?:pickup|pick[- ]?up|route|driver|arrival|booking|reservation)\b",
)

_NEGATED_SECURITY_PAYMENT_PATTERNS = _patterns(
    r"\b(?:safety|security)\s+advisory\b",
    r"\b(?:never|do\s+not|don't)\s+ask\b.{0,70}\b(?:otp|pin|payment|card|bank)\b",
    r"\bno\s+(?:payment|otp|pin)\b.{0,40}\brequired\b",
)

_GREETING_PATTERNS = _patterns(
    r"\b(?:good\s+(?:morning|afternoon|evening|night)|happy\s+(?:birthday|anniversary)|stay\s+blessed)\b",
    r"\b(?:did\s+you\s+eat|had\s+dinner|take\s+care)\b",
)

_URL_RE = re.compile(
    r"(?<![@\w])(?:https?://|www\.)[^\s<>()\[\]{}]+"
    r"|(?<![@\w])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com|org|net|in|io|co|app|info|xyz|top|site|online|link|me|ly|gd|gl)"
    r"(?:/[^\s<>()\[\]{}]*)?",
    flags=re.IGNORECASE,
)

_SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "cutt.ly",
        "rb.gy",
    }
)

_RISKY_DOMAIN_WORDS = frozenset(
    {
        "account",
        "alert",
        "bank",
        "claim",
        "delivery",
        "help",
        "login",
        "pay",
        "refund",
        "secure",
        "support",
        "verify",
        "wallet",
    }
)


def normalize_text(text: str) -> str:
    """Bound and Unicode-normalize untrusted input for deterministic matching."""

    bounded = (text or "")[:MAX_ANALYSIS_CHARACTERS]
    normalized = unicodedata.normalize("NFKC", bounded).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cf", "Cs"}
    )
    return _SPACE_RE.sub(" ", normalized).strip()


def _has(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _count(patterns: tuple[re.Pattern[str], ...], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text) is not None)


def _normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().casefold().rstrip(".")
    if "://" not in candidate:
        candidate = "https://" + candidate
    try:
        host = urlsplit(candidate).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.casefold().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return None


def extract_domains(text: str) -> tuple[str, ...]:
    """Extract and canonicalize URL hosts without fetching any URL."""

    domains: list[str] = []
    for match in _URL_RE.finditer(text[:MAX_ANALYSIS_CHARACTERS]):
        raw = match.group(0).rstrip(".,;:!?')\"")
        domain = _normalize_domain(raw)
        if domain and domain not in domains:
            domains.append(domain)
    return tuple(domains)


def _domain_matches(domain: str, expected: str | None) -> bool:
    canonical = _normalize_domain(expected)
    return bool(canonical and (domain == canonical or domain.endswith("." + canonical)))


def _business_is_trusted(context: MessageContext, domains: tuple[str, ...]) -> bool:
    business = context.business
    if business is None or not business.verified or business.account_age_days < 30:
        return False
    official = _normalize_domain(business.official_domain)
    sender_domain = _normalize_domain(business.domain_used_by_sender)
    if official and sender_domain and not _domain_matches(sender_domain, official):
        return False
    if domains and official and any(not _domain_matches(domain, official) for domain in domains):
        return False
    report_rate = business.user_reports_30d / max(1, business.messages_sent_30d)
    return report_rate < 0.03


def detect_safety(
    text: str, context: MessageContext, *, forwarded_count: int | None = None
) -> SafetySignals:
    """Detect manipulation, credential theft, deceptive links, and unsafe advice."""

    normalized = normalize_text(text)
    prompt_injection = _has(_PROMPT_INJECTION_PATTERNS, normalized)
    asks_for_secret = _has(_SECRET_REQUEST_PATTERNS, normalized)
    coercive = _has(_COERCIVE_PATTERNS, normalized)
    unsafe_advice = _has(_UNSAFE_ADVICE_PATTERNS, normalized)
    chain_message = _has(_CHAIN_PATTERNS, normalized)
    payment_request = _has(_PAYMENT_REQUEST_PATTERNS, normalized)
    advance_fee = _has(_ADVANCE_FEE_PATTERNS, normalized)
    domains = extract_domains(normalized)

    business = context.business
    official_domain = business.official_domain if business else None
    sender_domain = business.domain_used_by_sender if business else None
    raw_domain_mismatch = False
    if business is not None:
        official = _normalize_domain(official_domain)
        # A sender may legitimately use a marketing/redirect provider.  Treat a
        # mismatch as dangerous only when that domain is actually present in
        # this message; metadata alone must not turn an ordinary promotion into
        # a scam.
        if official and any(not _domain_matches(domain, official) for domain in domains):
            raw_domain_mismatch = True

    trusted_business = _business_is_trusted(context, domains)
    suspicious_link = any(
        domain in _SHORTENER_DOMAINS
        or re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", domain) is not None
        or (
            any(word in domain.replace(".", "-").split("-") for word in _RISKY_DOMAIN_WORDS)
            and not (official_domain and _domain_matches(domain, official_domain))
        )
        for domain in domains
    )
    if domains and (asks_for_secret or (payment_request and coercive)) and not trusted_business:
        suspicious_link = True
    if (
        re.search(
            r"\b(?:link|scan\s+(?:(?:this|the)\s+)?qr|qr\s+code|scan(?:\s+\w+){0,3}\s+(?:and|to)\s+pay)\b",
            normalized,
        )
        and payment_request
        and (coercive or re.search(r"\bsend\s+(?:a\s+)?screenshot\b", normalized))
        and not trusted_business
    ):
        suspicious_link = True

    impersonation_language = _has(_IMPERSONATION_PATTERNS, normalized)
    # Cross-brand marketing and affiliate links are common. A visible domain
    # mismatch becomes a hard safety signal only when paired with a sensitive,
    # coercive, or payment action; an ordinary poster remains a promotion.
    domain_mismatch = raw_domain_mismatch and (
        asks_for_secret or payment_request or coercive or advance_fee or impersonation_language
    )
    impersonation = (impersonation_language or advance_fee) and (
        context.message.conversation_type is not ConversationType.BUSINESS
        or not trusted_business
        or domain_mismatch
    )

    reasons: list[str] = []
    risk = 0.0
    poor_business_reputation = bool(
        business
        and not business.verified
        and business.account_age_days < 90
        and business.user_reports_30d >= 10
        and (
            business.user_reports_30d / max(1, business.messages_sent_30d) >= 0.025
            or (
                business.user_reports_30d >= 15
                and business.domain_used_by_sender_age_days < 30
            )
        )
    )
    if prompt_injection:
        reasons.append("embedded_policy_instruction")
        risk += 0.30
    if asks_for_secret:
        reasons.append("credential_request")
        risk += 0.66
    if domain_mismatch:
        reasons.append("business_domain_mismatch")
        risk += 0.76
    if suspicious_link:
        reasons.append("suspicious_link")
        risk += 0.48
    if impersonation:
        reasons.append("account_impersonation")
        risk += 0.36
    if poor_business_reputation and impersonation_language:
        reasons.append("poor_sender_reputation")
        risk += 0.34
    if advance_fee:
        reasons.append("advance_fee_request")
        risk += 0.48
    if unsafe_advice:
        reasons.append("unsafe_advice")
        risk += 0.82
    if chain_message:
        reasons.append("chain_forward")
        risk += 0.38
    if coercive and (asks_for_secret or suspicious_link or payment_request or impersonation):
        reasons.append("coercive_urgency")
        risk += 0.22

    forwards = context.message.forwarded_count if forwarded_count is None else forwarded_count
    if forwards >= 5 and (chain_message or suspicious_link or impersonation):
        reasons.append("widely_forwarded_risk")
        risk += min(0.18, forwards * 0.015)

    # Verification is useful only as a weak exculpatory signal.  An actual
    # credential request or mismatched domain can never be trusted away.
    if trusted_business and not asks_for_secret and not domain_mismatch:
        risk -= 0.10

    return SafetySignals(
        risk_score=round(max(0.0, min(1.0, risk)), 6),
        prompt_injection=prompt_injection,
        asks_for_secret=asks_for_secret,
        suspicious_link=suspicious_link,
        domain_mismatch=domain_mismatch,
        impersonation=impersonation,
        coercive_urgency=coercive,
        unsafe_advice=unsafe_advice,
        chain_message=chain_message,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_ratio(numerator: int, denominator: int, default: float = 0.5) -> float:
    return numerator / denominator if denominator > 0 else default


def _same_or_before(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return True
    left_aware = left.tzinfo is not None
    right_aware = right.tzinfo is not None
    if left_aware:
        left = left.astimezone(timezone.utc).replace(tzinfo=None)
    if right_aware:
        right = right.astimezone(timezone.utc).replace(tzinfo=None)
    return left <= right


def _in_quiet_hours(created_at: datetime | None, window: str | None) -> bool:
    if created_at is None or not window:
        return False
    match = re.fullmatch(
        r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", window
    )
    if not match:
        return False
    start_hour, start_minute, end_hour, end_minute = map(int, match.groups())
    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
        return False
    if not (0 <= start_minute <= 59 and 0 <= end_minute <= 59):
        return False
    current = created_at.timetz().replace(tzinfo=None)
    start = time(start_hour, start_minute)
    end = time(end_hour, end_minute)
    if start == end:
        return True
    if start < end:
        return start <= current < end
    return current >= start or current < end


def _high_notification_load(context: MessageContext) -> bool:
    created = context.message.created_at
    target_day = created.date() if created else None
    eligible = tuple(
        summary
        for summary in context.daily_summaries
        if summary.day is not None and (target_day is None or summary.day <= target_day)
    )
    if not eligible:
        return False
    today = next(
        (summary for summary in reversed(eligible) if summary.day == target_day),
        eligible[-1],
    )
    previous = eligible[-8:-1] if len(eligible) > 1 else ()
    average = (
        sum(summary.notifications_sent for summary in previous) / len(previous)
        if previous
        else float(today.notifications_sent)
    )
    return today.notifications_sent >= max(10.0, average * 1.5)


def _history_features(evidence: tuple[Evidence, ...]) -> tuple[float, float, bool]:
    if not evidence:
        return 0.0, 0.5, False
    total_weight = sum(max(0.05, item.score) for item in evidence)
    positive = 0.0
    negative = 0.0
    exact_repeat = False
    for item in evidence:
        weight = max(0.05, item.score)
        exact_repeat = exact_repeat or item.score >= 0.90
        if item.event is None:
            continue
        if item.event.message_replied:
            positive += weight
        elif item.event.message_opened:
            positive += weight * 0.55
        if item.event.message_reported:
            negative += weight * 1.25
        elif item.event.muted_after_message:
            negative += weight
        elif item.event.notification_dismissed:
            negative += weight * 0.75
    engagement = _clamp(0.5 + 0.5 * positive / total_weight - 0.6 * negative / total_weight)
    repetition = max(
        0.72 if exact_repeat else 0.0,
        _clamp(0.25 * max(item.score for item in evidence) + 0.8 * negative / total_weight),
    )
    return repetition, engagement, negative > positive and negative > 0


def _personalization(
    context: MessageContext,
    evidence: tuple[Evidence, ...],
    *,
    direct_mention: bool,
) -> tuple[float, float, bool, bool, tuple[str, ...]]:
    message = context.message
    relevance = 0.28
    engagement = 0.5
    trusted = False
    muted = False
    notes: list[str] = []

    user = context.user
    if user is not None:
        interactions = (
            user.messages_opened_30d
            + user.notifications_dismissed_30d
            + user.messages_reported_30d
        )
        engagement = _safe_ratio(user.messages_opened_30d, interactions, 0.5)
        engagement -= min(0.25, user.messages_reported_30d * 0.025)

    if message.conversation_type is ConversationType.PERSONAL:
        relevance += 0.40
    elif message.conversation_type is ConversationType.GROUP:
        membership = context.group_membership
        relevance += 0.08
        if membership is not None:
            muted = membership.group_muted_by_user
            group_actions = (
                membership.messages_read_30d
                + membership.notifications_dismissed_30d
            )
            group_engagement = _safe_ratio(
                membership.messages_read_30d, group_actions, engagement
            )
            reply_bonus = min(0.18, membership.replies_sent_30d * 0.025)
            engagement = 0.45 * engagement + 0.55 * group_engagement + reply_bonus
            relevance += min(0.24, membership.messages_read_30d * 0.008)
            relevance += min(0.12, membership.replies_sent_30d * 0.02)
            if membership.role.casefold() == "admin":
                relevance += 0.08
            if muted:
                relevance -= 0.22
                notes.append("recipient_muted_group")
    elif message.conversation_type is ConversationType.BUSINESS:
        relevance += 0.04
        relationship = context.business_history
        if relationship is not None:
            relevance += min(0.32, relationship.activity_count_180d * 0.035)
            interactions = (
                relationship.messages_opened_30d
                + relationship.messages_dismissed_30d
            )
            relationship_engagement = _safe_ratio(
                relationship.messages_opened_30d, interactions, engagement
            )
            engagement = 0.35 * engagement + 0.65 * relationship_engagement
            if relationship.messages_replied_30d:
                relevance += min(0.12, relationship.messages_replied_30d * 0.04)
        trusted = _business_is_trusted(context, ())
        if trusted:
            relevance += 0.10

    if direct_mention:
        relevance += 0.28
        notes.append("direct_mention")

    repetition, history_engagement, history_negative = _history_features(evidence)
    if evidence:
        engagement = 0.65 * engagement + 0.35 * history_engagement
        if history_engagement >= 0.7:
            relevance += 0.10
            if message.conversation_type is ConversationType.PERSONAL:
                trusted = True
        if history_negative:
            relevance -= 0.12
            notes.append("similar_history_was_rejected")

    return (
        _clamp(relevance),
        _clamp(engagement),
        trusted,
        muted,
        tuple(notes),
    )


def _category_scores(
    text: str,
    message: IncomingMessage,
    context: MessageContext,
    safety: SafetySignals,
    *,
    direct_mention: bool,
) -> tuple[dict[MessageType, float], float, bool, tuple[str, ...]]:
    scores = {message_type: 0.0 for message_type in MessageType}
    scores[MessageType.UNKNOWN] = 0.14
    semantic_notes: list[str] = []

    if message.conversation_type is ConversationType.PERSONAL:
        scores[MessageType.PERSONAL] += 0.58
    elif message.conversation_type is ConversationType.BUSINESS:
        scores[MessageType.BUSINESS_UPDATE] += 0.20

    event_hits = _count(_EVENT_PATTERNS, text)
    payment_hits = _count(_PAYMENT_PATTERNS, text)
    business_hits = _count(_BUSINESS_UPDATE_PATTERNS, text)
    promotion_hits = _count(_PROMOTION_PATTERNS, text)
    greeting_hits = _count(_GREETING_PATTERNS, text)
    scores[MessageType.EVENT] += min(0.80, event_hits * 0.30)
    scores[MessageType.PAYMENT] += min(0.85, payment_hits * 0.34)
    scores[MessageType.BUSINESS_UPDATE] += min(0.75, business_hits * 0.24)
    scores[MessageType.PROMOTION] += min(0.90, promotion_hits * 0.34)
    scores[MessageType.GREETING] += min(0.82, greeting_hits * 0.43)

    group_type = context.group.group_type.casefold() if context.group else ""
    marketplace = _has(_MARKETPLACE_PATTERNS, text) or (
        "marketplace" in group_type
        and _has(_MARKETPLACE_CONTINUATION_PATTERNS, text)
    )
    personal_conversation = _has(_PERSONAL_CONVERSATION_PATTERNS, text)
    vague_plan = _has(_VAGUE_PLAN_PATTERNS, text)
    first_contact = _has(_FIRST_CONTACT_PATTERNS, text)
    lost_item = _has(_LOST_ITEM_PATTERNS, text)
    work_urgent = _has(_WORK_URGENCY_PATTERNS, text)
    operational_alert = _has(_OPERATIONAL_ALERT_PATTERNS, text)
    near_term = _has(_NEAR_TERM_PATTERNS, text)
    school_event = _has(_SCHOOL_EVENT_PATTERNS, text)
    trusted_payment_notice = _has(_TRUSTED_PAYMENT_NOTICE_PATTERNS, text)
    negated_security_payment = _has(_NEGATED_SECURITY_PAYMENT_PATTERNS, text)
    immediate_health_call = _has(_IMMEDIATE_HEALTH_CALL_PATTERNS, text)
    business_schedule_change = bool(
        message.conversation_type is ConversationType.BUSINESS
        and _has(_BUSINESS_SCHEDULE_CHANGE_PATTERNS, text)
    )

    relationship_context = normalize_text(
        context.business_history.why_user_knows_account
        if context.business_history is not None
        else ""
    ).replace("_", " ")
    upcoming_event_context = bool(
        re.search(
            r"\b(?:appointment|booking|reservation|event|clinic|trip)\b",
            relationship_context,
        )
    )

    if marketplace:
        scores[MessageType.PROMOTION] += 0.95
        scores[MessageType.EVENT] *= 0.45
        semantic_notes.append("marketplace_promotion")
    if personal_conversation:
        scores[MessageType.PERSONAL] += 0.55
    if direct_mention and personal_conversation and not work_urgent:
        scores[MessageType.PERSONAL] += 0.30
        semantic_notes.append("direct_personal_request")
    if vague_plan:
        scores[MessageType.PERSONAL] = max(scores[MessageType.PERSONAL], 0.82)
        semantic_notes.append("vague_plan")
    if first_contact and not lost_item:
        scores[MessageType.UNKNOWN] += 0.68
        semantic_notes.append("unknown_first_contact")
    if lost_item:
        scores[MessageType.PERSONAL] += 0.62
        semantic_notes.append("lost_item")
    if work_urgent:
        scores[MessageType.PAYMENT] *= 0.12
        scores[MessageType.URGENT] += 0.30
        semantic_notes.append("work_urgent")
    if immediate_health_call:
        scores[MessageType.URGENT] += 0.72
        semantic_notes.append("immediate_health_call")
    if negated_security_payment:
        scores[MessageType.PAYMENT] *= 0.08
        scores[MessageType.BUSINESS_UPDATE] += 0.52
        semantic_notes.append("security_advisory")
    if trusted_payment_notice and safety.risk_score < 0.45:
        scores[MessageType.PAYMENT] = max(
            0.90, scores[MessageType.PAYMENT] + 0.66
        )
        semantic_notes.append("trusted_payment_notice")
    if school_event:
        scores[MessageType.EVENT] += 0.95
        semantic_notes.append("school_action_notice")
    if operational_alert:
        scores[MessageType.EVENT] += 0.25
        semantic_notes.append("operational_update")
    if upcoming_event_context:
        scores[MessageType.EVENT] += 0.46
        semantic_notes.append("upcoming_event_context")
    if business_schedule_change:
        scores[MessageType.BUSINESS_UPDATE] += 0.78
        semantic_notes.append("business_schedule_change")

    business = context.business
    relationship = context.business_history
    if (
        business is not None
        and not business.verified
        and business.account_age_days < 90
        and business.user_reports_30d >= 10
        and relationship is not None
        and relationship.messages_dismissed_30d
        >= max(3, relationship.messages_opened_30d * 2)
    ):
        scores[MessageType.SPAM] += 0.84
        semantic_notes.append("poor_sender_reputation")

    if (
        not text
        and message.media_type is MediaType.VOICE
        and context.group is not None
        and "family" in context.group.group_type.casefold()
    ):
        scores[MessageType.PERSONAL] += 0.46
        semantic_notes.append("family_voice_context")

    if message.forwarded_count > 0 or re.search(r"\b(?:fwd|forwarded|forwarding)\b", text):
        scores[MessageType.FORWARD] += min(
            0.92, 0.34 + message.forwarded_count * 0.055
        )
    if safety.chain_message:
        scores[MessageType.FORWARD] += 0.42
        scores[MessageType.SPAM] += 0.66
    if safety.prompt_injection:
        scores[MessageType.SPAM] += 0.72
    if safety.unsafe_advice:
        scores[MessageType.SPAM] += 0.72
    if safety.risk_score >= 0.45:
        scores[MessageType.SCAM] += safety.risk_score

    urgency = 0.08 if text else 0.0
    urgency += min(0.46, _count(_URGENCY_PATTERNS, text) * 0.20)
    explicit_time = _has(_TIME_CONSTRAINT_PATTERNS, text)
    if explicit_time:
        urgency += 0.24
    if direct_mention:
        urgency += 0.14
    if message.conversation_type is ConversationType.PERSONAL:
        urgency += 0.04
    if operational_alert and near_term:
        urgency += 0.44
        scores[MessageType.URGENT] += 0.28
        semantic_notes.append("operational_alert")
    if school_event:
        urgency += 0.34
        if explicit_time:
            urgency += 0.20
    if upcoming_event_context and explicit_time:
        urgency += 0.32
    if work_urgent:
        urgency += 0.32
    if immediate_health_call:
        urgency += 0.42
    if lost_item and (explicit_time or near_term):
        urgency += 0.34
        semantic_notes.append("lost_item_deadline")
    if trusted_payment_notice and explicit_time:
        urgency += 0.28
    if _has(_NON_URGENT_PATTERNS, text):
        urgency -= 0.58
        semantic_notes.append("explicitly_nonurgent")
    if vague_plan:
        urgency = min(urgency, 0.16)
    urgency = _clamp(urgency)
    if scores[MessageType.PROMOTION] >= 0.55 and safety.risk_score < 0.45:
        urgency = min(urgency, 0.42)
        semantic_notes.append("promotional_urgency")
    scores[MessageType.URGENT] += urgency
    if trusted_payment_notice:
        scores[MessageType.URGENT] = min(scores[MessageType.URGENT], 0.86)
    if business_schedule_change:
        scores[MessageType.URGENT] = min(scores[MessageType.URGENT], 0.94)
        scores[MessageType.EVENT] = min(scores[MessageType.EVENT], 0.96)
    if school_event:
        scores[MessageType.URGENT] = min(scores[MessageType.URGENT], 0.94)

    return (
        {key: round(_clamp(value), 6) for key, value in scores.items()},
        urgency,
        explicit_time,
        tuple(dict.fromkeys(semantic_notes)),
    )


def classify_message_type(
    category_scores: dict[MessageType, float], safety: SafetySignals
) -> MessageType:
    """Select a category with safety semantics taking precedence over urgency."""

    if safety.risk_score >= 0.68 and (
        safety.asks_for_secret
        or safety.suspicious_link
        or safety.domain_mismatch
        or safety.impersonation
    ):
        return MessageType.SCAM
    if safety.prompt_injection or safety.unsafe_advice:
        if category_scores[MessageType.SPAM] >= 0.65:
            return MessageType.SPAM
    if safety.chain_message:
        if category_scores[MessageType.GREETING] >= 0.40:
            return MessageType.GREETING
        if category_scores[MessageType.FORWARD] >= 0.60:
            return MessageType.FORWARD
        if category_scores[MessageType.SPAM] >= 0.65:
            return MessageType.SPAM
    priority = {
        MessageType.URGENT: 10,
        MessageType.PAYMENT: 9,
        MessageType.EVENT: 8,
        MessageType.BUSINESS_UPDATE: 7,
        MessageType.PROMOTION: 6,
        MessageType.PERSONAL: 5,
        MessageType.GREETING: 4,
        MessageType.FORWARD: 3,
        MessageType.SPAM: 2,
        MessageType.SCAM: 1,
        MessageType.UNKNOWN: 0,
    }
    return max(
        category_scores,
        key=lambda message_type: (category_scores[message_type], priority[message_type]),
    )


def analyze_signals(
    message: IncomingMessage,
    context: MessageContext,
    evidence: tuple[Evidence, ...] = (),
    *,
    content_override: str | None = None,
) -> RoutingSignals:
    """Build structured routing signals from message, context, and safe evidence."""

    if context.message.message_id != message.message_id:
        raise ValueError("message context does not match the message being analyzed")
    safe_evidence = ensure_same_user_evidence(message, evidence)
    combined = message.message_text
    if content_override:
        combined = f"{combined}\n{content_override}" if combined else content_override
    text = normalize_text(combined)

    user_id_pattern = re.escape(message.user_id.casefold())
    direct_mention = bool(
        re.search(rf"(?<![\w@])@{user_id_pattern}(?!\w)", text, flags=re.IGNORECASE)
    )
    safety = detect_safety(text, context)
    category_scores, urgency, explicit_time, semantic_notes = _category_scores(
        text, message, context, safety, direct_mention=direct_mention
    )
    message_type = classify_message_type(category_scores, safety)
    relevance, engagement, trusted, muted, notes = _personalization(
        context, safe_evidence, direct_mention=direct_mention
    )
    repetition, _history_engagement, _history_negative = _history_features(safe_evidence)

    relationship = context.business_history
    promotions_opted_out = bool(
        relationship
        and (
            (
                relationship.promotions_opted_out_at is not None
                and _same_or_before(
                    relationship.promotions_opted_out_at, message.created_at
                )
            )
            or not relationship.allows_promotions
        )
    )
    quiet_hours = _in_quiet_hours(
        message.created_at,
        context.user.do_not_disturb_window if context.user else None,
    )
    high_load = _high_notification_load(context)

    note_values = [*notes, *semantic_notes]
    if quiet_hours:
        note_values.append("quiet_hours")
    if promotions_opted_out:
        note_values.append("promotions_not_allowed")
    if high_load:
        note_values.append("high_notification_load")
    if content_override and not message.message_text:
        note_values.append("media_content_used")

    return RoutingSignals(
        message_type=message_type,
        category_scores=category_scores,
        urgency=round(urgency, 6),
        relevance=round(relevance, 6),
        engagement=round(engagement, 6),
        repetition=round(repetition, 6),
        quiet_hours=quiet_hours,
        direct_mention=direct_mention,
        explicit_time_constraint=explicit_time,
        trusted_context=trusted,
        muted_context=muted,
        promotions_opted_out=promotions_opted_out,
        high_notification_load=high_load,
        safety=safety,
        evidence=safe_evidence,
        notes=tuple(dict.fromkeys(note_values)),
    )


# Compact spelling for integrations that expose feature extraction as a stage.
extract_signals = analyze_signals
