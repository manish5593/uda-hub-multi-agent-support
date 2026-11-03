"""Rate and cost limits: stop floods of messages, repeated verification codes and runaway token spend.

Every counted event is one row in udahub.db's rate_limit_events table, so the limits hold across
restarts and processes. Keys are hashed because they can hold emails or IP addresses, and rows older
than two days are pruned.
"""
import hashlib
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from sqlalchemy import func

from agentic import config
from agentic.tools.db import session_scope, udahub_engine
from data.models import udahub

logger = logging.getLogger("udahub")

KEEP_FOR = timedelta(days=2)
TOKEN_WINDOW = timedelta(days=1)
CODE_WINDOW = timedelta(hours=1)
# Ticket metadata fields that name who is sending, besides the ticket itself
SENDER_FIELDS = ("external_user_id", "email", "client_ip")


def _key(kind: str, value) -> str:
    return hashlib.sha256(f"{kind}:{str(value).strip().lower()}".encode()).hexdigest()


@lru_cache(maxsize=None)
def _ensure_table(db_path: str) -> None:
    """Databases built before the limits existed don't have the table yet."""
    udahub.RateLimitEvent.__table__.create(udahub_engine(), checkfirst=True)


def _session():
    _ensure_table(str(config.UDAHUB_DB_PATH))
    return session_scope(udahub_engine())


def record(kind: str, value, amount: int = 1) -> None:
    now = datetime.now()
    E = udahub.RateLimitEvent
    with _session() as session:
        session.add(E(key=_key(kind, value), amount=amount, created_at=now))
        session.query(E).filter(E.created_at < now - KEEP_FOR).delete()


def used(kind: str, value, window: timedelta) -> int:
    E = udahub.RateLimitEvent
    with _session() as session:
        total = (session.query(func.coalesce(func.sum(E.amount), 0))
                 .filter(E.key == _key(kind, value), E.created_at >= datetime.now() - window)
                 .scalar())
    return int(total)


def check_message(ticket_id: str, metadata: Optional[dict] = None) -> Optional[tuple[str, str]]:
    """Count an incoming message against the limits, before any LLM call.

    Returns None if it may be handled, or (limit, reason) where limit is "too_many_messages" or
    "token_budget". A refused message isn't counted, so a customer who waits can write again.
    """
    metadata = metadata or {}
    senders = [("ticket", ticket_id)] + [(f, metadata[f]) for f in SENDER_FIELDS if metadata.get(f)]
    window = timedelta(seconds=config.RATE_LIMIT_WINDOW_SECONDS)
    for kind, value in senders:
        if used(f"message:{kind}", value, window) >= config.RATE_LIMIT_MESSAGES:
            return "too_many_messages", (f"more than {config.RATE_LIMIT_MESSAGES} messages in "
                                         f"{config.RATE_LIMIT_WINDOW_SECONDS}s for this {kind}")
    if tokens_used_today() >= config.DAILY_TOKEN_BUDGET:
        return "token_budget", f"the daily budget of {config.DAILY_TOKEN_BUDGET} LLM tokens is used up"
    for kind, value in senders:
        record(f"message:{kind}", value)
    return None


def can_send_code(email: str) -> bool:
    return used("code", email, CODE_WINDOW) < config.VERIFICATION_CODES_PER_EMAIL_PER_HOUR


def record_code(email: str) -> None:
    record("code", email)


def record_tokens(tokens: int) -> None:
    if tokens > 0:
        record("tokens", "all", tokens)


def tokens_used_today() -> int:
    return used("tokens", "all", TOKEN_WINDOW)


class TokenCounter(BaseCallbackHandler):
    """Adds the tokens of every LLM call to the daily budget."""

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            tokens = sum((getattr(getattr(g, "message", None), "usage_metadata", None) or {}).get("total_tokens", 0)
                         for gens in response.generations for g in gens)
            if not tokens:
                tokens = ((response.llm_output or {}).get("token_usage") or {}).get("total_tokens", 0)
            record_tokens(tokens)
        except Exception:
            # Counting must never break ticket handling
            logger.exception("Could not record LLM token usage")
