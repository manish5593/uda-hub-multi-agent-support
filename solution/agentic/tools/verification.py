"""One-time email codes: a guest proves they own an email before the agents act on its account.

Typing an email in the chat is not proof of identity, so the supervisor only links a guest to a
customer after they reply with the code sent to that customer's email. The pending verification is
kept in the graph state; only a salted hash of the code is stored, never the code itself.
"""
import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from agentic import config

CODE_PATTERN = re.compile(r"\b(\d{6})\b")


def _hash(code: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode()).hexdigest()


def send_code(email: str, code: str) -> None:
    """Deliver the code by email. Here it's appended to a local outbox standing in for an email provider."""
    path = config.OUTBOX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "to": email,
            "subject": "Your CultPass verification code",
            "body": f"Your CultPass verification code is {code}. It expires in "
                    f"{config.VERIFICATION_CODE_TTL_MINUTES} minutes. Never share it with anyone.",
        }) + "\n")


def start(email: str, profile: Optional[dict]) -> dict:
    """Start a verification for `email`. The code is only sent if the email belongs to a customer, but a
    pending verification is created either way, so the reply never reveals whether an account exists."""
    code = f"{secrets.randbelow(10 ** 6):06d}"
    salt = secrets.token_hex(8)
    if profile:
        send_code(email, code)
    expires = datetime.now(timezone.utc) + timedelta(minutes=config.VERIFICATION_CODE_TTL_MINUTES)
    return {
        "email": email.strip().lower(),
        "user_id": profile["user_id"] if profile else None,
        "user_name": profile["name"] if profile else None,
        "salt": salt,
        "code_hash": _hash(code, salt),
        "expires_at": expires.isoformat(),
        "attempts": 0,
    }


def code_in(text: str) -> Optional[str]:
    match = CODE_PATTERN.search(text or "")
    return match.group(1) if match else None


def check(pending: dict, code: str) -> tuple[str, Optional[dict]]:
    """Check a code against a pending verification.

    Returns (outcome, pending) where outcome is "verified", "wrong_code", "expired" or "locked".
    The pending verification is dropped (None) once it's used up, so a new email starts over.
    """
    if datetime.now(timezone.utc) > datetime.fromisoformat(pending["expires_at"]):
        return "expired", None
    if pending["user_id"] and hmac.compare_digest(_hash(code, pending["salt"]), pending["code_hash"]):
        return "verified", None
    attempts = pending["attempts"] + 1
    if attempts >= config.VERIFICATION_MAX_ATTEMPTS:
        return "locked", None
    return "wrong_code", {**pending, "attempts": attempts}
