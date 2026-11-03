"""Central configuration for UDA-Hub. Values can be overridden via env vars."""
import os
from pathlib import Path

STARTER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = STARTER_DIR / "data"

UDAHUB_DB_PATH = Path(os.getenv("UDAHUB_DB_PATH", DATA_DIR / "core" / "udahub.db"))
CULTPASS_DB_PATH = Path(os.getenv("CULTPASS_DB_PATH", DATA_DIR / "external" / "cultpass.db"))

DEFAULT_ACCOUNT_ID = os.getenv("UDAHUB_DEFAULT_ACCOUNT", "cultpass")
GUEST_EXTERNAL_USER_ID = "guest"

CHAT_MODEL = os.getenv("UDAHUB_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("UDAHUB_EMBEDDING_MODEL", "text-embedding-3-small")

# Decision thresholds
CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("UDAHUB_CLASSIFIER_MIN_CONFIDENCE", 0.5))
RESOLUTION_MIN_CONFIDENCE = float(os.getenv("UDAHUB_RESOLUTION_MIN_CONFIDENCE", 0.7))
# Minimum retrieval score for a KB article to count as relevant
RAG_MIN_SCORE_EMBEDDING = float(os.getenv("UDAHUB_RAG_MIN_SCORE", 0.30))
RAG_MIN_SCORE_KEYWORD = 0.10
RAG_TOP_K = 3

# How much context is fed to the agents
MAX_CONVERSATION_MESSAGES = 12
MAX_PRIOR_TICKETS = 3
MAX_MEMORIES = 10

# Structured event log: one JSON object per agent decision (JSON Lines)
EVENT_LOG_PATH = Path(os.getenv("UDAHUB_EVENT_LOG", STARTER_DIR / "logs" / "events.jsonl"))

# A guest proves they own an email with a one-time code before any account action
VERIFICATION_CODE_TTL_MINUTES = 10
VERIFICATION_MAX_ATTEMPTS = 3
# Stand-in for an email provider: each verification email is appended here (JSON Lines)
OUTBOX_PATH = Path(os.getenv("UDAHUB_OUTBOX", STARTER_DIR / "logs" / "outbox.jsonl"))

# Rate and cost limits (see agentic/rate_limits.py)
# Messages per ticket, per customer and per client IP within the window
RATE_LIMIT_MESSAGES = int(os.getenv("UDAHUB_RATE_LIMIT_MESSAGES", 6))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("UDAHUB_RATE_LIMIT_WINDOW_SECONDS", 60))
# Verification codes sent to one email per hour (counted whether or not the email has an account)
VERIFICATION_CODES_PER_EMAIL_PER_HOUR = 3
# LLM tokens (input + output) all tickets may use in 24 hours
DAILY_TOKEN_BUDGET = int(os.getenv("UDAHUB_DAILY_TOKEN_BUDGET", 2_000_000))
