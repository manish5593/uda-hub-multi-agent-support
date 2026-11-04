"""Optional second injection check: a small model trained to spot prompt injection.

The regex in agents/common.py knows this app's own fields but only the phrases written into it. The
model also catches reworded and non-English attacks, so the classifier flags a message when either
one does. It runs locally (no customer text leaves the machine) and only when `transformers` is
installed: pip install -r requirements-injection.txt. Set UDAHUB_INJECTION_MODEL="" to turn it off.
"""
import logging
from functools import lru_cache

from agentic import config

logger = logging.getLogger("udahub")


@lru_cache(maxsize=None)
def _detector(model_name: str):
    """The text-classification pipeline, loaded once, or None if the check is off or unavailable."""
    if not model_name:
        return None
    try:
        from transformers import pipeline
    except ImportError:
        logger.info("transformers is not installed: injection model check is off (the regex check still runs)")
        return None
    try:
        return pipeline("text-classification", model=model_name)
    except Exception:
        logger.exception("Could not load injection model %s: check is off", model_name)
        return None


def injection_score(text: str):
    """The model's confidence that `text` is a prompt injection (0-1), or None if the check didn't run."""
    detector = _detector(config.INJECTION_MODEL)
    if detector is None or not (text or "").strip():
        return None
    try:
        result = detector(text, truncation=True)[0]
    except Exception:
        # The regex check still runs, so a failure here must not stop the ticket
        logger.exception("Injection model check failed")
        return None
    return result["score"] if result["label"].upper() == "INJECTION" else 1 - result["score"]


def model_flags_injection(text: str) -> bool:
    score = injection_score(text)
    return score is not None and score >= config.INJECTION_MODEL_MIN_SCORE
