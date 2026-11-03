"""Tools that read and act on the CultPass (external) database.

Every action is scoped to a single customer: `build_account_tools` binds the
customer id into the tools so the LLM can never act on someone else's account.
"""
import functools
import logging
import uuid
from datetime import datetime
from typing import Literal, Optional

from langchain_core.tools import StructuredTool

from agentic.tools.db import cultpass_engine, session_scope
from agentic.tools import ticket_tools
from data.models import cultpass

logger = logging.getLogger(__name__)


def _fmt_dt(value: Optional[datetime]) -> Optional[str]:
    return value.strftime("%Y-%m-%d %H:%M") if value else None


def _subscription_dict(sub: Optional[cultpass.Subscription]) -> Optional[dict]:
    if sub is None:
        return None
    return {
        "subscription_id": sub.subscription_id,
        "status": sub.status,
        "tier": sub.tier,
        "monthly_quota": sub.monthly_quota,
        "started_at": _fmt_dt(sub.started_at),
        "ended_at": _fmt_dt(sub.ended_at),
    }


def _used_this_month(session, user_id: str) -> int:
    start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return (
        session.query(cultpass.Reservation)
        .filter(
            cultpass.Reservation.user_id == user_id,
            cultpass.Reservation.status == "reserved",
            cultpass.Reservation.created_at >= start,
        )
        .count()
    )


# ---------------------------------------------------------------------------
# Plain functions (unit-testable, no LLM involved)
# ---------------------------------------------------------------------------

def get_customer_profile(user_id: str) -> Optional[dict]:
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        if user is None:
            return None
        profile = {
            "user_id": user.user_id,
            "name": user.full_name,
            "email": user.email,
            "is_blocked": bool(user.is_blocked),
            "subscription": _subscription_dict(user.subscription),
        }
        if user.subscription:
            used = _used_this_month(session, user.user_id)
            profile["subscription"]["used_this_month"] = used
            profile["subscription"]["remaining_this_month"] = max(
                user.subscription.monthly_quota - used, 0
            )
        return profile


def find_customer_by_email(email: str) -> Optional[dict]:
    with session_scope(cultpass_engine()) as session:
        user = (
            session.query(cultpass.User)
            .filter(cultpass.User.email.ilike(email.strip()))
            .first()
        )
        user_id = user.user_id if user else None
    return get_customer_profile(user_id) if user_id else None


def list_reservations(user_id: str) -> list[dict]:
    with session_scope(cultpass_engine()) as session:
        rows = (
            session.query(cultpass.Reservation)
            .filter(cultpass.Reservation.user_id == user_id)
            .all()
        )
        return [
            {
                "reservation_id": r.reservation_id,
                "status": r.status,
                "experience_id": r.experience_id,
                "experience": r.experience.title,
                "location": r.experience.location,
                "when": _fmt_dt(r.experience.when),
                "is_premium": bool(r.experience.is_premium),
            }
            for r in rows
        ]


def experience_titles() -> list[str]:
    """Titles of every experience in the catalog, past and upcoming."""
    with session_scope(cultpass_engine()) as session:
        return [title for (title,) in session.query(cultpass.Experience.title).all()]


def search_experiences(query: str = "", location: str = "", only_available: bool = True) -> list[dict]:
    with session_scope(cultpass_engine()) as session:
        q = session.query(cultpass.Experience).filter(cultpass.Experience.when >= datetime.now())
        if location:
            q = q.filter(cultpass.Experience.location.ilike(f"%{location}%"))
        if only_available:
            q = q.filter(cultpass.Experience.slots_available > 0)
        results = []
        terms = [t for t in query.lower().split() if len(t) > 2]
        for exp in q.order_by(cultpass.Experience.when).all():
            text = f"{exp.title} {exp.description} {exp.location}".lower()
            if terms and not any(t in text for t in terms):
                continue
            results.append(
                {
                    "experience_id": exp.experience_id,
                    "title": exp.title,
                    "description": exp.description,
                    "location": exp.location,
                    "when": _fmt_dt(exp.when),
                    "slots_available": exp.slots_available,
                    "is_premium": bool(exp.is_premium),
                }
            )
        return results


def reserve_experience(user_id: str, experience_id: str) -> dict:
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        exp = session.get(cultpass.Experience, experience_id)
        if user is None or exp is None:
            return {"success": False, "error": "Unknown customer or experience id."}
        if user.is_blocked:
            return {"success": False, "error": "Account is blocked; reservations are disabled."}
        sub = user.subscription
        if sub is None or sub.status != "active":
            return {"success": False, "error": f"Subscription is not active (status: {sub.status if sub else 'none'})."}
        if exp.is_premium and sub.tier != "premium":
            return {"success": False, "error": "This is a premium experience and the customer is on the basic tier."}
        if exp.slots_available <= 0:
            return {"success": False, "error": "No slots available for this experience."}
        if _used_this_month(session, user_id) >= sub.monthly_quota:
            return {"success": False, "error": "Monthly quota already used."}
        already = (
            session.query(cultpass.Reservation)
            .filter_by(user_id=user_id, experience_id=experience_id, status="reserved")
            .first()
        )
        if already:
            return {"success": False, "error": f"Already reserved (reservation {already.reservation_id})."}
        reservation = cultpass.Reservation(
            reservation_id=str(uuid.uuid4())[:6],
            user_id=user_id,
            experience_id=experience_id,
            status="reserved",
        )
        exp.slots_available -= 1
        session.add(reservation)
        return {
            "success": True,
            "reservation_id": reservation.reservation_id,
            "experience": exp.title,
            "when": _fmt_dt(exp.when),
        }


def cancel_reservation(user_id: str, reservation_id: str) -> dict:
    with session_scope(cultpass_engine()) as session:
        r = session.get(cultpass.Reservation, reservation_id)
        if r is None or r.user_id != user_id:
            return {"success": False, "error": "Reservation not found for this customer."}
        if r.status != "reserved":
            return {"success": False, "error": f"Reservation is already {r.status}."}
        hours_left = (r.experience.when - datetime.now()).total_seconds() / 3600
        r.status = "cancelled"
        r.experience.slots_available += 1
        return {
            "success": True,
            "reservation_id": reservation_id,
            "experience": r.experience.title,
            "credit_returned": hours_left >= 24,
        }


def manage_subscription(user_id: str, action: str) -> dict:
    transitions = {
        "pause": ({"active"}, "paused"),
        "resume": ({"paused"}, "active"),
        "cancel": ({"active", "paused"}, "cancelled"),
    }
    if action not in transitions:
        return {"success": False, "error": f"Unknown action '{action}'."}
    allowed_from, new_status = transitions[action]
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        sub = user.subscription if user else None
        if sub is None:
            return {"success": False, "error": "Customer has no subscription."}
        if user.is_blocked:
            return {"success": False, "error": "Account is blocked; changes require human review."}
        if sub.status not in allowed_from:
            return {"success": False, "error": f"Cannot {action} a subscription that is '{sub.status}'."}
        old = sub.status
        sub.status = new_status
        if new_status == "cancelled":
            sub.ended_at = datetime.now()
        return {"success": True, "previous_status": old, "new_status": new_status}


# ---------------------------------------------------------------------------
# LangChain tools bound to one customer + ticket
# ---------------------------------------------------------------------------

def build_account_tools(user_id: str, ticket_id: str, confirmed_action: Optional[dict] = None) -> list[StructuredTool]:
    """Return tools pre-bound to `user_id`, so the agent can only touch this account.

    The booking and cancellation tools only accept ids that a lookup tool returned earlier in
    this turn, so the agent can't act on an id it guessed.

    Cancelling a reservation and changing the subscription need the customer's confirmation: the
    first call only returns `needs_confirmation` with the exact action. The change runs only when
    `confirmed_action` is that same action, which code sets after the customer replied yes; the
    LLM can't confirm on the customer's behalf.
    """
    # ids returned by lookups this turn, with a label used to point the agent at the right id
    seen_reservation_ids: dict[str, str] = {}
    seen_experience_ids: dict[str, str] = {}

    def _unknown_id(kind: str, given: str, seen: dict, lookup: str) -> dict:
        if not seen:
            return {"success": False, "error": f"Unknown {kind} id '{given}'. Call {lookup} first and use an id it returns."}
        valid = "; ".join(f"{i} ({label})" for i, label in seen.items())
        return {"success": False, "error": f"Unknown {kind} id '{given}'. Use the {kind}_id field from {lookup}, "
                f"not a position or a name. Valid ids: {valid}"}

    def _ask_to_confirm(action: dict, change: str) -> dict:
        return {"success": False, "needs_confirmation": True, "action": action,
                "error": f"Not done yet: this needs the customer's confirmation. Ask them to reply YES to {change}, "
                         "and don't call this tool again until they have."}

    def get_my_account() -> dict:
        """Get the customer's profile: blocked status, subscription status, tier and remaining monthly quota."""
        return get_customer_profile(user_id) or {"error": "Customer not found."}

    def get_my_reservations() -> list[dict]:
        """List the customer's reservations (id, status, experience, date)."""
        reservations = list_reservations(user_id)
        for r in reservations:
            seen_reservation_ids[r["reservation_id"]] = f"{r['experience']}, {r['when']}, {r['status']}"
            seen_experience_ids[r["experience_id"]] = r["experience"]
        return reservations

    def find_experiences(query: str = "", location: str = "") -> list[dict]:
        """Search upcoming experiences with free slots. `query` matches literal words in the title or description, and `location` matches the city or state; leave both empty to list the whole catalog. Each result says whether the customer has `already_booked` it."""
        experiences = search_experiences(query=query, location=location)
        booked = {r["experience_id"] for r in list_reservations(user_id) if r["status"] == "reserved"}
        for exp in experiences:
            exp["already_booked"] = exp["experience_id"] in booked
        for e in experiences:
            seen_experience_ids[e["experience_id"]] = f"{e['title']}, {e['when']}"
        return experiences

    def book_experience(experience_id: str) -> dict:
        """Reserve an experience for the customer. Only call after the customer clearly asked to book that specific experience. The id must come from find_experiences."""
        if experience_id not in seen_experience_ids:
            return _unknown_id("experience", experience_id, seen_experience_ids, "find_experiences")
        result = reserve_experience(user_id, experience_id)
        if result.get("success"):
            seen_reservation_ids[result["reservation_id"]] = f"{result['experience']}, {result['when']}, reserved"
        return result

    def cancel_my_reservation(reservation_id: str) -> dict:
        """Cancel one of the customer's reservations. The id must come from get_my_reservations. The first call asks the customer to confirm; after they reply yes, call it again with the same id."""
        requested = {"tool": "cancel_my_reservation", "reservation_id": reservation_id}
        if requested == confirmed_action:
            return cancel_reservation(user_id, reservation_id)
        if reservation_id not in seen_reservation_ids:
            return _unknown_id("reservation", reservation_id, seen_reservation_ids, "get_my_reservations")
        return _ask_to_confirm(requested, f"cancel {seen_reservation_ids[reservation_id]}")

    def change_subscription(action: Literal["pause", "resume", "cancel"]) -> dict:
        """Pause, resume or cancel the customer's subscription. Only call when the customer explicitly asked for this action. The first call asks the customer to confirm; after they reply yes, call it again with the same action."""
        requested = {"tool": "change_subscription", "action": action}
        if requested == confirmed_action:
            return manage_subscription(user_id, action)
        return _ask_to_confirm(requested, f"{action} their subscription")

    def request_refund(reason: str) -> dict:
        """Register a refund request for support-lead approval. This does NOT issue a refund; the ticket will be escalated. `reason` must describe the problem the customer reported, in their own terms."""
        reason = reason.strip()
        if len(reason) < 5:
            return {"success": False, "error": "A refund request needs a reason describing the problem."}
        ref = f"RF-{uuid.uuid4().hex[:6].upper()}"
        ticket_tools.add_message(
            ticket_id, "system", f"Refund request {ref} pending support-lead approval. Reason: {reason}"
        )
        return {
            "success": True,
            "refund_request_id": ref,
            "status": "pending_support_lead_approval",
            "note": "Refunds require support-lead approval; do not promise an amount or timeline.",
        }

    funcs = [
        get_my_account,
        get_my_reservations,
        find_experiences,
        book_experience,
        cancel_my_reservation,
        change_subscription,
        request_refund,
    ]
    return [StructuredTool.from_function(_safe(f)) for f in funcs]


def _safe(func):
    """Turn unexpected errors (e.g. the database being unavailable) into a structured error result.

    LangGraph's ToolNode only catches invalid-argument errors; anything else would crash the whole
    graph run and leave the customer without a reply. Business-rule failures are already returned
    as data by the functions above, so this only fires on real faults.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Tool %s failed", func.__name__)
            return {"success": False, "error": "Internal error while contacting the CultPass system. "
                    "Don't retry; tell the customer the team will follow up."}
    return wrapper


# Tools whose successful use means policy requires a human to follow up
ESCALATING_TOOLS = {"request_refund"}
