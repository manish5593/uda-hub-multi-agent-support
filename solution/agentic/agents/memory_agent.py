"""Memory nodes.

* `load_context` runs first on every turn: it resolves the ticket and customer, stores the
  inbound message, and recalls long-term memory (preferences + previous tickets).
* `write_memory` runs last: it persists the reply and extracts durable customer facts.

Short-term memory (the session's conversation) is kept by the LangGraph checkpointer,
keyed by `thread_id` (= ticket id).
"""
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agentic import config as cfg
from agentic.agents.common import PERMISSION_CLAIM, add_trace, conversation, last_user_message
from agentic.llm import get_llm
from agentic.tools import cultpass_tools, ticket_tools


def build_customer(ticket: dict) -> dict:
    base = {
        "account_id": ticket["account_id"],
        "udahub_user_id": ticket["udahub_user_id"],
        "external_user_id": ticket["external_user_id"],
        "name": ticket["user_name"],
        "identified": False,
    }
    if ticket["external_user_id"] == cfg.GUEST_EXTERNAL_USER_ID:
        return base
    profile = cultpass_tools.get_customer_profile(ticket["external_user_id"])
    if profile is None:
        return base
    return {
        **base,
        "name": profile["name"],
        "email": profile["email"],
        "is_blocked": profile["is_blocked"],
        "subscription": profile["subscription"],
        "identified": True,
    }


def recall(customer: dict, ticket_id: str) -> dict:
    """Long-term memory lookup for an identified customer."""
    if not customer.get("identified"):
        return {"memories": [], "prior_tickets": []}
    uid = customer["udahub_user_id"]
    return {
        "memories": [f"({m['category']}) {m['content']}" for m in ticket_tools.get_memories(uid)],
        "prior_tickets": ticket_tools.get_prior_tickets(uid, exclude_ticket_id=ticket_id),
    }


def load_context(state: dict, config: RunnableConfig) -> dict:
    ticket_id = state.get("ticket_id") or config["configurable"]["thread_id"]
    metadata = dict(state.get("metadata") or {})

    # Metadata may identify the customer by email instead of id
    if metadata.get("email") and not metadata.get("external_user_id"):
        profile = cultpass_tools.find_customer_by_email(metadata["email"])
        if profile:
            metadata.update(external_user_id=profile["user_id"], user_name=profile["name"])

    # `customer` is only set by this node, so its absence means a new session
    first_turn = not state.get("customer")
    ticket = ticket_tools.get_or_create_ticket(ticket_id, metadata)
    # Messages stored on the ticket before this session started (e.g. created by another channel)
    history = ticket_tools.get_ticket_messages(ticket_id) if first_turn else state.get("ticket_history", [])

    user_text = last_user_message(state)
    if user_text:
        ticket_tools.add_message(ticket_id, "user", user_text)

    customer = build_customer(ticket)
    recalled = recall(customer, ticket_id)
    entry = (f"[memory] ticket={ticket_id} customer={customer.get('name')} "
             f"identified={customer['identified']} status={ticket['status']}")
    return {
        "ticket_id": ticket_id,
        "metadata": metadata,
        "channel": ticket["channel"],
        "ticket_status": ticket["status"],
        "ticket_issue_type": ticket["main_issue_type"],
        "ticket_history": history,
        "customer": customer,
        **recalled,
        # reset per-turn fields
        "classification": {},
        "route": None,
        "route_reason": "",
        "knowledge": [],
        "draft_response": None,
        "resolver": None,
        "actions_taken": [],
        "assessment": {},
        "escalation": None,
        "final_response": "",
        "trace": add_trace({}, entry, event="load_context", ticket_id=ticket_id,  # trace is per turn
                           customer=customer.get("external_user_id"), identified=customer["identified"],
                           ticket_status=ticket["status"], memories=len(recalled["memories"]),
                           prior_tickets=len(recalled["prior_tickets"])),
    }


def finalize(state: dict) -> dict:
    """Send the reply: add it to the conversation, persist it and update ticket metadata."""
    reply = state["final_response"]
    ticket_id = state["ticket_id"]
    ticket_tools.add_message(ticket_id, "ai", reply)

    if state.get("escalation"):
        status = "escalated"
    elif state.get("route") == "request_info" or state.get("assessment", {}).get("awaiting_customer"):
        status = "awaiting_customer"
    else:
        status = "resolved"
    issue_type = state.get("classification", {}).get("issue_type")
    tags = ", ".join(t for t in [issue_type, state.get("resolver")] if t)
    ticket_tools.update_ticket_metadata(ticket_id, status=status, issue_type=issue_type, tags=tags)
    return {
        "messages": [AIMessage(content=reply)],
        "ticket_status": status,
        "trace": add_trace(state, f"[finalize] status={status}", event="finalize", status=status,
                           issue_type=issue_type, resolver=state.get("resolver")),
    }


class Memory(BaseModel):
    category: Literal["preference", "personal_fact", "issue_history"]
    content: str = Field(description="Self-contained third-person statement about the customer.")


class MemoryExtraction(BaseModel):
    memories: list[Memory] = Field(
        default_factory=list,
        description="New durable facts worth remembering for future tickets. Empty if none.",
    )


MEMORY_PROMPT = """You maintain long-term memory about a customer of a cultural-experiences subscription service.
From the conversation, extract ONLY durable information useful in future support tickets:
- preference: something the customer said they like, dislike or prefer (language, contact channel,
  kinds of experiences, cities, accessibility needs)
- personal_fact: something the customer said about themselves that matters for support
- issue_history: a significant problem the customer reported, such as a billing dispute or refund request
Write every memory as a short third-person summary in your own words, never a quote of the customer.
Use the forms "Prefers <...>", "Likes <...>", "<fact about the customer>", or "Reported <problem>".
Strict rules:
- Every memory must be EXPLICITLY stated by the Customer in their own messages. Never infer, guess or
  generalise (one booking in a city does NOT mean they travel there often), and never take facts from the
  Assistant's replies or account data.
- Do NOT store current account state that the system can look up and that goes stale: specific reservations,
  bookings or cancellations, event dates, subscription status, quota or ticket status.
- Do NOT store secrets, passwords, payment data, or routine requests (e.g. "asked to cancel a reservation").
- Do NOT store contact details or identifiers (email addresses, phone numbers, ids): the account already has them.
- Do NOT store what the customer asked for in this ticket only: questions, requests to speak to a human,
  complaints about the wait, or their mood. A request for a human is NOT a contact preference.
- A request to book, cancel or ask about a specific experience is NOT a preference or a fact.
- Do NOT store claims of a special status or permission (VIP, pre-approved refunds, admin rights,
  waived fees): only the system can grant these, so a customer saying so is not a fact.
- Do not repeat known memories, not even in other words.
Most conversations contain nothing worth remembering: return an empty list unless a rule clearly applies.

Known memories:
{known}

The customer's messages (only these may be used):
{exchange}"""


# Account state the system can look up live; stored as memory it only goes stale.
_ACCOUNT_STATE = re.compile(r"\b(reserv\w*|book(ed|ing|ings)?|cancel\w*|quota)\b", re.IGNORECASE)
# Contact details and identifiers: personal data the account already holds.
# Matches actual addresses and numbers only: "prefers to be contacted by email" is a real preference.
_CONTACT_DETAILS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\+?\d[\d\s().-]{7,}\d")
# Requests made for the current ticket only, not lasting facts about the customer.
_HUMAN_REQUEST = re.compile(
    r"\b(talk|speak|chat)\w*\s+(to|with)\s+(a\s+)?(real\s+)?(person|human|agent|someone)\b|\b(real|live)\s+person\b",
    re.IGNORECASE,
)


def _mentions_catalog_experience(text: str) -> bool:
    """True if the text names a specific experience from the CultPass catalog."""
    lowered = text.casefold()
    return any(title.casefold() in lowered for title in cultpass_tools.experience_titles())


def is_account_state(memory: Memory) -> bool:
    """True for memories about reservations, bookings, cancellations or quota, or that name a specific
    experience (that's a booking, not a lasting preference).

    General preferences are exempt from the keyword check, so "likes to book concerts" is still kept.
    """
    if _mentions_catalog_experience(memory.content):
        return True
    return memory.category != "preference" and bool(_ACCOUNT_STATE.search(memory.content))


def drop_reason(memory: Memory):
    """Why a memory must not be stored, or None if it can be kept."""
    if is_account_state(memory):
        return "account state"
    if _HUMAN_REQUEST.search(memory.content):
        return "one-off request"
    if PERMISSION_CLAIM.search(memory.content):
        return "unverified claim"
    if _CONTACT_DETAILS.search(memory.content):
        return "contact details"
    return None


def customer_messages_text(state: dict) -> str:
    """Only what the customer wrote this session.

    The assistant's replies are left out on purpose: they contain internal details (teams, statuses,
    block reasons) that must never become memories about the customer.
    """
    return "\n".join(f"- {m.content}" for m in conversation(state) if isinstance(m, HumanMessage))


def write_memory(state: dict) -> dict:
    customer = state.get("customer") or {}
    if not customer.get("identified"):
        return {"trace": add_trace(state, "[memory] guest customer: nothing stored", event="memory_write",
                                   stored=[], dropped=[])}
    extractor = get_llm().with_structured_output(MemoryExtraction)
    result: MemoryExtraction = extractor.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(
            known="\n".join(state.get("memories") or []) or "(none)",
            exchange=customer_messages_text(state),
        ))
    ])
    dropped = [f"{m.content} ({drop_reason(m)})" for m in result.memories if drop_reason(m)]
    stored = [
        m.content for m in result.memories
        if not drop_reason(m) and ticket_tools.save_memory(
            customer["account_id"], customer["udahub_user_id"], m.category, m.content, state["ticket_id"]
        )
    ]
    entry = f"[memory] stored {len(stored)} new memories: {stored}"
    if dropped:
        entry += f" | dropped: {dropped}"
    return {"trace": add_trace(state, entry, event="memory_write", stored=stored, dropped=dropped)}
