"""Helpers shared by the agents: tracing and prompt-context formatting."""
import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentic import config
from agentic.event_log import log_event

# Claims of a special status or permission. Only the system can grant these, so when a customer says
# so it is an unverified claim, never a fact (a prompt-injection route into memory and handoff notes).
PERMISSION_CLAIM = re.compile(
    r"\b(pre-?approv\w*|authori[sz]ed|entitled|exempt\w*|waived?|guaranteed|vip|admin\w*|"
    r"special\s+(access|permission|status|treatment))\b",
    re.IGNORECASE,
)


def add_trace(state: dict, entry: str, event: str = "note", ticket_id=None, **fields) -> list[str]:
    """Record one agent decision: in the ticket's `trace` (for display) and as a structured event in
    the searchable event log. `entry` starts with the agent's name in brackets, e.g. "[supervisor] ..."."""
    agent = entry[1:entry.index("]")] if entry.startswith("[") and "]" in entry else "workflow"
    log_event(ticket_id or state.get("ticket_id"), agent, event, entry, **fields)
    return [*state.get("trace", []), entry]


def last_user_message(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def conversation(state: dict, limit: int = config.MAX_CONVERSATION_MESSAGES) -> list[BaseMessage]:
    """Customer-facing conversation of this session (short-term memory)."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    return msgs[-limit:]


def conversation_text(state: dict) -> str:
    lines = []
    for m in conversation(state):
        role = "Customer" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def customer_context_block(state: dict) -> str:
    """Everything the agents should know about the customer, from short- and long-term memory."""
    customer = state.get("customer") or {}
    parts = []
    if customer.get("identified"):
        sub = customer.get("subscription")
        parts.append(
            f"Customer: {customer.get('name')} (id {customer.get('external_user_id')}, {customer.get('email')})\n"
            f"Account blocked: {customer.get('is_blocked')}\n"
            f"Subscription: {json.dumps(sub) if sub else 'none'}"
        )
    else:
        parts.append("Customer: not identified yet (guest).")
    meta = state.get("metadata") or {}
    parts.append(f"Ticket: {state.get('ticket_id')} | channel: {state.get('channel')} | status: {state.get('ticket_status')}"
                 + (f" | reported urgency: {meta['urgency']}" if meta.get("urgency") else ""))
    if state.get("memories"):
        parts.append("Known customer preferences/facts (long-term memory):\n"
                     + "\n".join(f"- {m}" for m in state["memories"]))
    if state.get("prior_tickets"):
        parts.append("Previous tickets from this customer:\n" + "\n".join(
            f"- [{t['created_at']}] {t['issue_type'] or 'unclassified'} ({t['status']}): "
            f"\"{t['customer_message']}\" -> \"{t['last_reply'] or 'no reply'}\""
            for t in state["prior_tickets"]
        ))
    if state.get("ticket_history"):
        parts.append("Messages already on this ticket before this session:\n" + "\n".join(
            f"- {m['role']}: {m['content']}" for m in state["ticket_history"]
        ))
    return "\n\n".join(parts)
