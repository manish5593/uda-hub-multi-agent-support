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

# Text that tries to instruct the AI instead of asking for help: overriding its instructions, posing as
# the system, or setting the classifier's own fields. Code flags it, so routing never relies on the model
# noticing (or on the confidence it reports, which the same text can set).
INJECTION_PATTERN = re.compile(
    r"\b(ignore|disregard|override)\b[^.\n]{0,30}\b(instructions?|rules|prompt)\b|"
    r"\[\s*system\b|\bsystem\s+(prompt|instructions?)\b|"
    r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?(ai|assistant|classifier|model|bot)\b|"
    r"\b(set|change|make)\b[^.\n]{0,20}\b(confidence|issue_type|urgency|classification)\b|"
    r"\b(confidence|issue_type|urgency)\s*[=:]|"
    r"\bclassify\s+(this|it|me|my\s+\w+)\s+as\b|"
    r"\byou\s+are\s+now\s+(an?\s+|the\s+)?(admin\w*|developer|system|unrestricted)\b|"
    r"\bdeveloper\s+mode\b|\bjailbreak|"
    r"</?\s*customer_message\s*>",
    re.IGNORECASE,
)
CUSTOMER_TAG = re.compile(r"</?\s*customer_message\s*>", re.IGNORECASE)


def as_customer_data(text: str) -> str:
    """Wrap customer text in tags that mark it as data, not instructions. Tags the customer typed are
    removed first, so they can't close the block early and write "instructions" after it."""
    return f"<customer_message>\n{CUSTOMER_TAG.sub('', text or '')}\n</customer_message>"


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


def conversation_text(state: dict, tag_customer: bool = False) -> str:
    """The conversation as text. With `tag_customer`, each customer message is wrapped by as_customer_data."""
    lines = []
    for m in conversation(state):
        if isinstance(m, HumanMessage):
            lines.append(f"Customer: {as_customer_data(m.content) if tag_customer else m.content}")
        else:
            lines.append(f"Assistant: {m.content}")
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
