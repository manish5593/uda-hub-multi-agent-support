"""Supervisor: identifies the customer and routes the ticket to the right agent.

Routing is deterministic and auditable: the LLM classifies, the supervisor applies
business rules on top of that classification.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agentic import config, rate_limits
from agentic.agents.common import add_trace, last_user_message
from agentic.agents.memory_agent import build_customer, recall
from agentic.llm import get_llm
from agentic.tools import cultpass_tools, ticket_tools, verification

ACCOUNT_ISSUES = {"reservation", "subscription", "billing_refund"}
BLOCKED_SENSITIVE_ISSUES = {"login_access", "reservation", "subscription", "billing_refund", "account_blocked", "complaint"}
# Problems that should reach a human when they come back: the documented fix didn't work.
# Routine requests (reservations, subscription changes, questions) are expected to repeat.
REPEAT_ESCALATION_ISSUES = {"login_access", "technical", "billing_refund", "account_privacy", "complaint"}


def repeat_issue_types(state: dict) -> set[str]:
    """Issue types this customer already reported: in previous tickets, or earlier on this ticket
    when it was marked resolved (the customer writing again means the answer didn't fix it)."""
    types = {t["issue_type"] for t in state.get("prior_tickets") or [] if t.get("issue_type")}
    if state.get("ticket_status") == "resolved" and state.get("ticket_issue_type"):
        types.add(state["ticket_issue_type"])
    return types


URGENCY_LEVELS = ["low", "medium", "high", "critical"]


def effective_urgency(classified, reported) -> str:
    """The more severe of the classifier's urgency and the urgency reported in the ticket metadata,
    so the LLM can raise a ticket's urgency but never lower what the source system reported."""
    levels = [u for u in (classified, reported) if u in URGENCY_LEVELS]
    return max(levels, key=URGENCY_LEVELS.index) if levels else "medium"


def route_ticket(classification: dict, customer: dict, ticket_status: str,
                 repeat_issues: frozenset = frozenset(), reported_urgency=None) -> tuple[str, str]:
    """Return (route, reason). Pure function so the decision logic is easy to test."""
    c = classification
    urgency = effective_urgency(c.get("urgency"), reported_urgency)
    if c.get("empty_message"):
        return "request_info", "empty message: asking the customer to describe their issue"
    if ticket_status == "escalated":
        return "escalation_agent", "ticket is already with the human team; forwarding the new information"
    if c.get("human_requested"):
        return "escalation_agent", "customer asked for a human"
    if c.get("issue_type") == "legal_or_safety":
        return "escalation_agent", "legal/safety issue requires a human"
    if urgency == "critical":
        source = "reported in the ticket metadata" if reported_urgency == "critical" else "classified"
        return "escalation_agent", f"critical urgency ({source}) requires a human"
    if c.get("injection_suspected"):
        return "escalation_agent", "the message tries to instruct the AI (possible prompt injection)"
    if c.get("confidence", 0) < config.CLASSIFIER_MIN_CONFIDENCE:
        return "escalation_agent", f"classification confidence {c.get('confidence', 0):.2f} is too low"
    if customer.get("identified") and customer.get("is_blocked") and (
        c.get("issue_type") in BLOCKED_SENSITIVE_ISSUES or c.get("requires_account_action")
    ):
        return "escalation_agent", "account is blocked: policy requires Trust & Safety review"
    if c.get("issue_type") == "account_blocked" and customer.get("identified"):
        return "escalation_agent", "customer reports a blocked account"
    if c.get("issue_type") in REPEAT_ESCALATION_ISSUES and c.get("issue_type") in repeat_issues:
        return "escalation_agent", f"repeat {c.get('issue_type')} issue: the customer reported it before"
    needs_account = needs_account_access(c)
    if needs_account and not customer.get("identified"):
        return "request_info", "account action needed but customer is not identified"
    if needs_account:
        return "account_agent", f"{c.get('issue_type')} needs account data/actions"
    return "knowledge_agent", f"{c.get('issue_type')} can be answered from the knowledge base"


def needs_account_access(classification: dict) -> bool:
    return bool(classification.get("requires_account_action") or classification.get("issue_type") in ACCOUNT_ISSUES)


def _verify_guest(state: dict) -> dict:
    """Identify a guest only once they prove they own the email they gave, with a one-time code.

    Returns the state updates: the pending verification, this turn's verification status and, once
    verified, the customer and their long-term memory.
    """
    pending = state.get("verification")
    email = (state.get("classification") or {}).get("customer_email")
    code = verification.code_in(last_user_message(state))

    if pending and code:
        status, pending = verification.check(pending, code)
        update = {"verification": pending, "verification_status": status}
        if status == "verified":
            previous = state["verification"]
            ticket_tools.link_ticket_to_customer(state["ticket_id"], previous["user_id"], previous["user_name"])
            customer = build_customer(ticket_tools.get_or_create_ticket(state["ticket_id"]))
            update.update(customer=customer, **recall(customer, state["ticket_id"]))
            customer_id = previous["user_id"]
        else:
            customer_id = None
        update["trace"] = add_trace(state, f"[supervisor] verification code {status}", event="identify",
                                    customer=customer_id, verification=status)
        return update

    # A new email (or the first one) for an account action: send a code, whether or not it has an account
    if email and needs_account_access(state.get("classification") or {}) and (
            not pending or pending["email"] != email.strip().lower()):
        # Counted for every email, registered or not, so the limit doesn't reveal which ones exist
        if not rate_limits.can_send_code(email):
            return {"verification_status": "too_many_codes",
                    "trace": add_trace(state, "[supervisor] too many verification codes for that email: none sent",
                                       event="identify", customer=None, verification="too_many_codes")}
        rate_limits.record_code(email)
        profile = cultpass_tools.find_customer_by_email(email)
        return {"verification": verification.start(email, profile), "verification_status": "code_sent",
                "trace": add_trace(state, "[supervisor] sent a verification code" if profile else
                                   "[supervisor] no customer for that email: same reply, no code sent",
                                   event="identify", customer=None, verification="code_sent")}
    return {}


def supervise(state: dict) -> dict:
    update = {}
    customer = state.get("customer") or {}

    if not customer.get("identified"):
        update = _verify_guest(state)
        customer = update.get("customer", customer)
        if "trace" in update:
            state = {**state, "trace": update["trace"]}

    route, reason = route_ticket(state["classification"], customer, state.get("ticket_status", "open"),
                                 frozenset(repeat_issue_types({**state, **update})),
                                 reported_urgency=(state.get("metadata") or {}).get("urgency"))
    update.update(route=route, route_reason=reason,
                  trace=add_trace(state, f"[supervisor] route -> {route}: {reason}", event="route", route=route, reason=reason))
    return update


REQUEST_INFO_MESSAGE = (
    "I'd be happy to help with that! To look into your account, could you please share the email "
    "address registered with your CultPass account?"
)


EMPTY_MESSAGE_REPLY = "Hi! It looks like your message came through empty. What can I help you with today?"

# One fixed message per verification step. The first is the same whether or not the email has an
# account, so the reply never reveals which emails are registered.
VERIFICATION_MESSAGES = {
    "code_sent": "Thanks! If {email} is registered with CultPass, we've just sent a 6-digit verification code "
                 "to it. Please reply with the code so I can access your account.",
    "wrong_code": "That code doesn't match. Please check the latest email from CultPass and reply with the "
                  "6-digit code.",
    "expired": "That code has expired. Please share your email address again and I'll send a new one.",
    "locked": "That code didn't match too many times. Please share your email address again and I'll send "
              "a new code.",
    "too_many_codes": "We've sent several verification codes to that email recently. For your security, please "
                      "reply with the latest code you received, or wait an hour and ask for a new one.",
}

TRANSLATE_PROMPT = """Translate this customer support message into {language}.
Keep the meaning, the friendly tone and any email address exactly as they are.
Reply with the translated message only."""


def _in_customer_language(message: str, language: str) -> str:
    """Return the fixed message in the customer's language. English needs no LLM call."""
    if not language or language.strip().lower() in {"english", "en"}:
        return message
    reply = get_llm().invoke([SystemMessage(content=TRANSLATE_PROMPT.format(language=language)),
                              HumanMessage(content=message)])
    return reply.content.strip() or message


def request_info(state: dict) -> dict:
    """Ask the customer for the information needed to continue (keeps the ticket open)."""
    classification = state.get("classification") or {}
    status = state.get("verification_status")
    pending = state.get("verification")
    message = REQUEST_INFO_MESSAGE
    asked = "to identify themselves"
    if classification.get("empty_message"):
        message, asked = EMPTY_MESSAGE_REPLY, "what they need help with"
    elif status in VERIFICATION_MESSAGES:
        message = VERIFICATION_MESSAGES[status].format(email=(pending or {}).get("email", "that email"))
        asked = "for their verification code" if pending else "to identify themselves"
    elif pending:
        # still waiting for the code from an earlier turn
        message, asked = VERIFICATION_MESSAGES["code_sent"].format(email=pending["email"]), "for their verification code"
    language = classification.get("language", "English")
    return {"final_response": _in_customer_language(message, language),
            "trace": add_trace(state, f"[request_info] asked customer {asked} ({language})",
                               event="request_info", asked=asked, language=language)}
