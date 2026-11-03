"""UDA-Hub orchestration graph.

    START -> load_context -> classifier -> supervisor ─┬─> knowledge_agent ─┐
                                                       ├─> account_agent ───┴─> quality_gate ─┬─> approve ──────────┐
                                                       ├─> escalation_agent <─────────────────┘                     │
                                                       └─> request_info ────────────────────────────────────────────┤
                                     END <- write_memory <- finalize <──────────────────────────────────────────────┘

See agentic/design/README.md for the full architecture.
"""
import logging
import traceback
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic.agents.account_agent import resolve_with_account_tools
from agentic.agents.classifier import classify
from agentic.agents.escalation_agent import escalate
from agentic.agents.knowledge_agent import resolve_with_knowledge
from agentic.agents.memory_agent import finalize, load_context, write_memory
from agentic.agents.quality_gate import approve, assess_resolution, should_escalate
from agentic.agents.supervisor import request_info, supervise
from agentic import rate_limits
from agentic.event_log import log_event
from agentic.state import UdaHubState
from agentic.tools import ticket_tools

logger = logging.getLogger("udahub")

FALLBACK_REPLY = ("Sorry, something went wrong on our side while handling your message. "
                  "A member of our support team will follow up with you by email.")
SLOW_DOWN_REPLY = ("You're sending messages faster than we can answer them. "
                   "Please wait a minute, then send your message again.")
BUSY_REPLY = ("Our assistant is very busy right now, so a member of our support team will pick up your "
              "message and follow up with you by email.")


def _after_supervisor(state: UdaHubState) -> str:
    return state["route"]


def _after_quality_gate(state: UdaHubState) -> str:
    escalate_, _ = should_escalate(state)
    return "escalation_agent" if escalate_ else "approve"


def build_graph(checkpointer=None):
    graph = StateGraph(UdaHubState)

    graph.add_node("load_context", load_context)
    graph.add_node("classifier", classify)
    graph.add_node("supervisor", supervise)
    graph.add_node("knowledge_agent", resolve_with_knowledge)
    graph.add_node("account_agent", resolve_with_account_tools)
    graph.add_node("quality_gate", assess_resolution)
    graph.add_node("approve", approve)
    graph.add_node("escalation_agent", escalate)
    graph.add_node("request_info", request_info)
    graph.add_node("finalize", finalize)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classifier")
    graph.add_edge("classifier", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        ["knowledge_agent", "account_agent", "escalation_agent", "request_info"],
    )
    graph.add_edge("knowledge_agent", "quality_gate")
    graph.add_edge("account_agent", "quality_gate")
    graph.add_conditional_edges("quality_gate", _after_quality_gate, ["approve", "escalation_agent"])
    for node in ("approve", "escalation_agent", "request_info"):
        graph.add_edge(node, "finalize")
    graph.add_edge("finalize", "write_memory")
    graph.add_edge("write_memory", END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


orchestrator = build_graph()


def process_ticket(text: str, ticket_id: str, metadata: Optional[dict] = None, agent=None) -> dict:
    """Send one customer message (plus optional ticket metadata) through the system.

    `metadata` can include: account_id, external_user_id or email, user_name, channel, urgency, tags.
    Returns the final state for this turn.
    """
    agent = agent or orchestrator
    payload = {"messages": [HumanMessage(content=text)]}
    if metadata:
        payload["metadata"] = metadata
    try:
        # Checked before the graph runs, so a refused message costs no LLM call
        limited = rate_limits.check_message(ticket_id, metadata)
        if limited:
            return _refuse(ticket_id, text, metadata, *limited)
        return agent.invoke(payload, config={"configurable": {"thread_id": ticket_id}})
    except Exception as exc:
        return _handle_failure(ticket_id, exc)


def _refuse(ticket_id: str, text: str, metadata: Optional[dict], limit: str, reason: str) -> dict:
    """Answer a message that is over a rate or cost limit without running the agents.

    Too many messages: ask the sender to slow down (nothing is saved). Token budget used up: the
    customer still deserves an answer, so the message is saved and the ticket handed to a human.
    """
    entry = f"[workflow] refused ({limit}): {reason}"
    log_event(ticket_id, "workflow", "rate_limited", entry, limit=limit, reason=reason)
    if limit == "too_many_messages":
        return {"ticket_id": ticket_id, "route": None, "ticket_status": "rate_limited",
                "final_response": SLOW_DOWN_REPLY, "trace": [entry]}
    try:
        ticket_tools.get_or_create_ticket(ticket_id, metadata)
        ticket_tools.add_message(ticket_id, "user", text)
        ticket_tools.add_message(ticket_id, "system", f"Not handled automatically: {reason}; needs a human.")
        ticket_tools.add_message(ticket_id, "ai", BUSY_REPLY)
        ticket_tools.update_ticket_metadata(ticket_id, status="escalated", tags="token_budget")
    except Exception:
        logger.exception("Could not hand ticket %s to a human after the token budget ran out", ticket_id)
    return {"ticket_id": ticket_id, "route": None, "ticket_status": "escalated", "final_response": BUSY_REPLY,
            "trace": [entry]}


def _handle_failure(ticket_id: str, exc: Exception) -> dict:
    """Never leave a customer without a reply: log the failure, flag the ticket for a human, and send
    a safe fallback. Covers e.g. the LLM being unreachable or an unknown account in the metadata."""
    entry = f"[workflow] failed: {type(exc).__name__}: {str(exc)[:200]}"
    # The traceback goes in the structured event, not the console, which only gets a one-line warning
    logger.warning("Ticket %s failed: %s: %s", ticket_id, type(exc).__name__, str(exc)[:200])
    log_event(ticket_id, "workflow", "error", entry, error_type=type(exc).__name__, error=str(exc)[:500],
              traceback=traceback.format_exc())
    try:
        if ticket_tools.ticket_exists(ticket_id):
            ticket_tools.add_message(ticket_id, "system", f"Automatic handling failed ({type(exc).__name__}); needs a human.")
            ticket_tools.add_message(ticket_id, "ai", FALLBACK_REPLY)
            ticket_tools.update_ticket_metadata(ticket_id, status="escalated", tags="system_error")
    except Exception:
        logger.exception("Could not flag ticket %s after a failure", ticket_id)
    return {"ticket_id": ticket_id, "route": None, "ticket_status": "error", "final_response": FALLBACK_REPLY,
            "error": f"{type(exc).__name__}: {exc}", "trace": [entry]}
