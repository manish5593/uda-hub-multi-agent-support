"""Account resolver: a ReAct agent that uses CultPass tools scoped to the current customer."""
import json
import re

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from agentic.agents.common import add_trace, conversation, customer_context_block, last_user_message
from agentic.llm import get_llm
from agentic.tools.cultpass_tools import build_account_tools
from agentic.tools.knowledge_tools import build_knowledge_tool

ACCOUNT_PROMPT = """You are a CultPass customer support agent with access to the customer's account tools.

Rules:
- Use the tools to check real account data before answering; never guess reservation ids, dates or quota.
- Never invent ids. Get reservation ids from get_my_reservations and experience ids from find_experiences
  BEFORE calling book_experience or cancel_my_reservation.
- For recommendations, search the catalog with find_experiences and suggest real upcoming experiences that
  match the customer's known preferences (premium ones only if their tier allows it). The search matches
  literal words, so for recommendations call find_experiences with no query (optionally a location) and choose
  from the whole catalog yourself. Recommend only the 2-3 that genuinely match their preferences, with one line on
  why each fits; leave out ones that don't fit rather than listing everything. Never recommend one marked
  already_booked; you can mention that they have it.
- Look up policies with search_knowledge_base before explaining them (cancellation, refunds, quota...).
- Only perform changes (book, cancel, change subscription) that the customer clearly asked for.
  If the request is ambiguous (e.g. which reservation), ask a short clarifying question instead.
- Cancelling a reservation and changing the subscription need the customer's confirmation, and the tool
  handles it: as soon as you know what they want changed, call the tool. Never ask for confirmation
  yourself first; the confirmation only counts if the tool asked for it. When the tool returns
  needs_confirmation, ask the customer to reply YES, naming exactly what will change, and stop.
  When they reply yes, call the same tool again with the same arguments. Never say it's done before it is.
- Refunds: never promise one. Use request_refund to register it for support-lead approval and tell
  the customer the billing team will review it.
- If a tool returns an error, explain it plainly and offer the next best option.
- Reply in the customer's language ({language}), be warm and concise, and confirm what you did.

Customer context:
{context}

Ticket summary: {summary}"""


# A clear yes from the customer, with nothing that takes it back
_YES = re.compile(r"^\W*(yes|yeah|yep|y|sure|ok|okay|confirm\w*|go ahead|please do|do it|sim|s[ií]|oui|ja)\b",
                  re.IGNORECASE)
_NO = re.compile(r"\b(no|not|don'?t|never|wait|stop)\b", re.IGNORECASE)


def is_confirmation(text: str) -> bool:
    return bool(_YES.search(text or "")) and not _NO.search(text or "")


def _actions(messages: list) -> list[dict]:
    """Extract the tool calls (and their results) the agent made, for auditing and escalation."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    actions = []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                raw = results.get(call["id"], "")
                try:
                    result = json.loads(raw)
                except (TypeError, ValueError):
                    result = raw
                actions.append({"tool": call["name"], "args": call["args"], "result": result})
    return actions


def resolve_with_account_tools(state: dict) -> dict:
    customer = state["customer"]
    classification = state.get("classification", {})
    # Code, not the LLM, decides whether the customer confirmed the change asked about last turn
    confirmable = state.get("confirmable_action")
    confirmed = confirmable if confirmable and is_confirmation(last_user_message(state)) else None
    tools = build_account_tools(customer["external_user_id"], state["ticket_id"], confirmed_action=confirmed)
    tools.append(build_knowledge_tool(customer["account_id"]))

    agent = create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=SystemMessage(content=ACCOUNT_PROMPT.format(
            language=classification.get("language", "English"),
            context=customer_context_block(state),
            summary=classification.get("summary", ""),
        )),
    )
    try:
        result = agent.invoke({"messages": conversation(state)}, {"recursion_limit": 15})
    except GraphRecursionError:
        # The agent looped without converging: hand over to the quality gate with no draft
        return {
            "draft_response": None,
            "resolver": "account_agent",
            "trace": add_trace(state, "[account_agent] gave up after too many steps", event="tools", tools=[],
                               error="recursion_limit"),
        }
    actions = _actions(result["messages"])
    # The change now waiting for the customer's yes (only the next message can confirm it)
    pending = next((a["result"]["action"] for a in reversed(actions)
                    if isinstance(a["result"], dict) and a["result"].get("needs_confirmation")), None)
    return {
        "draft_response": result["messages"][-1].content,
        "actions_taken": actions,
        "pending_confirmation": pending,
        "resolver": "account_agent",
        "trace": add_trace(state, f"[account_agent] tools used: {[a['tool'] for a in actions]}", event="tools",
                           tools=[{"tool": a["tool"], "args": a["args"],
                                   "success": a["result"].get("success") if isinstance(a["result"], dict) else True}
                                  for a in actions]),
    }
