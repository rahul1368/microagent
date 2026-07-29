"""A small but real-feeling demo, used to record the README gif.

A support agent handles a refund request. It looks up the order, checks the
policy, issues a refund (which needs a human to sign off), and replies to the
customer. Watch the story print out, including the approval step.

Runs straight from a clone with no install needed:

    python3 demo.py
"""

import sys
from pathlib import Path

# Make the package importable without installing, so the demo runs anywhere.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from microagent import (  # noqa: E402
    Action,
    Agent,
    FunctionPolicy,
    final_answer_tool,
    tool,
    with_approval,
)
from microagent.trace import show  # noqa: E402


@tool(description="Look up an order by its id.")
def look_up_order(order: str) -> str:
    return "A1234, Wireless Headphones, $79, delivered, reported defective"


@tool(description="Check whether an order qualifies for a refund.")
def check_policy(order: str) -> str:
    return "eligible, within the 30 day window and reported defective"


@tool(name="issue_refund", description="Send a refund to the customer. Needs human sign off.")
def _issue_refund(order: str, amount: int) -> str:
    return f"approved by a human, ${amount} refunded to the original card"


# The refund tool only fires after a human says yes. Here the human approves.
issue_refund = with_approval(_issue_refund, approver=lambda name, args: True)


def brain(ctx, tools):
    """A scripted brain that walks the support flow one step at a time."""
    last = ctx.last()
    stage = last.name if last is not None and last.role == "tool" else None

    if stage is None:
        return Action("look_up_order", {"order": "A1234"}, "let me pull up the order first")
    if stage == "look_up_order":
        return Action("check_policy", {"order": "A1234"}, "now check if it qualifies")
    if stage == "check_policy":
        return Action(
            "issue_refund",
            {"order": "A1234", "amount": 79},
            "it qualifies, and a refund needs a human to sign off",
        )
    return Action(
        "final_answer",
        {"answer": "Good news, A1234 qualifies. I refunded $79 to your card, here in 3 to 5 days."},
        "all set, time to reply to the customer",
    )


if __name__ == "__main__":
    agent = Agent(
        FunctionPolicy(brain),
        tools=[look_up_order, check_policy, issue_refund, final_answer_tool()],
        name="support",
        max_steps=6,
    )
    show(agent.run("Can I get a refund for order A1234? It arrived broken."))
