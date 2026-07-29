"""A tiny demo, used to record the README gif and to try things by hand.

Runs straight from a clone with no install needed:

    python3 demo.py
"""

import sys
from pathlib import Path

# Make the package importable without installing, so the demo runs anywhere.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from microagent import Action, Agent, FunctionPolicy, final_answer_tool, tool  # noqa: E402
from microagent.trace import show  # noqa: E402


@tool(description="Multiply two integers a and b.")
def multiply(a: int, b: int) -> int:
    return a * b


def brain(ctx, tools):
    last = ctx.last()
    if last is not None and last.name == "multiply":
        return Action(
            "final_answer",
            {"answer": f"The product is {last.content}."},
            reasoning="I have the number, time to report it",
        )
    return Action("multiply", {"a": 6, "b": 7}, reasoning="I need the product first")


if __name__ == "__main__":
    agent = Agent(FunctionPolicy(brain), tools=[multiply, final_answer_tool()], name="calculator")
    show(agent.run("what is 6 times 7?"))
