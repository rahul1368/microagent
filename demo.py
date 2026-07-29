"""A tiny demo, used to record the README gif and to try things by hand.

Run:  python demo.py
"""

from microagent import Action, Agent, FunctionPolicy, final_answer_tool, tool
from microagent.trace import show


@tool(description="Multiply two integers a and b.")
def multiply(a: int, b: int) -> int:
    return a * b


def brain(ctx, tools):
    last = ctx.last()
    if last is not None and last.name == "multiply":
        return Action("final_answer", {"answer": f"The product is {last.content}."})
    return Action("multiply", {"a": 6, "b": 7}, reasoning="I need the product first")


if __name__ == "__main__":
    agent = Agent(FunctionPolicy(brain), tools=[multiply, final_answer_tool()], name="calculator")
    show(agent.run("what is 6 times 7?"))
