"""Example 02, multi-step tool use.

The point: an agent that takes several turns, call a tool, read the
observation, then finish. The Policy is still hand-written (a tiny state
machine) so it runs offline with no LLM, but it exercises the real loop:
decide -> act -> fold observation back into Context -> decide again.

Run:  python examples/02_tool_use.py
"""

from microagent import Action, Agent, Context, FunctionPolicy, Tool, final_answer_tool, tool


@tool(description="Multiply two integers a and b.")
def multiply(a: int, b: int) -> int:
    return a * b


def build() -> Agent:
    def brain(context: Context, tools):
        # Has the calculator already run? If so, answer with its result.
        last = context.last()
        if last is not None and last.name == "multiply":
            return Action(
                tool="final_answer",
                args={"answer": f"The product is {last.content}."},
                reasoning="calculator has produced a result, report it",
            )
        # Otherwise, do the multiplication first.
        return Action(
            tool="multiply",
            args={"a": 6, "b": 7},
            reasoning="need the product before I can answer",
        )

    return Agent(
        policy=FunctionPolicy(brain),
        tools=[multiply, final_answer_tool()],
        name="calculator",
        max_steps=4,
    )


if __name__ == "__main__":
    agent = build()
    print(agent.answer("what is 6 times 7?"))
    print("\n--- full trace ---")
    print(agent.run("what is 6 times 7?"))
