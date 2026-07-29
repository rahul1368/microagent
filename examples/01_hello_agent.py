"""Example 01, the smallest possible agent.

The point: show the whole loop with the simplest brain imaginable. The Policy
here is one line of Python (via FunctionPolicy). A real brain would be an LLM -
but the Agent, the loop, and the primitives are identical either way.

Run:  python examples/01_hello_agent.py
"""

from microagent import Agent, Context, FunctionPolicy, Tool, final_answer_tool


def build() -> Agent:
    # A brain so simple it isn't even AI: always give the final answer.
    def brain(context: Context, tools):
        user = context.last().content
        from microagent import Action

        return Action(
            tool="final_answer",
            args={"answer": f"Hello! You said: {user!r}"},
            reasoning="nothing to look up, just greet the user",
        )

    return Agent(
        policy=FunctionPolicy(brain),
        tools=[final_answer_tool()],
        name="greeter",
    )


if __name__ == "__main__":
    agent = build()
    print(agent.answer("hi there"))
    print("\n--- full trace ---")
    print(agent.run("hi there"))
