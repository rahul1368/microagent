"""Example 03, multi-agent, for free.

The point: there is no "multi-agent" primitive. A manager agent simply has
another agent among its tools (Agent.as_tool). Composition, the same way a
big nn.Module is built from smaller ones.

Run:  python examples/03_multi_agent.py
"""

from microagent import Action, Agent, Context, FunctionPolicy, final_answer_tool, tool


# --- a small specialist agent: it "researches" a topic ------------------------
@tool(description="Look up a fact about a topic.")
def lookup(topic: str) -> str:
    facts = {
        "pytorch": "PyTorch is an ML framework, its core ideas are Tensors and autograd.",
        "agents": "An agent is a loop: decide, act, observe, repeat until done.",
    }
    return facts.get(topic.lower(), f"No fact on file for {topic!r}.")


def researcher() -> Agent:
    def brain(context: Context, tools):
        last = context.last()
        if last is not None and last.name == "lookup":
            return Action("final_answer", {"answer": last.content}, "return the fact I found")
        topic = context.messages[-1].content
        return Action("lookup", {"topic": topic}, "look the topic up first")

    return Agent(
        FunctionPolicy(brain),
        tools=[lookup, final_answer_tool()],
        name="researcher",
        description="Answers factual questions about a single topic.",
    )


# --- the manager: its tool IS the researcher agent ----------------------------
def manager() -> Agent:
    research_tool = researcher().as_tool(
        name="research", description="Delegate a factual question to the researcher."
    )

    def brain(context: Context, tools):
        last = context.last()
        if last is not None and last.name == "research":
            return Action(
                "final_answer",
                {"answer": f"My researcher reports: {last.content}"},
                "wrap up the delegated answer",
            )
        return Action("research", {"input": "pytorch"}, "delegate to the researcher")

    return Agent(
        FunctionPolicy(brain),
        tools=[research_tool, final_answer_tool()],
        name="manager",
        max_steps=4,
    )


if __name__ == "__main__":
    boss = manager()
    print(boss.answer("tell me about pytorch"))
    print("\n--- manager trace (the researcher ran inside the 'research' tool) ---")
    print(boss.run("tell me about pytorch"))
