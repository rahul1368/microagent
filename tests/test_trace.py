"""Tests for the trace reader."""

from microagent import Action, Agent, FunctionPolicy, final_answer_tool, tool
from microagent.trace import render


def test_render_tells_the_story_in_order():
    @tool(description="double a number")
    def double(n: int) -> int:
        return n * 2

    def brain(ctx, tools):
        last = ctx.last()
        if last is not None and last.name == "double":
            return Action("final_answer", {"answer": last.content})
        return Action("double", {"n": 21}, reasoning="I need to double it first")

    agent = Agent(FunctionPolicy(brain), tools=[double, final_answer_tool()], max_steps=4)
    story = render(agent.run("double 21"))

    # The story reads top to bottom: question, thought, tool result, answer.
    assert 'you asked: "double 21"' in story
    assert "I need to double it first" in story
    assert "double returned: 42" in story
    assert story.strip().endswith("final answer: 42")

    # The user's line comes before the final answer line.
    assert story.index("you asked") < story.index("final answer")
