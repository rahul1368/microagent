"""Tests for the Agent loop and composition, the proof the primitives hold.

Each test builds a real kind of agent (chatbot, tool-user, multi-agent, guarded)
out of the SAME five primitives. If they all pass, the primitive set composes.
"""

from microagent import (
    Action,
    Agent,
    Context,
    FunctionPolicy,
    Message,
    final_answer_tool,
    tool,
    with_approval,
)


def _answer_policy(answer: str) -> FunctionPolicy:
    return FunctionPolicy(lambda ctx, tools: Action("final_answer", {"answer": answer}))


def test_simple_chatbot_runs_to_terminal():
    agent = Agent(_answer_policy("hello"), tools=[final_answer_tool()])
    assert agent.answer("hi") == "hello"


def test_multi_step_tool_use():
    @tool(description="double a number")
    def double(n: int) -> int:
        return n * 2

    def brain(ctx: Context, tools):
        last = ctx.last()
        if last is not None and last.name == "double":
            return Action("final_answer", {"answer": last.content})
        return Action("double", {"n": 21})

    agent = Agent(FunctionPolicy(brain), tools=[double, final_answer_tool()], max_steps=4)
    assert agent.answer("double 21") == "42"


def test_agent_as_tool_composition():
    inner = Agent(_answer_policy("inner-result"), tools=[final_answer_tool()], name="inner")
    inner_tool = inner.as_tool(description="the inner agent")

    def brain(ctx: Context, tools):
        last = ctx.last()
        if last is not None and last.name == "inner":
            return Action("final_answer", {"answer": f"outer saw: {last.content}"})
        return Action("inner", {"input": "go"})

    outer = Agent(FunctionPolicy(brain), tools=[inner_tool, final_answer_tool()], max_steps=4)
    assert outer.answer("delegate") == "outer saw: inner-result"


def test_approval_gate_inside_agent():
    @tool(description="delete something")
    def delete(path: str) -> str:
        return f"deleted {path}"

    guarded = with_approval(delete, lambda name, args: False)

    def brain(ctx: Context, tools):
        last = ctx.last()
        if last is not None and last.name == "delete":
            return Action("final_answer", {"answer": last.content})
        return Action("delete", {"path": "/x"})

    agent = Agent(FunctionPolicy(brain), tools=[guarded, final_answer_tool()], max_steps=4)
    assert "denied" in agent.answer("delete /x")


def test_unknown_tool_does_not_crash():
    policy = FunctionPolicy(lambda ctx, tools: Action("nope", {}))
    agent = Agent(policy, tools=[final_answer_tool()], max_steps=2)
    ctx = agent.run("hi")
    # Loop survives, records the error, and stops at max_steps.
    assert any("unknown tool" in m.content for m in ctx)


def test_max_steps_is_respected():
    # A brain that never calls a terminal tool must still halt.
    @tool(description="noop")
    def noop() -> str:
        return "ok"

    agent = Agent(
        FunctionPolicy(lambda ctx, tools: Action("noop", {})),
        tools=[noop],
        max_steps=3,
    )
    ctx = agent.run("go")
    # 1 user msg + 3 steps * (assistant + tool) = 7 messages.
    assert len(ctx) == 1 + 3 * 2


def test_system_prompt_is_prepended():
    agent = Agent(_answer_policy("hi"), tools=[final_answer_tool()], system="be nice")
    ctx = agent.run("hello")
    assert ctx.messages[0].role == "system"
    assert ctx.messages[0].content == "be nice"
