"""Unit tests for the individual primitives."""

import pytest

from microagent import (
    Action,
    Context,
    Message,
    Tool,
    final_answer_tool,
    tool,
    with_approval,
)


def test_message_is_immutable():
    m = Message("user", "hi")
    with pytest.raises(Exception):
        m.content = "bye"  # frozen dataclass


def test_context_is_append_only():
    c0 = Context()
    c1 = c0.add(Message("user", "a"))
    c2 = c1.add(Message("assistant", "b"))
    # Old contexts are unchanged, history only ever grows.
    assert len(c0) == 0
    assert len(c1) == 1
    assert len(c2) == 2
    assert c2.last().content == "b"


def test_context_of_and_iter():
    c = Context.of(Message("user", "x"), Message("user", "y"))
    assert [m.content for m in c] == ["x", "y"]


def test_tool_call_stringifies_result():
    @tool(description="add")
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, Tool)
    assert add(a=2, b=3) == "5"
    assert add.description == "add"


def test_final_answer_tool_is_terminal():
    fa = final_answer_tool()
    assert fa.terminal is True
    assert fa(answer="done") == "done"


def test_with_approval_blocks_when_denied():
    @tool(description="danger")
    def danger(x: str) -> str:
        return f"did {x}"

    denied = with_approval(danger, lambda name, args: False)
    allowed = with_approval(danger, lambda name, args: True)

    assert "denied" in denied(x="thing")
    assert allowed(x="thing") == "did thing"


def test_action_defaults():
    a = Action(tool="final_answer")
    assert a.args == {}
    assert a.reasoning == ""
