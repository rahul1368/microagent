"""Tool, a typed function the agent can call to sense or act on the world.

A Tool is the agent's *hands*. It maps keyword arguments to an observation (a
string the agent can read back). Tools are the ONLY way an agent affects
anything, which is exactly why safety lives here (see ``with_approval``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    """A named, described, callable unit of capability.

    Attributes:
        name: how the Policy refers to this tool.
        description: what it does (shown to an LLM Policy so it can choose).
        fn: the underlying Python callable. Its result is stringified.
        terminal: if True, calling this tool ends the agent loop (e.g. a
            ``final_answer`` tool). This is how an agent decides it is done.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    terminal: bool = False

    def __call__(self, **kwargs: Any) -> str:
        return str(self.fn(**kwargs))


def tool(
    name: str | None = None,
    description: str = "",
    terminal: bool = False,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a plain function into a :class:`Tool`.

    >>> @tool(description="Add two numbers.")
    ... def add(a: int, b: int) -> int:
    ...     return a + b
    >>> add(a=2, b=3)
    '5'
    """

    def wrap(fn: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            fn=fn,
            terminal=terminal,
        )

    return wrap


def with_approval(
    inner: Tool,
    approver: Callable[[str, dict[str, Any]], bool],
) -> Tool:
    """Wrap a Tool so a human (or another Policy) must approve before it runs.

    This is the "safe by design" idea made concrete, and note that it is
    *composition*, not a new primitive. Safety is simply a Tool that wraps
    another Tool. If ``approver`` returns False, the action is refused and the
    world is never touched. (This is exactly the layer autonomous agents that
    act on your behalf tend to skip.)
    """

    def guarded(**kwargs: Any) -> Any:
        if not approver(inner.name, kwargs):
            return f"[denied] action '{inner.name}' was not approved by the human."
        return inner.fn(**kwargs)

    return Tool(
        name=inner.name,
        description=inner.description,
        fn=guarded,
        terminal=inner.terminal,
    )


def final_answer_tool(name: str = "final_answer") -> Tool:
    """A conventional terminal tool: return the answer to the user and stop."""

    return Tool(
        name=name,
        description="Give the final answer to the user and end the task.",
        fn=lambda answer: answer,
        terminal=True,
    )
