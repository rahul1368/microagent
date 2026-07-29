"""Turn an agent run into a plain story you can read.

A ``Context`` holds everything that happened, but a raw list of messages is
hard to scan. This module reads that history and prints it the way you would
read a receipt: who asked, what the brain decided, what each tool handed back,
and the final answer at the bottom.

The core never imports this. It is a reading aid you pull in when you want to
see what an agent did.
"""

from __future__ import annotations

from .context import Context
from .message import ASSISTANT, SYSTEM, TOOL, USER


def render(context: Context) -> str:
    """Return the run as a readable, top-to-bottom story."""
    lines: list[str] = []
    final: str | None = None

    for msg in context:
        if msg.role == SYSTEM:
            lines.append(f"(setup) {msg.content}")
        elif msg.role == USER:
            lines.append(f'you asked: "{msg.content}"')
        elif msg.role == ASSISTANT:
            who = msg.name or "the agent"
            lines.append(f"  {who} thought: {msg.content}")
        elif msg.role == TOOL:
            tool_name = msg.name or "tool"
            lines.append(f"    -> {tool_name} returned: {msg.content}")
            final = msg.content

    if final is not None:
        lines.append("")
        lines.append(f"final answer: {final}")

    return "\n".join(lines)


def show(context: Context) -> None:
    """Print the run as a readable story."""
    print(render(context))
