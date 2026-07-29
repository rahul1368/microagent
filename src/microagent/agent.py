"""Agent, a composable unit that runs the loop.

An Agent bundles a Policy + a set of Tools + a stop condition. Running it
repeats one Step:

    decide (Policy) -> act (Tool) -> fold the observation back into Context

...until a terminal tool is called or ``max_steps`` is reached.

Crucially, an Agent is itself usable as a Tool (``as_tool``). That single fact
IS the whole multi-agent story: a "manager" agent whose tools are other agents.
Composition, not a new concept, the way ``nn.Module`` nests inside ``nn.Module``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .context import Context
from .message import ASSISTANT, SYSTEM, TOOL, USER, Message
from .policy import Action, Policy
from .tool import Tool


@dataclass(frozen=True)
class Step:
    """A record of one turn of the loop, kept for inspection."""

    action: Action
    observation: str
    terminal: bool


class Agent:
    """The loop, made reusable and nestable."""

    def __init__(
        self,
        policy: Policy,
        tools: Iterable[Tool],
        *,
        system: str | None = None,
        max_steps: int = 8,
        name: str = "agent",
        description: str = "",
    ) -> None:
        self.policy = policy
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.system = system
        self.max_steps = max_steps
        self.name = name
        self.description = description

    def step(self, context: Context) -> tuple[Context, Step]:
        """Run exactly one turn and return the grown Context plus a record."""
        action = self.policy.decide(context, self.tools)

        tool = self.tools.get(action.tool)
        if tool is None:
            observation = (
                f"[error] unknown tool '{action.tool}'. available tools: {sorted(self.tools)}"
            )
            terminal = False
        else:
            observation = tool(**action.args)
            terminal = tool.terminal

        # Append-only: record the decision, then the observation.
        thought = action.reasoning or f"call {action.tool}({action.args})"
        context = context.add(
            Message(ASSISTANT, thought, name=self.name),
            Message(TOOL, observation, name=action.tool),
        )
        return context, Step(action=action, observation=observation, terminal=terminal)

    def run(self, user_input: str, context: Context | None = None) -> Context:
        """Drive the loop to completion and return the full history.

        The final answer is the ``content`` of the last message when a terminal
        tool fired. Use :meth:`answer` to get just that string.
        """
        ctx = context or Context()
        if self.system:
            ctx = ctx.add(Message(SYSTEM, self.system, name=self.name))
        ctx = ctx.add(Message(USER, user_input))

        for _ in range(self.max_steps):
            ctx, step = self.step(ctx)
            if step.terminal:
                break
        return ctx

    def answer(self, user_input: str, context: Context | None = None) -> str:
        """Convenience: run and return only the final observation string."""
        final = self.run(user_input, context)
        last = final.last()
        return last.content if last else ""

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this whole Agent as a single Tool another Agent can call."""

        def run_sub(input: str) -> str:
            return self.answer(input)

        return Tool(
            name=name or self.name,
            description=description or self.description or f"Delegate to the {self.name} agent.",
            fn=run_sub,
        )
