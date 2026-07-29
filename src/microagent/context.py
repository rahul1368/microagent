"""Context, an ordered, append-only log of Messages.

Context is everything the agent knows *right now*. It is the value that flows
through the system, so it plays the role that the Tensor plays in PyTorch.

It is immutable on purpose: ``add`` returns a NEW Context rather than mutating
the old one. Because history only ever grows, every step of an agent's
reasoning is preserved, inspectable, and replayable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .message import Message


@dataclass(frozen=True)
class Context:
    """An immutable sequence of Messages."""

    messages: tuple[Message, ...] = ()

    def add(self, *msgs: Message) -> Context:
        """Return a new Context with ``msgs`` appended. The original is unchanged."""
        return Context(self.messages + tuple(msgs))

    def last(self) -> Message | None:
        """The most recent Message, or None if the Context is empty."""
        return self.messages[-1] if self.messages else None

    @classmethod
    def of(cls, *msgs: Message) -> Context:
        """Build a Context directly from some Messages."""
        return cls(tuple(msgs))

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.messages)
