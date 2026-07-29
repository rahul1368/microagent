"""Message, the atomic unit of information in an agent system.

Everything that happens is a Message: a user's request, the model's decision,
a tool call, an observation from the world. Messages are immutable, so the
history is append-only and therefore replayable and debuggable, the same
property that makes a PyTorch computation graph inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The four roles every message plays. Kept as plain strings (not an enum) so
# the primitive stays trivially serializable and easy to read in a log.
USER = "user"
ASSISTANT = "assistant"
TOOL = "tool"
SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    """One immutable entry in an agent's history.

    Attributes:
        role: who/what produced it, ``user``, ``assistant``, ``tool``, ``system``.
        content: the human-readable payload.
        name: optional label, e.g. the tool that produced this observation.
        meta: free-form metadata (never used for control flow, just for humans).
    """

    role: str
    content: str
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        who = self.name or self.role
        return f"[{who}] {self.content}"
