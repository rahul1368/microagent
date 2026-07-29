"""Policy, the decider.

Given the current Context and the available Tools, a Policy chooses the next
Action. It is the *brain*, and it is deliberately swappable: it can be a
hand-written rule, an LLM, or a small neural net you trained yourself. The
Agent does not care which, it only asks the Policy "what next?".

This is the seam where your own PyTorch work plugs in: a trained intent/router
net is just another Policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .context import Context
from .tool import Tool


@dataclass(frozen=True)
class Action:
    """A decision: call tool ``tool`` with keyword ``args``.

    ``reasoning`` is an optional human-readable note about *why*, it is
    recorded in the Context but never used for control flow.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class Policy(ABC):
    """Abstract base for every brain."""

    @abstractmethod
    def decide(self, context: Context, tools: Mapping[str, Tool]) -> Action:
        """Look at the world so far and choose the next Action."""
        raise NotImplementedError


@dataclass
class FunctionPolicy(Policy):
    """Adapt a plain function into a Policy.

    Handy for tests, examples, and rule-based agents where the "brain" is a
    few lines of Python instead of a model call.
    """

    fn: Callable[[Context, Mapping[str, Tool]], Action]

    def decide(self, context: Context, tools: Mapping[str, Tool]) -> Action:
        return self.fn(context, tools)
