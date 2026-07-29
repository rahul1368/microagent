"""microagent, the micrograd of AI agents.

Five primitives, ~200 lines, zero dependencies. Not a production framework
(use LangGraph / the OpenAI Agents SDK for that), a from-scratch reference you
can read end to end in one sitting and truly understand how agents work.

    Message   the atomic unit of information            (the scalar)
    Context   an append-only log of Messages            (the Tensor, value that flows)
    Tool      a function the agent can call             (a leaf operation)
    Policy    the brain that chooses the next Action    (the swappable learned function)
    Agent     Policy + Tools + a loop, is itself a Tool (nn.Module, composition)
"""

from .agent import Agent, Step
from .context import Context
from .message import ASSISTANT, SYSTEM, TOOL, USER, Message
from .policy import Action, FunctionPolicy, Policy
from .tool import Tool, final_answer_tool, tool, with_approval

__version__ = "0.0.1"

__all__ = [
    "Message",
    "USER",
    "ASSISTANT",
    "TOOL",
    "SYSTEM",
    "Context",
    "Tool",
    "tool",
    "with_approval",
    "final_answer_tool",
    "Policy",
    "Action",
    "FunctionPolicy",
    "Agent",
    "Step",
    "__version__",
]
