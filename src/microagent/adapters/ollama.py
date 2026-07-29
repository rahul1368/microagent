"""OllamaPolicy, a real LLM brain, using only the Python standard library.

This proves the central thesis of microagent: the Agent never changes. To go
from a hand-written rule Policy to a genuine LLM, you swap ONE object. Nothing
else in the loop moves.

Requirements: a local Ollama running (https://ollama.com), e.g.::

    ollama pull llama3.1
    ollama serve            # usually already running

We deliberately use only ``json`` and ``urllib`` (stdlib) so installing
microagent never pulls in an HTTP or SDK dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..context import Context
from ..message import ASSISTANT, SYSTEM, TOOL, USER
from ..policy import Action, Policy
from ..tool import Tool

_SYSTEM_TEMPLATE = """You are the decision-making brain of an agent.

On each turn you must choose exactly ONE tool to call. Available tools:
{tool_list}

Reply with ONLY a JSON object, no prose, in this exact shape:
{{"tool": "<tool_name>", "args": {{...}}, "reasoning": "<one short sentence>"}}

Call the terminal tool (e.g. "final_answer") when you have the answer."""


def _role_for_ollama(role: str) -> str:
    # Ollama's chat API understands user/assistant/system, map tool -> user
    # so observations are visible to the model as fresh input.
    return {USER: "user", ASSISTANT: "assistant", SYSTEM: "system", TOOL: "user"}.get(role, "user")


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of a model reply."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model reply: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in model reply: {text!r}")


@dataclass
class OllamaPolicy(Policy):
    """Choose the next Action by asking a local Ollama model."""

    model: str = "llama3.1"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: float = 60.0

    def decide(self, context: Context, tools: Mapping[str, Tool]) -> Action:
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
        system = _SYSTEM_TEMPLATE.format(tool_list=tool_list)

        messages = [{"role": "system", "content": system}]
        for m in context:
            messages.append({"role": _role_for_ollama(m.role), "content": m.content})

        reply = self._chat(messages)
        data = _extract_json(reply)
        return Action(
            tool=data["tool"],
            args=data.get("args", {}) or {},
            reasoning=data.get("reasoning", ""),
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"could not reach Ollama at {self.host}. Is it running? "
                f"(`ollama serve` / `ollama pull {self.model}`)"
            ) from exc
        return body["message"]["content"]
