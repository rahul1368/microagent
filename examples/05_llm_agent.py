"""Example 05, the same agent, but with a real LLM brain.

The point (the whole thesis): compared to the earlier examples, ONLY the Policy
changes. The tools, the loop, the Context, the Agent, all identical. Swap the
brain, keep the body.

Requires a local Ollama:  https://ollama.com
    ollama pull llama3.1
    ollama serve

Run:  python examples/05_llm_agent.py
"""

from microagent import Agent, final_answer_tool, tool
from microagent.adapters.ollama import OllamaPolicy


@tool(description="Multiply two integers a and b.")
def multiply(a: int, b: int) -> int:
    return a * b


@tool(description="Add two integers a and b.")
def add(a: int, b: int) -> int:
    return a + b


def build(model: str = "llama3.1") -> Agent:
    return Agent(
        policy=OllamaPolicy(model=model),  # <-- the ONLY difference from example 02
        tools=[multiply, add, final_answer_tool()],
        name="math-llm",
        max_steps=6,
    )


if __name__ == "__main__":
    agent = build()
    try:
        print(agent.answer("What is 23 times 19, then add 100?"))
    except RuntimeError as exc:
        print(f"(skipped, {exc})")
