"""Production note: is the agent any good?

You cannot improve what you do not measure. Before you trust an agent, you want
to know how often it gets the right answer, and ideally whether it went about it
the right way.

This is where the notebook you never erase pays off. Every run leaves a full
record of what happened. So checking an agent is just grading a test. You keep
an answer key, run the agent on each question, and score the results.

We can grade two things:

1. The outcome. Did the final answer contain what we expected?
2. The working. Did it call the tool we expected along the way? The full trace
   is right there, so we can check the steps, not just the answer.

Run:  python notes/03_evaluation.py
"""

from microagent import Action, Agent, Context, FunctionPolicy, final_answer_tool, tool


@tool(description="Multiply two integers.")
def multiply(a: int, b: int) -> int:
    return a * b


@tool(description="Add two integers.")
def add(a: int, b: int) -> int:
    return a + b


def brain(ctx: Context, tools):
    last = ctx.last()
    if last is not None and last.name in ("multiply", "add"):
        return Action("final_answer", {"answer": f"The answer is {last.content}."})
    text = ctx.messages[-1].content
    if "times" in text:
        return Action("multiply", {"a": 6, "b": 7})
    return Action("add", {"a": 20, "b": 22})


def build_agent() -> Agent:
    return Agent(FunctionPolicy(brain), tools=[multiply, add, final_answer_tool()], max_steps=4)


# The answer key. Each case has the question, the expected answer, and the
# tool we expect the agent to use to get there.
CASES = [
    {"input": "what is 6 times 7", "expects": "42", "should_call": "multiply"},
    {"input": "add 20 and 22", "expects": "42", "should_call": "add"},
    {"input": "what is 6 times 7 please", "expects": "42", "should_call": "multiply"},
]


def evaluate(agent: Agent, cases) -> None:
    outcome_passes = 0
    working_passes = 0

    for case in cases:
        trace = agent.run(case["input"])
        final = trace.last().content
        tools_used = {m.name for m in trace if m.role == "tool"}

        outcome_ok = case["expects"] in final
        working_ok = case["should_call"] in tools_used
        outcome_passes += outcome_ok
        working_passes += working_ok

        mark = "pass" if outcome_ok and working_ok else "FAIL"
        print(f"[{mark}] {case['input']!r}")
        print(f"       answer ok: {outcome_ok}, used {case['should_call']}: {working_ok}")

    n = len(cases)
    print()
    print(f"outcome score: {outcome_passes}/{n}   working score: {working_passes}/{n}")


if __name__ == "__main__":
    evaluate(build_agent(), CASES)
