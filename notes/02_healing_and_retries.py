"""Production note: how an agent recovers from mistakes.

Real tools fail. A network blips. A model picks the wrong worker. In an office,
a good assistant does not freeze when something goes wrong. They notice, and
they try again or try another way.

There are two kinds of healing, and neither one needs a new building block.

1. Free healing. When a tool errors, we write the error into the notebook as a
   result. On the next turn the brain sees its own mistake and can fix it. The
   loop we already built gives us this for nothing.

2. Retries. For a worker who is just flaky, we wrap them so a failed call is
   tried again before we give up. This is the same wrapping trick as the
   approval gate. A tool that wraps a tool.

Run:  python notes/02_healing_and_retries.py
"""

from microagent import Action, Agent, Context, FunctionPolicy, Tool, final_answer_tool


# --- kind 2: a retry wrapper for a flaky worker ------------------------------
def with_retry(inner: Tool, attempts: int = 3) -> Tool:
    def retried(**kwargs):
        last_error = None
        for _ in range(attempts):
            try:
                return inner.fn(**kwargs)
            except Exception as error:
                last_error = error
        return f"[failed after {attempts} tries] {last_error}"

    return Tool(inner.name, inner.description, retried, inner.terminal)


# A worker that fails the first two times, then works. Like a busy phone line.
_calls = {"n": 0}


def flaky_fetch(url: str) -> str:
    _calls["n"] += 1
    if _calls["n"] < 3:
        raise ConnectionError("timed out")
    return f"contents of {url}"


fetch = Tool("fetch", "Fetch a url. Sometimes flaky.", flaky_fetch)
safe_fetch = with_retry(fetch, attempts=3)


# --- kind 1: the brain sees an error and corrects itself ---------------------
def self_correcting_brain(ctx: Context, tools):
    last = ctx.last()
    if last is not None and "contents of" in last.content:
        return Action("final_answer", {"answer": last.content}, "got the page, report it")
    # First move: try a worker that does not exist, on purpose.
    if last is None or last.role == "user":
        return Action("downlaod", {"url": "example.com"}, "typo, this worker does not exist")
    # The notebook now shows an 'unknown tool' error, so switch to the real one.
    return Action("fetch", {"url": "example.com"}, "that failed, use the correct worker")


if __name__ == "__main__":
    print("Kind 2: retries on a flaky worker")
    print(" ", safe_fetch(url="example.com"), f"(after {_calls['n']} attempts)")
    print()

    print("Kind 1: the brain sees its own error and corrects")
    agent = Agent(
        FunctionPolicy(self_correcting_brain),
        tools=[safe_fetch, final_answer_tool()],
        max_steps=5,
    )
    from microagent.trace import show

    show(agent.run("get me example.com"))
