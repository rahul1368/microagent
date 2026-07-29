"""Production note: doing things at the same time.

Our core does one thing at a time. The assistant calls one worker, waits for
the answer, then calls the next. That is perfect for learning, because you can
follow every step. It is also fine for many real jobs.

But sometimes you need to call three workers at once and not sit waiting for
each in turn. Think of a single cashier versus several tills open together. The
work is the same, but the second way finishes far sooner when the workers spend
their time waiting (on the network, say).

Here is the honest truth about async. It does not need new building blocks. It
needs the same loop, with two small words added: the calls become "await", and
the tools become "async". Below is the whole idea in one tiny async agent, plus
a demo of running two slow workers at the same time.

Run:  python notes/05_going_async.py
"""

import asyncio
import time

from microagent import Context, Message


# An async tool is just a normal tool whose function can await.
async def slow_search(query: str) -> str:
    await asyncio.sleep(0.3)  # pretend this is a network call
    return f"results for {query!r}"


async def slow_lookup(topic: str) -> str:
    await asyncio.sleep(0.3)
    return f"facts about {topic!r}"


# --- the same loop, now with await ------------------------------------------
class AsyncAgent:
    def __init__(self, brain, tools, max_steps=6):
        self.brain = brain
        self.tools = tools
        self.max_steps = max_steps

    async def run(self, user_input: str) -> Context:
        ctx = Context().add(Message("user", user_input))
        for _ in range(self.max_steps):
            action = self.brain(ctx)  # deciding stays instant
            tool = self.tools[action["tool"]]
            result = await tool(**action["args"])  # the only change: await
            ctx = ctx.add(Message("tool", result, name=action["tool"]))
            if action["tool"] == "done":
                break
        return ctx


async def demo_parallel():
    # The real win: run two slow workers at the same time.
    start = time.perf_counter()
    both = await asyncio.gather(
        slow_search(query="agents"),
        slow_lookup(topic="pytorch"),
    )
    elapsed = time.perf_counter() - start
    print("ran two 0.3s workers at the same time")
    print(f"  results: {both}")
    print(f"  total time: {elapsed:.2f}s (not 0.6s, because they overlapped)")


if __name__ == "__main__":
    asyncio.run(demo_parallel())
    print()
    print("The lesson: async is the same five blocks with 'await' added.")
    print("Keep the simple version to learn. Reach for async when workers wait a lot.")
