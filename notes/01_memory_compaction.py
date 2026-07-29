"""Production note: memory that does not grow forever.

The problem, in one picture. Every turn adds notes to the notebook, and the
notebook never gets erased. That is lovely for reading back a run, but a real
agent that runs for hours would fill the notebook until it is too big to hand
to the model, too slow, and too expensive.

The fix is the same thing you do with a messy desk. You do not keep every old
sticky note. Every so often you sweep the old ones into a single summary note
and keep only that plus the few most recent ones.

In our world that is one function: Context in, a shorter Context out. Because a
Context is never edited in place, compaction just builds a new, smaller one.
Nothing in the core changes. This is a pattern you add on top, not a new block.

Run:  python notes/01_memory_compaction.py
"""

from microagent import Context, Message


def default_summary(messages) -> str:
    """A stand-in summarizer. In real life a model would write this."""
    pieces = [f"{m.name or m.role}: {m.content}" for m in messages]
    joined = " | ".join(pieces)
    return joined if len(joined) <= 200 else joined[:197] + "..."


def compact(ctx: Context, keep_last: int = 2, summarize=default_summary) -> Context:
    """Replace the old notes with one summary note, keep the recent ones."""
    if len(ctx) <= keep_last:
        return ctx
    old = ctx.messages[:-keep_last]
    recent = ctx.messages[-keep_last:]
    summary = Message("system", f"Summary of earlier steps: {summarize(old)}")
    return Context((summary,) + recent)


if __name__ == "__main__":
    # Pretend an agent has been running a while and the notebook is long.
    ctx = Context()
    for i in range(1, 9):
        ctx = ctx.add(Message("user", f"question {i}"))
        ctx = ctx.add(Message("tool", f"answer {i}", name="worker"))

    print(f"before: {len(ctx)} notes in the notebook")

    smaller = compact(ctx, keep_last=2)

    print(f"after:  {len(smaller)} notes")
    print()
    print("the compacted notebook now reads:")
    for m in smaller:
        print(f"  [{m.name or m.role}] {m.content}")
