# Production notes

The core of microagent stays small on purpose. But the real world has hard
problems, and a good teacher does not pretend they do not exist. These notes
show, in the smallest honest way, how each hard problem plugs into the five
blocks you already know.

Each one is a short runnable script. None of them changes the core. They are
patterns you add on top, taught as chapters, not shipped as features.

## The map

These answer the common "but what about..." questions in order.

| Question | Note | Short answer |
|----------|------|--------------|
| What is a "skill"? | (see [`examples/03_multi_agent.py`](../examples/03_multi_agent.py)) | A group of tools, or a whole agent used as a tool. Composition, not a new block. |
| How does memory not blow up? | [`01_memory_compaction.py`](01_memory_compaction.py) | A function: long Context in, shorter Context out. Sweep old notes into one summary. |
| How does an agent recover from failure? | [`02_healing_and_retries.py`](02_healing_and_retries.py) | Feed errors back so the brain sees them, plus a retry wrapper for flaky tools. |
| How do you know it is any good? | [`03_evaluation.py`](03_evaluation.py) | Grade it against an answer key. The saved trace lets you check the working, not just the answer. |
| Where do prompt techniques live? | [`04_prompt_techniques.py`](04_prompt_techniques.py) | Inside the Policy. Zero-shot, few-shot, and ReAct are just different notes to the model. |
| How do you do things at the same time? | [`05_going_async.py`](05_going_async.py) | The same loop with `await` added. Run slow workers together with `asyncio.gather`. |

## Run them

```bash
python notes/01_memory_compaction.py
python notes/02_healing_and_retries.py
python notes/03_evaluation.py
python notes/04_prompt_techniques.py
python notes/05_going_async.py
```

## The point

Notice the pattern across all of them. The scary production problems do not need
new fundamentals. They need good use of the five blocks, plus a small wrapper or
a small function on top. That is the strongest sign the five blocks are the right
ones. It is also exactly why the heavy, hardened versions of these belong in a
real product, not in a framework named micro.
