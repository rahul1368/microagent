# How to read this project

The whole thing is small on purpose. You can read all of it in one sitting.
Here is the order that will make it click fastest.

## Start here

1. **[`notebooks/build_microagent_from_scratch.py`](../notebooks/build_microagent_from_scratch.py)**
   Build the five pieces yourself, from nothing, with a running example at each
   step. Open it as a notebook in VS Code, or just run it as a script. Start
   here even if you plan to read the source. It gives you the mental picture
   everything else hangs on.

## Then the source, in this order

Read the files in the order the ideas build on each other:

2. **[`message.py`](../src/microagent/message.py)** a single note. The smallest thing.
3. **[`context.py`](../src/microagent/context.py)** the notebook of notes that you only ever add to.
4. **[`tool.py`](../src/microagent/tool.py)** a coworker the agent can call. Also where safety lives (`with_approval`).
5. **[`policy.py`](../src/microagent/policy.py)** the brain that picks the next move. The part you swap.
6. **[`agent.py`](../src/microagent/agent.py)** the loop that ties it all together, and can act as a tool itself.

That is the entire core. Everything after this is optional.

## Optional extras

7. **[`trace.py`](../src/microagent/trace.py)** prints a run as a plain story you can read top to bottom.
8. **[`adapters/ollama.py`](../src/microagent/adapters/ollama.py)** a real language model brain, using only the standard library.

## See it run

9. **[`examples/`](../examples/)** small scripts, one idea each. Numbers 01 to 04
   run offline with no model. Number 05 uses a real model through Ollama.

## Go deeper on the why

10. **[`DESIGN.md`](../DESIGN.md)** why these five pieces and not more, how this
    compares to the big frameworks, and the questions still left open.
