# Contributing

Thank you for wanting to help. This project has one unusual rule, and it shapes
every decision here, so please read this first.

## The one rule: it stays small

microagent is a teaching project. Its whole value is that a person can read the
core in one sitting and understand every line. The name is a promise. The core
stays micro.

That means the goal of a change is never "add more power". It is always "help
someone understand it faster, or reach more people". A bigger, more capable
microagent would be a worse microagent, because it would stop being the thing
that makes it useful.

If you want a full, production ready agent framework with every feature built
in, that already exists and is very good. Use [LangGraph](https://github.com/langchain-ai/langgraph)
or the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). This
project exists to help you understand what is underneath those.

## The test for a change

Before proposing something, ask which kind it is:

- **Does it help someone understand?** A clearer explanation, a better analogy,
  a small runnable example, a fix to a confusing name, a diagram, a test that
  doubles as documentation. These are very welcome.
- **Does it grow the framework?** A new feature with config, edge cases, and its
  own dependencies that someone would have to maintain. This does not belong
  here, even if it is genuinely useful. That is what a production framework is
  for.

A good yardstick: "here is the smallest example that shows how memory
compaction works" belongs. "A full memory system with three storage backends
and retry logic" does not.

## The core has zero dependencies

`src/microagent` (message, context, tool, policy, agent) must never import a
third party package. That constraint is part of the teaching. Optional extras
(like a model adapter) can bring their own dependencies, but the core never
imports the extras.

## Practical notes

- Keep the writing plain. Short sentences, everyday words, real examples. No
  jargon where a plain word will do.
- Run the tests before you open a pull request: `pip install -e ".[dev]"` then
  `pytest`.
- New behavior comes with a small test. Tests here also serve as examples, so
  write them to be read.

## Good first contributions

- A clearer analogy or explanation anywhere it reads awkwardly.
- A new small example that teaches one idea.
- A "production notes" chapter that shows, in the smallest honest way, how one
  real problem (memory, retries, evaluation) plugs into the five pieces.
