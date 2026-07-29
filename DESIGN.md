# DESIGN: the building blocks of an agent framework

This file is the thinking behind microagent. The code is small on purpose. The
thinking is the real work.

---

## 1. How you find the building blocks

You do not guess them. You look for the smallest set that you can build
everything in the domain from. PyTorch passed this test. You can build a CNN, an
RNN, and a Transformer from just three things: `Tensor`, `autograd`, and
`Module`. So those three are the right building blocks.

We use the same test for agents. Pick a small set, then try to build every kind
of agent from it. If they all come out cleanly, the set is good. If one of them
forces a brand new idea, the set is missing something.

## 2. The five building blocks

| Block | One line | Shape | PyTorch parallel |
|-------|----------|-------|-----------------|
| `Message` | one piece of information | `{role, content, name?}` | scalar |
| `Context` | a growing, never-edited list of messages | `Message[]` | Tensor |
| `Tool` | a function that looks something up or acts | `Input -> Result` | basic operation |
| `Policy` | the brain that picks the next action | `Context -> Action` | the swappable part |
| `Agent` | a Policy, some Tools, and a loop. Also usable as a Tool | `loop(...)` that is also a Tool | `nn.Module` |

The thing that ties them together is one small function, a `Step`:

```
step(context):
    action = Policy(context)                 # decide
    result = run(action, Tools)              # act
    return context + [action, result]        # add both to the history
```

An `Agent` just repeats that step until a final tool is called, or until it hits
a step limit.

### Why the history is never edited

`Context` only grows. It never gets changed in place. This one choice means every
step the agent took is still there afterwards. You can read it, replay it, and
debug it. This is the same reason a PyTorch graph is easy to inspect. It also
matches how an LLM already reads a list of messages, so nothing feels forced.

## 3. The test: can you build everything from five blocks

Here is every common kind of agent, built from the same five blocks:

- **Chatbot.** An Agent with a brain and one `respond` tool. Works.
- **Tool-using agent.** An Agent with a brain and tools like `search` and `calc`. Works.
- **RAG.** Add a `retrieve` tool and let the brain read its result. No new block. Works.
- **Multi-agent.** An Agent whose tools are other Agents, because an Agent can be used as a Tool. Works.
- **Human approval and safety.** A Tool that wraps another Tool (`with_approval`). No new block. Works.
- **Memory.** Either part of Context, or a `memory.read` and `memory.write` tool. Works.

All six come out of the same five blocks. None of them needed a sixth. That is a
good sign the set is right. The most likely future addition is a separate
`Planner` block for laying out a multi-step plan up front. See the open questions.

## 4. This is the common ground, not a new invention

I checked this on purpose before building. Every one of these blocks already
ships inside real frameworks. microagent is the boiled-down version, not a new
idea:

| microagent block | already exists as |
|------------------|-------------------|
| `Context` (never edited) | LangGraph state with its `add_messages` merge, AutoGen chat history |
| `Tool` | the same everywhere, in every framework |
| `Policy` | "instructions plus a model". Smolagents has the brain write code instead of JSON |
| `Agent` | OpenAI Agents SDK: instructions, a model, tools, and handoffs |
| `Agent` used as a `Tool` | OpenAI SDK handoffs, CrewAI crews, AutoGen conversations, LangGraph graph nodes |
| `with_approval` | OpenAI SDK guardrails |

What this means: the goal is not to invent better blocks. That ground is taken,
and it is backed by OpenAI, Google, Hugging Face, and LangChain. The open goal is
the micrograd one. Be the clearest from-scratch version for people who want to
understand these blocks. That is what this repo is for.

## 5. Where your own trained brain fits

The `Policy` is the swap point. A `Policy` is anything that takes a `Context` and
returns an `Action`:

- a plain rule (`FunctionPolicy`), used in the offline examples
- an LLM (`adapters/ollama.OllamaPolicy`)
- a small neural net you trained yourself, for example a router that guesses
  which tool a request needs, so you do not have to call an LLM every turn

Swapping any of these leaves the Agent, the tools, and the loop untouched.

## 6. Questions still open

A first version does not settle these. Working through them is the real work, and
building a real product on top tends to answer them:

1. **Is `Policy` really separate from `Tool`?** Or is "think" just a special tool
   that returns a message? Merging them is cleaner but blurs deciding and doing.
   We kept them apart so it reads clearly.
2. **Where does memory that outlives a run live?** `Context` covers one run. For
   memory that survives a restart, is that a tool, or a new block? Start with a
   tool and watch for strain.
3. **Should the loop stay simple, or go async?** Real agents stream text and run
   tools at the same time. microagent stays simple and one-step-at-a-time so it
   stays readable. That is a real trade-off, so decide it early.
4. **Who owns the control flow?** The `Agent` loop, or a separate `Planner`? This
   is the most likely honest sixth block.

## 7. What this does not cover

No streaming, no async, no saving state, no retries, no auto-tuning. Those are
real and they matter. They are also exactly what production frameworks are for.
microagent stops right at the point where you understand the ideas, so the next
framework you read or build has nothing hidden in it.
