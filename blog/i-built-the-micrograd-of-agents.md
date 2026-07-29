---
title: I built the micrograd of AI agents
published: false
description: A tiny from-scratch agent framework in about 200 lines, so you can actually see how agents work.
tags: ai, python, machinelearning, tutorial
cover_image: https://raw.githubusercontent.com/rahul1368/microagent/main/assets/demo.gif
canonical_url:
---

Every agent framework hides the same five ideas. I rebuilt them in about 200 lines of Python so you can read the whole thing in one sitting and really understand it.

![A support agent handling a refund request, printed as a plain story](https://raw.githubusercontent.com/rahul1368/microagent/main/assets/demo.gif)

## The itch

I use agent frameworks like LangGraph and the OpenAI Agents SDK. They are good. But when you are learning, they can feel like magic. You wire a few things together, an agent appears, and you are not quite sure what happened in the middle.

I did not want more magic. I wanted to see the machine.

So I did the thing that worked so well for me with neural networks. If you have watched Andrej Karpathy build [micrograd](https://github.com/karpathy/micrograd), you know the feeling. He rebuilt the core idea of PyTorch in about 150 lines, not to replace it, but to make it impossible to misunderstand. After that, PyTorch stops being magic.

I wanted the same thing for agents. So I built [microagent](https://github.com/rahul1368/microagent).

## The trick: stop thinking about AI for a minute

The fastest way to understand an agent is to picture a small office.

A new assistant starts on their first day. They have a notebook, a few coworkers they can ask for help, and one simple habit. Keep working until the job is done.

That is the whole idea. Everything below is just that office, written in code. It comes down to five small pieces.

## 1. Message, a single note

The smallest thing in the office is one note. Someone asks a question. The assistant writes down a decision. A coworker hands back an answer. Each of those is one note.

```python
@dataclass(frozen=True)
class Message:
    role: str            # "user", "assistant", or "tool"
    content: str
    name: str | None = None
```

It is frozen, which is a fancy word for written in pen. Once a note exists, it never changes. That matters in a second.

## 2. Context, the notebook

One note is useless on its own. The assistant needs the whole running history. That history is the Context. Think of it as the office notebook.

Here is the one rule that makes everything simple. You never erase a page. You only add a new note to the end.

```python
@dataclass(frozen=True)
class Context:
    messages: tuple = ()

    def add(self, *msgs):
        # We do not change this notebook. We hand back a new one.
        return Context(self.messages + tuple(msgs))

    def last(self):
        return self.messages[-1] if self.messages else None
```

Because the notebook only grows, the full story of what happened is always right there. You can read it back, replay it, and debug it. Nothing is hidden. Memory is not a feature we add later. It is just the notebook we never erase.

## 3. Tool, a coworker you can call

The assistant cannot do everything alone. Need a number crunched? Call the person with the calculator. Need a file fetched? Call the person who knows the filing cabinet.

A Tool is exactly that. You hand it a request, it hands back a result.

```python
@dataclass
class Tool:
    name: str
    description: str
    fn: object                 # the function to run
    terminal: bool = False     # does calling this end the job?

    def __call__(self, **kwargs):
        return str(self.fn(**kwargs))
```

One tool is special. It is the coworker whose job is to say we are done, the answer is X. We mark that one as terminal.

## 4. Policy, the brain

Now the interesting part. Who decides what to do next? Look at the notebook, pick the next move. That decision maker is the Policy. It is the brain.

A decision has two parts, which tool to call and the arguments to call it with.

```python
@dataclass(frozen=True)
class Action:
    tool: str
    args: dict = field(default_factory=dict)
    reasoning: str = ""
```

Here is the most important idea in the whole project. The brain is swappable. Today you can write it as a few lines of plain Python, no AI at all, just to watch the office run. Tomorrow you swap in a real language model, or even a small neural network you trained yourself. The office does not change. Only the brain does.

Swap the brain, keep the body.

## 5. Agent, the assistant who keeps working

Finally we tie it together. The Agent holds the brain, knows its coworkers, and follows one habit in a loop.

```python
class Agent:
    def __init__(self, brain, tools, max_steps=6):
        self.brain = brain
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, user_input):
        ctx = Context().add(Message("user", user_input))
        for _ in range(self.max_steps):
            action = self.brain(ctx, self.tools)      # decide
            tool = self.tools[action.tool]
            result = tool(**action.args)              # act
            ctx = ctx.add(                            # write it down
                Message("assistant", action.reasoning),
                Message("tool", result, name=action.tool),
            )
            if tool.terminal:                         # are we done?
                break
        return ctx
```

That loop is the entire engine of every AI agent you have heard of. Decide, act, write it down, repeat. Big frameworks dress it up, but underneath, this is it.

## Watch it do real work

Here is a support agent handling a refund. It looks up the order, checks the policy, issues the refund after a human signs off, and replies to the customer. Same five pieces, pointed at real tools.

```
you asked: "Can I get a refund for order A1234? It arrived broken."
  support thought: let me pull up the order first
    -> look_up_order returned: A1234, Wireless Headphones, $79, delivered, reported defective
  support thought: now check if it qualifies
    -> check_policy returned: eligible, within the 30 day window and reported defective
  support thought: it qualifies, and a refund needs a human to sign off
    -> issue_refund returned: approved by a human, $79 refunded to the original card
  support thought: all set, time to reply to the customer
    -> final_answer returned: Good news, A1234 qualifies. I refunded $79 to your card, here in 3 to 5 days.
```

Notice the third step. A risky action waited for a human to approve. That is not a special feature. It is a tool that wraps another tool.

## The scary stuff is not new pieces

This is the part I did not expect, and it is the best reason to build small.

When you try to add the hard, real world features, none of them need a new building block.

- Multi-agent is just an agent used as a coworker. An agent can be handed to another agent as a tool. That is the whole of multi-agent systems.
- Safety is just a coworker with a gate. Wrap a risky tool in one that asks for approval first.
- Memory that does not blow up is one function. A long notebook in, a shorter summarized notebook out.
- Recovering from errors is mostly free. Write the error into the notebook, and the brain sees its own mistake on the next turn and fixes it.

Every one of these is the same five pieces, arranged a little differently. When the hard problems do not demand new fundamentals, that is a strong sign the five pieces are the right ones.

## The honest part

microagent is not a production framework, and it does not try to be. If you need to ship, use [LangGraph](https://github.com/langchain-ai/langgraph) or the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). They are excellent, and they solve the messy real world problems that a teaching project should not pretend to.

The whole point of microagent is different. Read every line, understand every piece, then go build the real thing knowing exactly what is underneath it.

## Try it

The repo is here: [github.com/rahul1368/microagent](https://github.com/rahul1368/microagent).

The best way in is the from-scratch notebook. It rebuilds all five pieces one at a time, with the office story, and runs at every step. After that, the code will feel like something you already know, because you will have written it yourself.

If this helped, a star on the repo means a lot, and I would love to hear which part finally made agents click for you.
