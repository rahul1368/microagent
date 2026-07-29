# %% [markdown]
# # Build microagent from scratch
#
# This notebook builds a working AI agent from nothing, one small piece at a
# time. By the end you will have written the same five building blocks that live
# in `src/microagent`, and you will understand every line of them.
#
# The trick to understanding agents is to stop thinking about AI for a minute
# and think about a small office.
#
# Picture a new assistant on their first day. They have a notebook, a few
# coworkers they can ask for help, and one simple habit: keep working until the
# job is done. That is the whole idea. Everything below is just that office,
# written in code.
#
# We will build it in five pieces:
#
# 1. **Message** a single note
# 2. **Context** the notebook that holds every note
# 3. **Tool** a coworker you can call for help
# 4. **Policy** the brain that decides what to do next
# 5. **Agent** the assistant that ties it all together and keeps working

# %% [markdown]
# ## 1. Message: a single note
#
# The smallest thing in our office is one note. Someone asks a question. The
# assistant jots down a decision. A coworker hands back an answer. Each of those
# is one note.
#
# A note is tiny. It only needs to say who wrote it and what it says. We also
# make it **frozen**, which is a fancy word for "written in pen". Once a note is
# written, it never changes. That will matter a lot in a moment.

# %%
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str            # who wrote it: "user", "assistant", or "tool"
    content: str         # what it says
    name: str | None = None   # optional label, like which coworker replied


note = Message("user", "what is 6 times 7?")
print(note.role, "->", note.content)

# %% [markdown]
# ## 2. Context: the notebook
#
# One note on its own is useless. The assistant needs the whole running history:
# the question, every decision, every answer so far. That history is the
# **Context**. Think of it as the office notebook.
#
# Here is the one rule that makes everything else simple. You never erase a page.
# You only ever add a new note to the end. The notebook only grows.
#
# Why does that matter? Because it means the full story of what happened is
# always sitting right there. You can read it back, you can replay it, you can
# debug it. Nothing is hidden. When we build the assistant, "remembering" is not
# a feature we add. It is just the notebook we never erase.

# %%
@dataclass(frozen=True)
class Context:
    messages: tuple = ()

    def add(self, *msgs):
        # Note the trick: we do not change this notebook, we hand back a new
        # one with the extra notes at the end. The old one is untouched.
        return Context(self.messages + tuple(msgs))

    def last(self):
        return self.messages[-1] if self.messages else None

    def __iter__(self):
        return iter(self.messages)


history = Context().add(Message("user", "what is 6 times 7?"))
history = history.add(Message("assistant", "let me work that out"))
for m in history:
    print(f"[{m.role}] {m.content}")

# %% [markdown]
# ## 3. Tool: a coworker you can call
#
# The assistant cannot do everything alone. Sometimes they need a specialist.
# Need a number crunched? Call the person with the calculator. Need a file
# fetched? Call the person who knows the filing cabinet.
#
# A **Tool** is exactly that: a coworker you can call. You hand them a request,
# they hand back a result. That is the only way our assistant ever touches the
# real world, which is handy, because it means there is one clear place to put a
# safety check later.
#
# One tool is special. It is the coworker whose job is to say "we are done here,
# the answer is X". We mark that one as `terminal`, because calling it ends the
# work.

# %%
@dataclass
class Tool:
    name: str
    description: str
    fn: object                 # the actual function to run
    terminal: bool = False     # does calling this end the job?

    def __call__(self, **kwargs):
        return str(self.fn(**kwargs))


multiply = Tool("multiply", "Multiply two numbers.", lambda a, b: a * b)
final_answer = Tool("final_answer", "Give the answer and stop.", lambda answer: answer, terminal=True)

print(multiply(a=6, b=7))

# %% [markdown]
# ## 4. Policy: the brain
#
# Now the interesting part. Who decides what to do next? Look at the notebook,
# and pick the next move: which coworker to call, and with what request. That
# decision maker is the **Policy**. It is the brain of the office.
#
# A decision has two parts: which tool to call, and the arguments to call it
# with. We put that in a small `Action`.
#
# Here is the most important idea in the whole project. The brain is swappable.
# Today we will write it as a few lines of plain Python, no AI at all, just so we
# can see the office run. Tomorrow you swap in a real language model, or even a
# small neural network you trained yourself. The office does not change. Only the
# brain does. Swap the brain, keep the body.

# %%
@dataclass(frozen=True)
class Action:
    tool: str
    args: dict = field(default_factory=dict)
    reasoning: str = ""


# A brain made of a few if-statements. It is not smart, but it is a real brain:
# it looks at the notebook and decides the next move.
def simple_brain(context, tools):
    last = context.last()
    if last is not None and last.name == "multiply":
        # The calculator already answered, so report it and stop.
        return Action("final_answer", {"answer": f"The product is {last.content}."},
                      reasoning="I have the number, time to report it")
    # Nothing computed yet, so do the multiplication first.
    return Action("multiply", {"a": 6, "b": 7},
                  reasoning="I need the product before I can answer")

# %% [markdown]
# ## 5. Agent: the assistant who keeps working
#
# Finally we tie it together. The **Agent** is the assistant. It holds the brain,
# it knows its coworkers, and it follows one simple habit in a loop:
#
# 1. look at the notebook
# 2. ask the brain for the next move
# 3. call that coworker and get a result
# 4. write both the decision and the result into the notebook
# 5. repeat, until the "we are done" coworker is called
#
# That loop is the entire engine of every AI agent you have ever heard of. Big
# frameworks dress it up, but underneath, this is it.
#
# We also add a `max_steps` limit, the same way you would not let a new assistant
# spin forever on one task. If they have not finished after a few tries, they
# stop and hand back what they have.

# %%
class Agent:
    def __init__(self, brain, tools, max_steps=6, name="assistant"):
        self.brain = brain
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.name = name

    def run(self, user_input):
        # Start the notebook with the user's question.
        ctx = Context().add(Message("user", user_input))

        for _ in range(self.max_steps):
            action = self.brain(ctx, self.tools)      # 1 and 2: decide
            tool = self.tools[action.tool]
            result = tool(**action.args)              # 3: act

            ctx = ctx.add(                            # 4: write it all down
                Message("assistant", action.reasoning, name=self.name),
                Message("tool", result, name=action.tool),
            )
            if tool.terminal:                         # 5: are we done?
                break
        return ctx


agent = Agent(simple_brain, tools=[multiply, final_answer], name="calculator")
result = agent.run("what is 6 times 7?")

# %% [markdown]
# ### Read back what happened
#
# Because the notebook was never erased, we can read the whole story top to
# bottom. This little printer is all our trace reader really is.

# %%
def tell_the_story(ctx):
    for m in ctx:
        if m.role == "user":
            print(f'you asked: "{m.content}"')
        elif m.role == "assistant":
            print(f"  {m.name} thought: {m.content}")
        elif m.role == "tool":
            print(f"    -> {m.name} returned: {m.content}")


tell_the_story(result)

# %% [markdown]
# ## Bonus 1: an agent can be a coworker too
#
# Here is where it gets powerful, and it costs us nothing. An assistant can also
# be a coworker for a bigger boss. In a company, a manager runs a team, and that
# manager is themselves a worker for a director.
#
# In our code, that means an Agent can be wrapped up and handed to another Agent
# as just another Tool. That single idea is the whole of "multi-agent systems".
# There is no new building block. It is the five pieces we already have.

# %%
def make_agent_a_tool(inner_agent, name, description):
    def run_inner(request):
        final_notebook = inner_agent.run(request)
        return final_notebook.last().content
    return Tool(name, description, run_inner)


# The manager's only skill is delegating to the calculator assistant.
calculator_as_coworker = make_agent_a_tool(agent, "calculator", "Does math for you.")


def manager_brain(context, tools):
    last = context.last()
    if last is not None and last.name == "calculator":
        return Action("final_answer", {"answer": f"My calculator says: {last.content}"},
                      reasoning="pass the delegated answer up")
    return Action("calculator", {"request": "what is 6 times 7?"},
                  reasoning="I will let the calculator handle the math")


manager = Agent(manager_brain, tools=[calculator_as_coworker, final_answer], name="manager")
tell_the_story(manager.run("get me the math done"))

# %% [markdown]
# ## Bonus 2: safety is just a coworker with a gate
#
# What if a coworker can do something risky, like deleting a file? You do not
# want the brain firing that off on its own. In an office, a big purchase needs a
# manager's sign-off first.
#
# We do the same thing, and again it is not a new building block. We wrap the
# risky tool in another tool that asks for approval first. If approval is denied,
# the real action never runs.

# %%
def with_approval(inner_tool, ask):
    def guarded(**kwargs):
        if not ask(inner_tool.name, kwargs):
            return f"[denied] '{inner_tool.name}' was not approved"
        return inner_tool.fn(**kwargs)
    return Tool(inner_tool.name, inner_tool.description, guarded, inner_tool.terminal)


delete_file = Tool("delete_file", "Delete a file. Dangerous.", lambda path: f"deleted {path}")

# An approver that refuses anything under /important.
def human_check(name, args):
    ok = "/important" not in args.get("path", "")
    print(f"[approval] {name}({args}) -> {'ALLOW' if ok else 'DENY'}")
    return ok


safe_delete = with_approval(delete_file, human_check)
print(safe_delete(path="/important/data.db"))   # refused
print(safe_delete(path="/tmp/scratch.txt"))      # allowed

# %% [markdown]
# ## That is the whole thing
#
# You just built a real agent from five small parts:
#
# - **Message** a note
# - **Context** the notebook you never erase
# - **Tool** a coworker you can call
# - **Policy** the brain that picks the next move
# - **Agent** the assistant that keeps working until done
#
# And you saw that the scary sounding stuff is not new building blocks:
#
# - multi-agent is just an agent used as a coworker
# - safety is just a coworker with an approval gate
#
# The versions in `src/microagent` are these exact ideas, only tidied up with
# type hints, docstrings, and a proper swappable brain. Now that you have built
# them yourself, none of that code is a mystery.
#
# Next step: open `src/microagent/policy.py` and swap `simple_brain` for a real
# model. The office will not notice the difference.
