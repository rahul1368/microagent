"""Production note: where prompt techniques live.

You hear a lot of names. Zero-shot, few-shot, chain of thought, ReAct. They can
sound like separate inventions. They are not. Every one of them is just a
different way of writing the note you hand to the model.

And in our design there is exactly one place that writes that note: the Policy,
the brain. So a prompt technique is not a new building block. It is a choice you
make inside the Policy about what to put in the prompt.

To show it without needing a model, we will not call anything. We will just
print the prompt that three different techniques would build for the same
question. Look at how the note to the model changes, while everything else in
the office stays the same.

Run:  python notes/04_prompt_techniques.py
"""

from microagent import Context, Message, final_answer_tool, tool


@tool(description="Multiply two integers a and b.")
def multiply(a: int, b: int) -> int:
    return a * b


TOOLS = {t.name: t for t in [multiply, final_answer_tool()]}


def tool_list(tools) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in tools.values())


# --- technique 1: zero-shot. Just ask. ---------------------------------------
def zero_shot_prompt(ctx: Context, tools) -> str:
    return (
        "You can call these tools:\n"
        f"{tool_list(tools)}\n\n"
        f"Question: {ctx.last().content}\n"
        "Reply with the tool to call."
    )


# --- technique 2: few-shot. Show a worked example first. ----------------------
def few_shot_prompt(ctx: Context, tools) -> str:
    return (
        "You can call these tools:\n"
        f"{tool_list(tools)}\n\n"
        "Example:\n"
        "Question: what is 2 times 5\n"
        "Answer: call multiply with a=2, b=5\n\n"
        f"Question: {ctx.last().content}\n"
        "Answer:"
    )


# --- technique 3: ReAct. Ask it to think out loud, then act. ------------------
def react_prompt(ctx: Context, tools) -> str:
    return (
        "You can call these tools:\n"
        f"{tool_list(tools)}\n\n"
        "Work in this format:\n"
        "Thought: reason about what to do\n"
        "Action: the tool to call and its arguments\n\n"
        f"Question: {ctx.last().content}\n"
        "Thought:"
    )


if __name__ == "__main__":
    ctx = Context.of(Message("user", "what is 6 times 7"))

    for name, builder in [
        ("zero-shot", zero_shot_prompt),
        ("few-shot", few_shot_prompt),
        ("ReAct", react_prompt),
    ]:
        print("=" * 60)
        print(f"technique: {name}")
        print("=" * 60)
        print(builder(ctx, TOOLS))
        print()

    print("Same office, same question. Only the note to the model changed.")
    print("That note is built inside the Policy. That is all a technique is.")
