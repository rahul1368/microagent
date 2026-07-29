"""Example 04, safe by design (human in the loop).

The point: a risky action should not fire without approval. In microagent that
is not a special feature, it is a Tool wrapping a Tool (with_approval). This is
exactly the layer that autonomous "act on your behalf" agents tend to skip.

Run:  python examples/04_approval_gate.py
"""

from microagent import Action, Agent, Context, FunctionPolicy, Tool, final_answer_tool, tool, with_approval


@tool(description="Permanently delete a file. DANGEROUS.")
def delete_file(path: str) -> str:
    return f"deleted {path}"


def build(approver) -> Agent:
    safe_delete = with_approval(delete_file, approver)

    def brain(context: Context, tools):
        last = context.last()
        if last is not None and last.name == "delete_file":
            return Action("final_answer", {"answer": last.content}, "report what happened")
        return Action("delete_file", {"path": "/important/data.db"}, "user asked to delete it")

    return Agent(
        FunctionPolicy(brain),
        tools=[safe_delete, final_answer_tool()],
        name="ops",
        max_steps=3,
    )


if __name__ == "__main__":
    # An approver that refuses anything touching /important.
    def human(tool_name: str, args: dict) -> bool:
        decision = "/important" not in args.get("path", "")
        print(f"[approval] {tool_name}({args}) -> {'ALLOW' if decision else 'DENY'}")
        return decision

    agent = build(human)
    print(agent.answer("delete /important/data.db"))
    print("\n(now the same agent + tool, but an approver that says yes)")

    agent2 = build(lambda name, args: True)
    print(agent2.answer("delete /important/data.db"))
