"""Run the examples embedded in docstrings, so they can never go stale."""

import doctest
import importlib


def test_tool_docstrings_run():
    # Import the module by name. The attribute microagent.tool is the `tool`
    # decorator function, which shadows the submodule of the same name.
    module = importlib.import_module("microagent.tool")
    result = doctest.testmod(module, verbose=False)
    assert result.attempted > 0, "expected at least one doctest example"
    assert result.failed == 0
