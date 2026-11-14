"""
Turn plain Python functions into the tool descriptions Ollama expects.

Agents hand the LLM normal functions (see agents/tools/math_tools.py). The
model cannot read Python, so each function becomes a small JSON schema: its
name, the first part of its docstring, and one entry per argument with the
text from the docstring's "Args:" section.
"""

import inspect

# Python type -> JSON schema type; arguments without a known type are "string"
JSON_TYPES = {int: "integer", float: "number", bool: "boolean", str: "string",
              list: "array", dict: "object"}


def function_to_tool(func):
    """
    Build an Ollama tool schema for one function.

    A dict is treated as a ready-made schema and returned as it is.

    Args:
        func: a Python function with an "Args:" section in its docstring.
    """
    if isinstance(func, dict):
        return func
    if not callable(func):
        raise ValueError(f"A tool must be a function or a dict, got {type(func).__name__}")

    summary, arg_texts = _read_docstring(inspect.getdoc(func) or "")
    properties = {}

    for name, param in inspect.signature(func).parameters.items():
        properties[name] = {"description": arg_texts.get(name, ""),
                            "type": JSON_TYPES.get(param.annotation, "string")}

    # the official client listed every argument as required, even ones with a
    # default value; we do the same so models see exactly the tools they saw before
    required = list(properties)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": summary,
            "parameters": {"type": "object", "required": required, "properties": properties},
        },
    }


def tools_to_schemas(tools):
    """
    Turn a list of functions (or ready-made dicts) into tool schemas.

    Args:
        tools: list of functions or dicts, or None.
    """
    return [function_to_tool(tool) for tool in tools or []]


def _read_docstring(text):
    """Split a docstring into its summary and a {argument: description} dict."""
    summary_lines, arg_texts = [], {}
    section = "summary"
    last_arg = None

    for line in text.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("args:"):
            section = "args"
        elif lowered.startswith(("returns:", "yields:", "raises:")):
            section = "other"
        elif section == "summary":
            summary_lines.append(line.strip())
        elif section == "args" and ":" in line:
            last_arg, _, description = line.strip().partition(":")
            last_arg = last_arg.split("(")[0].strip()      # "a (int)" -> "a"
            arg_texts[last_arg] = description.strip()
        elif section == "args" and last_arg and line.strip():
            arg_texts[last_arg] += " " + line.strip()      # description goes on

    return "\n".join(summary_lines).strip(), arg_texts
