"""
Math tools the agent can call.

The LLM sends arguments as text, so every tool converts them with float().
The docstrings are sent to the LLM, so keep them clear.
"""


def add(a, b):
    """
    Add two numbers: a + b

    Args:
        a: first number
        b: second number
    """
    return float(a) + float(b)


def subtract(a, b):
    """
    Subtract two numbers: a - b

    Args:
        a: number to subtract from
        b: number to subtract
    """
    return float(a) - float(b)


def multiply(a, b):
    """
    Multiply two numbers: a * b

    Args:
        a: first number
        b: second number
    """
    return float(a) * float(b)


def divide(a, b):
    """
    Divide two numbers: a / b

    Args:
        a: number to divide
        b: number to divide by
    """
    if float(b) == 0:
        return "undefined (division by zero)"
    return float(a) / float(b)


MATH_TOOLS = [add, subtract, multiply, divide]
