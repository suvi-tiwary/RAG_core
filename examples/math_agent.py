"""
Run the example math agent (needs a model that supports tools).

Before running:
    ollama pull qwen3:4b

Run:
    python examples/math_agent.py
"""

from rag_titan import get_settings, setup_logging
from rag_titan.agents import MathAgent
from rag_titan.factory import build_agent_llm


def main():
    """Ask the math agent one question."""
    setup_logging("INFO")

    settings = get_settings()
    agent = MathAgent(build_agent_llm(settings), max_steps=settings["agent_max_steps"])

    print(agent.run("What is (12 + 8) * 3 divided by 4?"))


if __name__ == "__main__":
    main()
