"""
Base class for every generator (grounded, streaming, citation-checked, ...).

To add one, inherit from BaseGenerator and implement generate().
"""

from rag_titan.generation.context_builder import build_context


class BaseGenerator:
    """
    Writes the final answer from the question and the chosen chunks.

    Args:
        llm: any BaseLLM.
        instructions: the system prompt for the LLM.
    """

    def __init__(self, llm, instructions):
        self.llm = llm
        self.instructions = instructions

    def generate(self, query, chunks):
        """Return the final answer text for the question."""
        raise NotImplementedError

    def _build_user_input(self, query, chunks):
        """Put the context and the question into one user message."""
        context = build_context(chunks)
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
