"""Asks the LLM if the found chunks can answer the question."""

from rag_titan.generation.context_builder import build_context
from rag_titan.prompts import EVIDENCE_CHECK_PROMPT


class EvidenceChecker:
    """
    Asks the LLM whether the chunks are enough to answer the question.

    Args:
        llm: any BaseLLM.
    """

    def __init__(self, llm):
        self.llm = llm

    def has_enough(self, query, chunks):
        """Return True when the LLM replies "yes"."""
        if not chunks:
            return False

        user_input = f"QUESTION:\n{query}\n\nCONTEXT:\n{build_context(chunks)}"
        reply = self.llm.ask(EVIDENCE_CHECK_PROMPT, user_input)

        return reply.strip().lower().startswith("yes")
