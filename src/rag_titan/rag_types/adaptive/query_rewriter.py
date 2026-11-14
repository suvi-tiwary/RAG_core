"""Uses the LLM to write a better search query."""

from rag_titan.prompts import QUERY_REWRITE_PROMPT


class QueryRewriter:
    """
    Rewrites a question into a better search query.

    Args:
        llm: any BaseLLM.
    """

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query):
        """Return the rewritten query, or the original if the LLM is empty."""
        rewritten = self.llm.ask(QUERY_REWRITE_PROMPT, query)
        return rewritten or query
