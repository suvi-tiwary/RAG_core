"""
Base class for every RAG type.

To add one, inherit from BaseRAG and:
    1. describe its steps in FLOW (the docs playground draws this diagram)
    2. implement run(), recording every step with a Trace
answer() then works for free.
"""

# kinds of FLOW nodes, they decide the colour in the flow diagram
NODE_KINDS = ["search", "llm", "output"]


class BaseRAG:
    """
    Common interface for RAG types.

    FLOW lists the steps and the arrows between them:
        {"nodes": [{"id": "retrieve", "label": "Retrieve", "kind": "search"}, ...],
         "edges": [["retrieve", "rerank"], ...]}
    Every step that run() records must be a node, and two steps in a row
    must be joined by an edge.
    """

    FLOW = {"nodes": [], "edges": []}

    def run(self, query):
        """
        Answer the question and show how.

        Returns:
            A dict with "answer" (the text), "sources" (the chunks used for
            the answer) and "steps" (one {"step", "input", "output", "ms"}
            dict per step, in order).
        """
        raise NotImplementedError

    def answer(self, query):
        """Return only the answer text."""
        return self.run(query)["answer"]
