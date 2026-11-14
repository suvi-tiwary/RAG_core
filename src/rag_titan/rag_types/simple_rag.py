"""
Simple RAG:
    Question -> Retrieve -> Rerank -> Generate
"""

from rag_titan.rag_types.base_rag import BaseRAG
from rag_titan.rag_types.trace import Trace
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleRAG(BaseRAG):
    """
    The classic RAG: one search, then one answer.

    Args:
        retriever: any BaseRetriever, for example a DenseRetriever.
        reranker: a BaseReranker.
        generator: a BaseGenerator.
        top_k: how many chunks to retrieve.
        top_n: how many chunks to keep after reranking.
    """

    FLOW = {
        "nodes": [
            {"id": "retrieve", "label": "Retrieve", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
        ],
        "edges": [["retrieve", "rerank"], ["rerank", "generate"]],
    }

    def __init__(self, retriever, reranker, generator, top_k=5, top_n=2):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n

    def run(self, query):
        """Answer the question from the indexed documents and record each step."""
        query = query.strip()
        trace = Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        matches = trace.record("retrieve", self.retriever.retrieve, query=query, top_k=self.top_k)
        best_chunks = trace.record(
            "rerank", self.reranker.rerank, query=query, matches=matches, top_n=self.top_n
        )

        for chunk in best_chunks:
            logger.info("Selected %s from %s (rerank %.4f)",
                        chunk["id"], chunk["source"], chunk["rerank_score"])

        answer = trace.record("generate", self.generator.generate, query=query, chunks=best_chunks)
        return trace.result(answer, best_chunks)
