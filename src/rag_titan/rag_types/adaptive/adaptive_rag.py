"""
Adaptive RAG:

Question -> Choose route -> Retrieve -> Rerank -> Check evidence
                                                  |
                                  Enough evidence -> Answer
                                                  |
                              Not enough -> Rewrite and retry
"""

from rag_titan.prompts import DIRECT_REPLY_PROMPT, NO_ANSWER
from rag_titan.rag_types.adaptive.evidence_checker import EvidenceChecker
from rag_titan.rag_types.adaptive.query_rewriter import QueryRewriter
from rag_titan.rag_types.adaptive.router import Router
from rag_titan.rag_types.base_rag import BaseRAG
from rag_titan.rag_types.trace import Trace
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

# how many chunks to retrieve (top_k) and keep (top_n) for each route
SEARCH_SIZES = {
    "simple": {"top_k": 4, "top_n": 2},
    "complex": {"top_k": 8, "top_n": 4},
}


class AdaptiveRAG(BaseRAG):
    """
    A RAG that changes its effort based on the question.

    Greetings get a direct reply, simple questions a small search,
    complex questions a bigger search. When the evidence is weak,
    the question is rewritten and the search is tried again.

    Args:
        llm: any BaseLLM, used for routing, checking and rewriting.
        retriever: any BaseRetriever, for example a DenseRetriever.
        reranker: a BaseReranker.
        generator: a BaseGenerator.
        max_attempts: how many searches to try before giving up.
    """

    FLOW = {
        "nodes": [
            {"id": "route", "label": "Choose route", "kind": "llm"},
            {"id": "direct_reply", "label": "Direct reply", "kind": "output"},
            {"id": "retrieve", "label": "Retrieve", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "check", "label": "Check evidence", "kind": "llm"},
            {"id": "rewrite", "label": "Rewrite query", "kind": "llm"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
            {"id": "give_up", "label": "No answer", "kind": "output"},
        ],
        "edges": [
            ["route", "direct_reply"], ["route", "retrieve"],
            ["retrieve", "rerank"], ["rerank", "check"],
            ["check", "generate"], ["check", "rewrite"], ["check", "give_up"],
            ["rewrite", "retrieve"],
        ],
    }

    def __init__(self, llm, retriever, reranker, generator, max_attempts=2):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.max_attempts = max_attempts

        self.router = Router(llm)
        self.evidence_checker = EvidenceChecker(llm)
        self.query_rewriter = QueryRewriter(llm)

    def run(self, query):
        """Answer the question from the indexed documents and record each step."""
        query = query.strip()
        trace = Trace()

        if not query:
            return trace.result("Please enter a question.")

        route = trace.record("route", self.router.route, query=query)
        logger.info("Selected route: %s", route)

        if route == "direct":
            reply = trace.record("direct_reply", self.llm.ask,
                                 instructions=DIRECT_REPLY_PROMPT, user_input=query)
            return trace.result(reply)

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        sizes = SEARCH_SIZES[route]
        return self._search_and_answer(query, sizes["top_k"], sizes["top_n"], trace)

    def _search_and_answer(self, query, top_k, top_n, trace):
        """Search, check the evidence, and retry with a rewritten query."""
        # the original question never changes, only the search query does
        search_query = query

        for attempt in range(1, self.max_attempts + 1):
            logger.info("Search attempt %d: '%s'", attempt, search_query)

            matches = trace.record("retrieve", self.retriever.retrieve,
                                   query=search_query, top_k=top_k)
            chunks = trace.record("rerank", self.reranker.rerank,
                                  query=query, matches=matches, top_n=top_n)
            enough = trace.record("check", self.evidence_checker.has_enough,
                                  query=query, chunks=chunks)

            if enough:
                logger.info("Enough evidence found")
                answer = trace.record("generate", self.generator.generate,
                                      query=query, chunks=chunks)
                return trace.result(answer, chunks)

            if attempt == self.max_attempts:
                break

            logger.info("Not enough evidence, rewriting and widening the search")
            search_query = trace.record("rewrite", self.query_rewriter.rewrite, query=query)
            top_k += 4
            top_n += 2

        trace.add("give_up", {"attempts": self.max_attempts}, NO_ANSWER)
        return trace.result(NO_ANSWER)
