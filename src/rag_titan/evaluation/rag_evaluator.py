"""
RAGEvaluator: evaluate every stage of a RAG pipeline with one call.

    report = RAGEvaluator(k=5).evaluate(
        chunks=chunks,                # what your chunker produced
        queries=queries,              # per question: gold labels + what each stage produced
        document_text=text,           # or documents={doc_id: text}
    )
    print(report["retrieval"]["summary"]["recall_at_k"])
    print(report["root_causes"]["first_failing_stage"])
    print(report["scorecard"]["overall"]["score"])           # 0 to 100

It does not run your RAG system: you pass what it already produced. A stage
is evaluated only when its data is there, so you can start with retrieval
alone and add the other stages later.
"""

from rag_titan.evaluation.root_cause import RootCauseAnalyzer
from rag_titan.evaluation.scoring import ScoreAggregator
from rag_titan.evaluation.stages import (
    AugmentationEvaluator,
    ChunkEvaluator,
    EmbeddingEvaluator,
    EndToEndEvaluator,
    GenerationEvaluator,
    RerankingEvaluator,
    RetrievalEvaluator,
)
from rag_titan.evaluation.stages.queries import check_k, check_queries

# every stage, in pipeline order
STAGES = ["chunking", "embedding", "retrieval", "reranking", "augmentation",
          "generation", "end_to_end"]

# extra inputs passed on to ChunkEvaluator.evaluate()
CHUNK_OPTIONS = ["expected_chunks", "expected_evidence", "structural_blocks",
                 "required_metadata", "expected_metadata", "size_limits",
                 "boundary_tolerance", "sentence_vectors", "chunk_vectors"]

# keys of the "embeddings" input, passed on to EmbeddingEvaluator.evaluate()
EMBEDDING_KEYS = ["vectors", "query_vectors", "relevant_ids", "indexed_ids", "index_results"]


class RAGEvaluator:
    """Runs the stage evaluators that have data, adds root-cause hints and a scorecard."""

    def __init__(self, k=5, thresholds=None, scorer=None):
        """
        Args:
            k: how many results to score in the ranking metrics (Recall@K, ...).
            thresholds: {metric: minimum} for the root-cause hints.
            scorer: a ScoreAggregator with your own weights, targets or stages.
        """
        self.k = check_k(k)
        self.analyzer = RootCauseAnalyzer(thresholds)
        self.scorer = scorer or ScoreAggregator()

    def evaluate(self, chunks=None, queries=None, document_text=None, documents=None,
                 embeddings=None, max_context_tokens=None, **chunk_options):
        """
        Evaluate every stage that has data.

        Args:
            chunks: the chunks of your index (see ChunkEvaluator).
            queries: list of query dicts. Each stage reads its own keys:
                "retrieved_ids", "reranked_ids", "context_ids", "answer", ...
            document_text: the source text of a single document.
            documents: {doc_id: text} when there are several documents.
            embeddings: dict with "vectors" and optionally "query_vectors",
                "relevant_ids", "indexed_ids", "index_results".
            max_context_tokens: the context budget in words.
            **chunk_options: any other ChunkEvaluator input, e.g.
                expected_evidence=[...], size_limits=(50, 300).

        Returns:
            A report dict: one entry per evaluated stage, plus "root_causes"
            and "scorecard".
        """
        unknown = sorted(set(chunk_options) - set(CHUNK_OPTIONS))
        if unknown:
            raise ValueError(f"Unknown input {unknown}: use one of {CHUNK_OPTIONS}")
        queries = check_queries(queries or [])
        source = {"document_text": document_text, "documents": documents}

        report = {}
        if chunks is not None:
            if embeddings and "vectors" in embeddings:
                chunk_options.setdefault("chunk_vectors", embeddings["vectors"])
            report["chunking"] = ChunkEvaluator().evaluate(chunks, **source, **chunk_options)
        if embeddings is not None:
            report["embedding"] = self._embedding(embeddings)

        report.update(self._query_stages(chunks or [], queries, source, max_context_tokens))
        report["root_causes"] = self.analyzer.evaluate(report)
        report["scorecard"] = self.scorer.evaluate(report)
        return report

    def _embedding(self, embeddings):
        """Check the "embeddings" input and run the embedding stage."""
        if not isinstance(embeddings, dict) or "vectors" not in embeddings:
            raise ValueError("embeddings must be a dict with 'vectors'")
        unknown = sorted(set(embeddings) - set(EMBEDDING_KEYS))
        if unknown:
            raise ValueError(f"Unknown embeddings key {unknown}: use one of {EMBEDDING_KEYS}")
        return EmbeddingEvaluator().evaluate(**embeddings, k=self.k)

    def _query_stages(self, chunks, queries, source, max_context_tokens):
        """Run each stage that reads the queries, when a query has its data."""

        def has(key):
            return any(key in query for query in queries)

        report = {}
        if has("retrieved_ids"):
            report["retrieval"] = RetrievalEvaluator().evaluate(chunks, queries, self.k, **source)
        if has("reranked_ids"):
            report["reranking"] = RerankingEvaluator().evaluate(queries, self.k)
        if has("context_ids"):
            report["augmentation"] = AugmentationEvaluator().evaluate(
                chunks, queries, max_context_tokens=max_context_tokens, **source)
        if has("answer"):
            report["generation"] = GenerationEvaluator().evaluate(queries, chunks)
            report["end_to_end"] = EndToEndEvaluator().evaluate(queries)
        return report
