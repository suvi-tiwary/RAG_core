"""
Chunking evaluation: score the chunks right after the document is split.

    text = "Leave is 25 days. Sick pay is full."
    scores = ChunkEvaluator().evaluate(
        chunks=["Leave is 25 days. ", "Sick pay is full."],
        document_text=text,
        expected_evidence=["25 days", "Sick pay is full"],
        size_limits=(2, 50),
    )
    print(scores["coverage"], scores["evidence_completeness"])   # 1.0 1.0

Only chunks are needed. Every extra input (source text, gold evidence, ideal
chunks, vectors, ...) switches on more metrics; the others stay None.
"""

from rag_titan.evaluation.stages.chunk_boundaries import boundary_f1
from rag_titan.evaluation.stages.chunk_cohesion import lexical_scores, semantic_scores
from rag_titan.evaluation.stages.chunk_shape import (
    check_size_limits,
    coverage_scores,
    fragmentation,
    intact_rate,
    metadata_accuracy,
    metadata_completeness,
    size_scores,
)
from rag_titan.evaluation.stages.spans import as_documents, evidence_spans, normalize_chunks


class ChunkEvaluator:
    """Scores the output of a chunker against the source text and gold labels."""

    def evaluate(self, chunks, document_text=None, documents=None, expected_chunks=None,
                 expected_evidence=None, structural_blocks=None, required_metadata=None,
                 expected_metadata=None, size_limits=None, boundary_tolerance=0,
                 sentence_vectors=None, chunk_vectors=None):
        """
        Score the chunks.

        Args:
            chunks: texts, or dicts with "id", "text", "start", "end", "doc_id", "metadata".
            document_text: the source text of a single document.
            documents: {doc_id: text} when there are several documents.
            expected_chunks: the ideal chunks marked by a human (for boundary F1).
            expected_evidence: gold quotes or {"start", "end"} spans that must stay whole.
            structural_blocks: tables, lists or code that must stay whole (same format).
            required_metadata: keys every chunk must have, e.g. ["source", "page"].
            expected_metadata: {chunk_id: {key: value}} of correct metadata values.
            size_limits: (min, max) words of a valid chunk.
            boundary_tolerance: characters two boundaries may be apart and still match.
            sentence_vectors: {chunk_id: [vector per sentence]} for semantic cohesion.
            chunk_vectors: {chunk_id: vector} for semantic chunk stickiness.

        Returns:
            A dict of metric name -> value (None = not measured) and "warnings".
        """
        check_size_limits(size_limits)
        documents = as_documents(document_text, documents)
        items, warnings = normalize_chunks(chunks, documents)

        scores = {"chunk_count": len(items)}
        scores.update(coverage_scores(items, documents))
        scores.update(size_scores(items, size_limits))
        scores.update(self._gold_scores(items, documents, expected_evidence, structural_blocks))
        scores["metadata_completeness"] = metadata_completeness(items, required_metadata)
        scores["metadata_accuracy"] = metadata_accuracy(items, expected_metadata)
        scores.update(lexical_scores(items))
        scores.update(semantic_scores(items, sentence_vectors, chunk_vectors))
        scores["boundary_f1"] = self._boundary_f1(items, expected_chunks, documents,
                                                  boundary_tolerance)
        scores["warnings"] = warnings
        return scores

    def _gold_scores(self, items, documents, expected_evidence, structural_blocks):
        """Check that gold evidence and structural blocks were not cut by a boundary."""
        self._need_documents(documents, expected_evidence, "expected_evidence")
        self._need_documents(documents, structural_blocks, "structural_blocks")

        evidence = evidence_spans(expected_evidence, documents)
        blocks = evidence_spans(structural_blocks, documents)
        return {"evidence_completeness": intact_rate(evidence, items),
                "evidence_fragmentation": fragmentation(evidence, items),
                "structural_integrity": intact_rate(blocks, items)}

    def _boundary_f1(self, items, expected_chunks, documents, tolerance):
        """Place the ideal chunks in the text and compare the split points."""
        if expected_chunks is None:
            return None
        self._need_documents(documents, expected_chunks, "expected_chunks")
        expected_items, _ = normalize_chunks(expected_chunks, documents)
        return boundary_f1(items, expected_items, tolerance)

    def _need_documents(self, documents, value, name):
        """Gold spans are places in the source text, so the text must be given."""
        if value and not documents:
            raise ValueError(f"{name} needs the source text: give document_text or documents")
