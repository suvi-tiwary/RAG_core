"""Tests for ChunkEvaluator: every chunking metric with hand-worked values."""

import math

import pytest

from rag_titan.evaluation.stages import ChunkEvaluator

# 11 words: 4 + 4 + 3
TEXT = "Leave is 25 days. Sick pay is full. Travel needs approval."
CHUNKS = [
    {"id": "c1", "start": 0, "end": 18},     # "Leave is 25 days. "
    {"id": "c2", "start": 18, "end": 36},    # "Sick pay is full. "
    {"id": "c3", "start": 36, "end": 58},    # "Travel needs approval."
]


def evaluate(chunks=CHUNKS, **options):
    options.setdefault("document_text", TEXT)
    return ChunkEvaluator().evaluate(chunks, **options)


def test_coverage_and_sizes():
    scores = evaluate(size_limits=(4, 10))

    assert scores["chunk_count"] == 3
    assert scores["total_source_tokens"] == 11
    assert scores["coverage"] == 1
    assert scores["overlap_ratio"] == 0
    assert scores["mean_chunk_tokens"] == pytest.approx(11 / 3)
    assert (scores["min_chunk_tokens"], scores["max_chunk_tokens"]) == (3, 4)
    assert scores["size_cv"] == pytest.approx(math.sqrt(2 / 9) / (11 / 3))
    assert scores["size_compliance"] == pytest.approx(2 / 3)     # the 3-word chunk is too small
    assert scores["warnings"] == []


def test_lost_content_lowers_coverage():
    scores = evaluate([CHUNKS[0], CHUNKS[2]])

    assert scores["coverage"] == pytest.approx(7 / 11)


def test_overlapping_chunks_raise_redundancy():
    scores = evaluate([{"id": "a", "start": 0, "end": 36}, {"id": "b", "start": 18, "end": 58}])

    assert scores["coverage"] == 1
    assert scores["overlap_ratio"] == pytest.approx((8 + 7 - 11) / 11)


def test_evidence_and_structure_kept_whole():
    scores = evaluate(expected_evidence=["25 days. Sick pay", "Sick pay is full"],
                      structural_blocks=["Travel needs approval."])

    assert scores["evidence_completeness"] == 0.5          # the first quote is cut at 18
    assert scores["evidence_fragmentation"] == 1.5         # it touches 2 chunks, the other 1
    assert scores["structural_integrity"] == 1


def test_boundary_f1_with_and_without_tolerance():
    ideal = [{"start": 0, "end": 18}, {"start": 18, "end": 58}]     # one ideal boundary: 18
    assert evaluate(expected_chunks=ideal)["boundary_f1"] == pytest.approx(2 / 3)   # P 1/2, R 1

    shifted = [{"id": "a", "start": 0, "end": 20}, {"id": "b", "start": 20, "end": 58}]
    assert evaluate(shifted, expected_chunks=ideal)["boundary_f1"] == 0
    assert evaluate(shifted, expected_chunks=ideal, boundary_tolerance=2)["boundary_f1"] == 1

    whole = [{"id": "a", "start": 0, "end": 58}]
    assert evaluate(whole, expected_chunks=[{"start": 0, "end": 58}])["boundary_f1"] == 1


def test_metadata_checks():
    chunks = [dict(CHUNKS[0], metadata={"source": "hr.md", "page": 1}),
              dict(CHUNKS[1], metadata={"source": "hr.md"})]
    scores = evaluate(chunks, required_metadata=["source", "page"],
                      expected_metadata={"c1": {"page": 1}, "c2": {"page": 2}})

    assert scores["metadata_completeness"] == 0.5
    assert scores["metadata_accuracy"] == 0.5


def test_lexical_proxies_need_no_source_text():
    scores = ChunkEvaluator().evaluate(["red blue. blue green.", "green sky."])

    assert scores["lexical_cohesion"] == pytest.approx(1 / 3)           # {blue} of 3 words
    assert scores["lexical_chunk_stickiness"] == pytest.approx(1 / 4)   # {green} of 4 words
    assert scores["lexical_boundary_clarity"] == pytest.approx(3 / 4)
    assert scores["coverage"] is None
    assert scores["total_source_tokens"] is None
    assert scores["boundary_f1"] is None


def test_semantic_metrics_from_vectors():
    scores = evaluate(
        CHUNKS[:2],
        sentence_vectors={"c1": [[1, 0], [0, 1]], "c2": [[0, 1], [0, 1]]},
        chunk_vectors={"c1": [1, 0], "c2": [1, 0]},
    )

    assert scores["semantic_cohesion"] == pytest.approx(0.5)            # c1: 0, c2: 1
    assert scores["semantic_boundary_clarity"] == pytest.approx(0)      # same vector both sides
    assert scores["semantic_chunk_stickiness"] == pytest.approx(1)


def test_semantic_metrics_are_none_without_vectors():
    scores = evaluate()

    assert scores["semantic_cohesion"] is None
    assert scores["semantic_boundary_clarity"] is None
    assert scores["semantic_chunk_stickiness"] is None


def test_several_documents():
    documents = {"a.md": "Leave is 25 days.", "b.md": "Sick pay is full."}
    chunks = [{"id": "a1", "doc_id": "a.md", "start": 0, "end": 9},
              {"id": "a2", "doc_id": "a.md", "start": 9, "end": 17},
              {"id": "b1", "doc_id": "b.md", "start": 0, "end": 17}]
    ideal = [{"doc_id": "a.md", "start": 0, "end": 17}, {"doc_id": "b.md", "start": 0, "end": 17}]
    scores = ChunkEvaluator().evaluate(chunks, documents=documents, expected_chunks=ideal,
                                       expected_evidence=["25 days", "Sick pay"])

    assert scores["coverage"] == 1
    assert scores["evidence_completeness"] == 1
    assert scores["boundary_f1"] == 0      # split a.md at 9, the ideal has no boundary


def test_unplaced_chunk_gives_warning_and_no_coverage():
    scores = evaluate([CHUNKS[0], "not in the text"])

    assert scores["coverage"] is None
    assert scores["warnings"] == ["chunk_2: text not found exactly once; give start and end"]


def test_errors():
    with pytest.raises(ValueError, match="size_limits"):
        evaluate(size_limits=(10, 2))
    with pytest.raises(ValueError, match="needs the source text"):
        ChunkEvaluator().evaluate(["a b"], expected_evidence=["a"])
    with pytest.raises(ValueError, match="boundary_tolerance"):
        evaluate(expected_chunks=CHUNKS, boundary_tolerance=-1)
    with pytest.raises(ValueError, match="sentence_vectors"):
        evaluate(sentence_vectors=[[1, 0]])
    with pytest.raises(ValueError, match="expected_metadata"):
        evaluate(expected_metadata=["page"])
