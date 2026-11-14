"""
Golden questions: the test questions and gold labels used to evaluate a rag-titan pipeline.

A golden question is a dict with "id", "question" and the gold labels you have:
    {"id": "q1", "question": "How many leave days do I get?",
     "expected_evidence": ["gets 20 paid leave days per calendar year"],   # exact quotes
     "expected_answer": "20 paid leave days per calendar year.",
     "required_facts": ["20"]}

Which chunks are the right ones to retrieve is worked out from the first of
these a question has:
    "expected_chunk_ids" or "relevance"   chunk ids you labelled yourself
    "expected_evidence"                   quotes: the chunks that hold them (best:
                                          it keeps working when you change the chunker)
    "required_evidence_ids"               file or page labels such as "hr.pdf#p4":
                                          EVERY chunk of that file or page

With file labels a big file has many gold chunks, so Recall@K and citation
recall look low: nobody retrieves a whole file. Hit@K, MRR and Precision@K
stay meaningful. Use quotes when you need exact recall.
"""

import json
from pathlib import Path

from rag_titan.evaluation.metrics import matches_evidence
from rag_titan.evaluation.stages.queries import check_queries
from rag_titan.evaluation.stages.spans import evidence_spans

# gold labels copied from a golden question to the query, when present
GOLD_KEYS = ["expected_evidence", "expected_chunk_ids", "relevance", "expected_answer",
             "required_facts", "expected_citations", "answerable", "max_context_tokens"]


def load_golden(questions):
    """Return the golden questions from a JSON file path or a list, after checking them."""
    if not isinstance(questions, list):
        path = Path(questions)
        if not path.is_file():
            raise FileNotFoundError(f"Golden questions file not found: {path}")
        try:
            questions = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not valid JSON (line {error.lineno}): {error.msg}")

    if not isinstance(questions, list) or not questions:
        raise ValueError("Golden questions must be a non-empty list of records")
    check_queries(questions)
    for record in questions:
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError(f"Question {record['id']!r}: 'question' must be a non-empty text")
    return questions


def gold_chunk_ids(record, chunks, documents):
    """
    Return the ids of the gold chunks of a golden question, or None when the
    question has its own chunk labels or no evidence labels at all.

    Args:
        record: the golden question.
        chunks: the rag-titan chunks ("id", "source", "start_offset", "end_offset", ...).
        documents: {source: text} of the indexed files.
    """
    if "expected_chunk_ids" in record or "relevance" in record:
        return None

    if "expected_evidence" in record:
        spans = evidence_spans(record["expected_evidence"], documents)
        return [chunk["id"] for chunk in chunks if _touches(chunk, spans)]

    labels = record.get("required_evidence_ids")
    if labels:
        return [chunk["id"] for chunk in chunks
                if any(matches_evidence(chunk, label) for label in labels)]
    return None


def _touches(chunk, spans):
    """Return True when the chunk holds (part of) one of the evidence spans."""
    start, end = chunk.get("start_offset"), chunk.get("end_offset")
    if start is None or end is None:
        return False
    return any(doc_id == chunk["source"] and start < span_end and end > span_start
               for doc_id, span_start, span_end in spans)


def check_golden(questions, documents=None):
    """
    Check a golden set before you trust the scores made with it.

    Args:
        questions: path to a golden questions JSON file, or a list of dicts.
        documents: {doc_id: text}; when given, every quote must be found in them.

    Returns:
        A dict with "question_count", "validated_count", "validated_share",
        "problems" (a list of {"id", "problem"}) and "ready".
        A person marks a question as checked with "validated": true.
        "ready" is True when there is no problem and every question is validated.
    """
    questions = load_golden(questions)
    problems = []
    for record in questions:
        problems += [{"id": record["id"], "problem": problem}
                     for problem in _record_problems(record, documents)]

    validated = sum(record.get("validated") is True for record in questions)
    return {
        "question_count": len(questions),
        "validated_count": validated,
        "validated_share": validated / len(questions),
        "problems": problems,
        "ready": not problems and validated == len(questions),
    }


def _record_problems(record, documents):
    """Return the problems of one golden question as a list of sentences."""
    problems = []
    answerable = record.get("answerable", True)
    evidence_keys = ("expected_evidence", "expected_chunk_ids", "relevance",
                     "required_evidence_ids")
    has_evidence = any(record.get(key) for key in evidence_keys)

    if answerable and not has_evidence:
        problems.append("no gold evidence: add expected_evidence quotes")
    if answerable and not record.get("expected_answer"):
        problems.append("no expected_answer")
    if not answerable and has_evidence:
        problems.append("marked unanswerable but has gold evidence")

    answer = (record.get("expected_answer") or "").lower()
    for fact in record.get("required_facts") or []:
        if answer and fact.lower() not in answer:
            problems.append(f"required fact {fact!r} is not in the expected_answer")

    for quote in (record.get("expected_evidence") or []) if documents else []:
        try:
            evidence_spans([quote], documents)
        except ValueError:
            problems.append(f"quote {str(quote)[:40]!r} is not found exactly once")
    return problems
