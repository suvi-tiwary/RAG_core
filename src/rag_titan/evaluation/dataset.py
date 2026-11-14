"""
build_dataset(): turn the logs of ANY RAG system + your golden questions into
the data file that `rag-titan evaluate --input` and RAGEvaluator read.

    from rag_titan.evaluation import build_dataset

    data = build_dataset(
        chunks=[{"id": "p1", "text": "The daily meal allowance is 45 euros."}],
        runs=[{"question": "How much is the meal allowance?",
               "retrieved_ids": ["p1"], "context_ids": ["p1"],
               "answer": "45 euros a day [p1]."}],
        golden="golden.json",
        documents={"policy.md": policy_text},
    )

It finds where every chunk sits in its document, copies your labels from the
golden questions and works out the gold chunk ids from the evidence quotes.
What your RAG did (the run) is copied as it is: labels never come from it.
"""

from rag_titan.evaluation.collect import split_citations
from rag_titan.evaluation.golden import GOLD_KEYS, load_golden
from rag_titan.evaluation.stages.spans import as_documents, evidence_spans

# what a run may contain: the outputs your RAG produced, under rag-titan's names
RUN_KEYS = {"id", "question", "retrieved_ids", "reranked_ids", "context_ids", "context",
            "answer", "raw_answer", "citations", "abstained", "verified_claims",
            "task_success", "latency_ms", "retrieval_latency_ms", "generation_latency_ms",
            "cost_usd", "total_tokens"}


def build_dataset(chunks, runs, golden, documents, require_validated=True):
    """
    Return {"documents", "chunks", "queries", "expected_evidence"} ready for evaluation.

    Args:
        chunks: your chunks, dicts with "id" and "text" ("doc_id" when there
            are several documents; the text must occur in that document).
        runs: one dict per question your RAG answered, with "question" (or the
            golden "id") and its outputs: "retrieved_ids", "context_ids",
            "answer", latencies, ... (see RUN_KEYS).
        golden: golden questions, a JSON file path or a list of dicts.
        documents: {name: text} of the documents your chunks come from.
        require_validated: refuse golden questions a person has not reviewed.
    """
    documents = as_documents(documents=documents)
    placed = place_chunks(chunks, documents)
    golden = _check_reviewed(load_golden(golden), require_validated)
    by_id = {record["id"]: record for record in golden}
    by_question = {record["question"].strip(): record for record in golden}

    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list of dicts")
    queries = [_query(run, number, by_id, by_question, placed, documents)
               for number, run in enumerate(runs, start=1)]
    # every quote once, so chunking can check that no boundary cuts the evidence
    quotes = list(dict.fromkeys(quote for query in queries
                                for quote in query.get("expected_evidence") or []))
    data = {"documents": documents, "chunks": placed, "queries": queries}
    if quotes:
        data["expected_evidence"] = quotes
    return data


def place_chunks(chunks, documents):
    """Return the chunks with "doc_id", "start" and "end" in their document."""
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks must be a non-empty list of dicts with 'id' and 'text'")
    placed, seen = [], set()
    cursor = {}   # where the last chunk of each document started: chunks come in order
    for chunk in chunks:
        if not isinstance(chunk, dict) or "id" not in chunk or not chunk.get("text"):
            raise ValueError(f"Every chunk needs an 'id' and a 'text', got {chunk!r}")
        if chunk["id"] in seen:
            raise ValueError(f"Chunk id {chunk['id']!r} is used more than once")
        seen.add(chunk["id"])
        doc_id = _doc_of(chunk, documents)
        text = documents[doc_id]
        start = text.find(chunk["text"], cursor.get(doc_id, 0))
        if start < 0:
            start = text.find(chunk["text"])
        if start < 0:
            raise ValueError(f"Chunk {chunk['id']!r}: its text is not in document {doc_id!r} "
                             "(it must be copied word for word)")
        cursor[doc_id] = start
        placed.append({**chunk, "doc_id": doc_id, "start": start,
                       "end": start + len(chunk["text"])})
    return placed


def _doc_of(chunk, documents):
    """The chunk's document: its "doc_id", the only document, or the one holding its text."""
    if chunk.get("doc_id") is not None:
        if chunk["doc_id"] not in documents:
            raise ValueError(f"Chunk {chunk['id']!r}: unknown doc_id {chunk['doc_id']!r}")
        return chunk["doc_id"]
    holding = [name for name, text in documents.items() if chunk["text"] in text]
    if len(holding) == 1 or len(documents) == 1:
        return holding[0] if holding else next(iter(documents))
    raise ValueError(f"Chunk {chunk['id']!r}: give its 'doc_id', the text is in "
                     f"{len(holding)} documents")


def _check_reviewed(golden, require_validated):
    """Refuse unreviewed golden questions: a score is only as good as its labels."""
    waiting = [record["id"] for record in golden if record.get("validated") is not True]
    if require_validated and waiting:
        raise ValueError(f"Golden questions not reviewed yet: {', '.join(map(str, waiting))}. "
                         "Check them and set validated to true, or pass "
                         "require_validated=False")
    return golden


def _query(run, number, by_id, by_question, placed, documents):
    """Join one run with its golden question: outputs from the run, labels from golden."""
    if not isinstance(run, dict):
        raise ValueError(f"Run {number} must be a dict")
    unknown = set(run) - RUN_KEYS
    if unknown:
        raise ValueError(f"Run {number}: unknown keys {sorted(unknown)}; rename them to "
                         f"rag-titan's names: {sorted(RUN_KEYS)}")
    record = by_id.get(run.get("id")) or by_question.get(str(run.get("question", "")).strip())
    if record is None:
        raise ValueError(f"Run {number}: no golden question with this id or question text")

    query = {"id": record["id"], "text": record["question"]}
    query.update({key: value for key, value in run.items() if key not in ("id", "question")})
    query.update({key: record[key] for key in GOLD_KEYS if key in record})
    _add_gold_ids(query, placed, documents)
    _add_citations(query, placed)
    return query


def _add_gold_ids(query, placed, documents):
    """Gold chunk ids = the chunks that hold (part of) an evidence quote."""
    if "expected_chunk_ids" in query or "relevance" in query or not query.get("expected_evidence"):
        return
    spans = evidence_spans(query["expected_evidence"], documents)
    query["expected_chunk_ids"] = [chunk["id"] for chunk in placed if any(
        doc == chunk["doc_id"] and chunk["start"] < end and chunk["end"] > start
        for doc, start, end in spans)]
    query.setdefault("expected_citations", query["expected_chunk_ids"])


def _add_citations(query, placed):
    """Without "citations", take markers like [p1] out of the answer and use them."""
    if "citations" in query or not query.get("answer"):
        return
    answer, citations = split_citations(query["answer"], [chunk["id"] for chunk in placed])
    query.setdefault("raw_answer", query["answer"])
    query.update(answer=answer, citations=citations)
