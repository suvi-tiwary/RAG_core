"""
Collect the evaluation data of a rag-titan pipeline, so you do not write it by hand.

    rag = RAGTitan(rag_type="simple")
    dataset = collect_dataset(rag, ["leave.md", "remote.md"], "golden.json")
    report = RAGEvaluator(k=5).evaluate(**dataset)

It indexes the files, asks every golden question, and records what each stage
really produced: the chunks, the retrieved and reranked ids, the context given
to the LLM, the answer and the timings.

Golden questions and their gold labels are described in golden.py.
"""

import re
import time

from rag_titan.evaluation.end_to_end import NOT_ANSWERED
from rag_titan.evaluation.golden import GOLD_KEYS, gold_chunk_ids, load_golden

# a citation marker in an answer, e.g. "[chunk_3]" or "[chunk_3, page 2]"
MARKER = re.compile(r"\s*\[([^\[\]]*)\]")


def collect_dataset(rag, files, questions, include_embeddings=False):
    """
    Index the files, run every golden question, and return the RAGEvaluator input.

    Args:
        rag: a RAGTitan.
        files: list of file paths to index (the index is rebuilt).
        questions: path to a golden questions JSON file, or a list of dicts.
        include_embeddings: also collect the chunk and query vectors for the
            embedding stage (embeds every question once more).

    Returns:
        A dict with "documents", "chunks", "queries" and, when asked,
        "embeddings": pass it to RAGEvaluator().evaluate(**dataset).
    """
    questions = load_golden(questions)
    if not isinstance(files, list) or not files:
        raise ValueError("files must be a non-empty list of file paths")

    loaded = rag.pipeline.load_documents(files)
    documents = {document["source"]: document["text"] for document in loaded}
    if len(documents) != len(loaded):
        raise ValueError("Two files have the same name: rename one of them")
    rag.index(files)

    raw_chunks = rag.pipeline.chunker.chunk(loaded)
    chunks = [_as_eval_chunk(chunk, documents) for chunk in raw_chunks]
    queries = [_run_question(rag, record, gold_chunk_ids(record, raw_chunks, documents))
               for record in questions]

    dataset = {"documents": documents, "chunks": chunks, "queries": queries}
    if include_embeddings:
        dataset["embeddings"] = _embeddings(rag, queries)
    return dataset


def _as_eval_chunk(chunk, documents):
    """Turn a rag-titan chunk into the shape the stage evaluators read."""
    start, end = chunk.get("start_offset"), chunk.get("end_offset")
    extra = ("id", "text", "source", "start_offset", "end_offset")
    item = {"id": chunk["id"], "doc_id": chunk["source"], "text": chunk["text"],
            "metadata": {"source": chunk["source"],
                         **{key: value for key, value in chunk.items() if key not in extra}}}

    # Chunkers may tidy the text (e.g. line breaks become spaces), so the words
    # are taken from the document at the chunk's offsets: same words, exact place.
    if start is not None and end is not None:
        item.update(start=start, end=end, text=documents[chunk["source"]][start:end])
    return item


def _run_question(rag, record, gold_ids):
    """Ask one golden question and record what every stage produced."""
    start = time.perf_counter()
    result = rag.rag.run(record["question"])
    total_ms = (time.perf_counter() - start) * 1000

    query = {"id": record["id"], "text": record["question"]}
    query.update({key: record[key] for key in GOLD_KEYS if key in record})
    if gold_ids is not None:
        query["expected_chunk_ids"] = gold_ids

    for step, key in (("retrieve", "retrieved_ids"), ("rerank", "reranked_ids")):
        outputs = [item["output"] for item in result["steps"] if item["step"] == step]
        if outputs:     # the last search is the one the answer was made from
            query[key] = [chunk["id"] for chunk in outputs[-1]]

    context_ids = [chunk["id"] for chunk in result["sources"]]
    answer, citations = split_citations(result["answer"], context_ids)
    if "expected_chunk_ids" in query:
        query.setdefault("expected_citations", query["expected_chunk_ids"])
    query.update({
        "context_ids": context_ids,
        "answer": answer,
        "raw_answer": result["answer"],
        "citations": citations,
        "abstained": result["answer"].strip() in NOT_ANSWERED,
        "latency_ms": total_ms,
        "retrieval_latency_ms": _step_ms(result["steps"], "retrieve"),
        "generation_latency_ms": _step_ms(result["steps"], "generate"),
    })
    return query


def split_citations(answer, chunk_ids):
    """
    Take the citation markers out of an answer.

    Args:
        answer: the answer text, e.g. "You get 20 days [chunk_1, page 1]."
        chunk_ids: the ids that can be cited (the context chunks).

    Returns:
        (answer without the markers, cited chunk ids in order). The words of
        the answer are then compared with the reference without "[chunk_1]".
    """
    citations = []

    def remove(match):
        cited = [chunk_id for chunk_id in chunk_ids
                 if re.search(rf"(?<!\w){re.escape(chunk_id)}(?!\w)", match.group(1))]
        citations.extend(chunk_id for chunk_id in cited if chunk_id not in citations)
        return "" if cited else match.group(0)     # other [brackets] stay

    return MARKER.sub(remove, answer).strip(), citations


def _step_ms(steps, name):
    """Return the total time of the steps with this name, or None when there is none."""
    times = [step["ms"] for step in steps if step["step"] == name]
    return sum(times) if times else None


def _embeddings(rag, queries):
    """Collect the stored chunk vectors and embed every question."""
    retriever = rag.parts["retriever"]
    records = getattr(getattr(retriever, "store", None), "records", None)
    if records is None or not hasattr(retriever, "embedder"):
        raise ValueError("include_embeddings needs a retriever with an embedder and a "
                         "MemoryStore; pass your own 'embeddings' to RAGEvaluator instead")

    return {
        "vectors": {record["id"]: [float(value) for value in record["vector"]]
                    for record in records},
        "query_vectors": {query["id"]: [float(value) for value in
                                        retriever.embedder.embed(query["text"])]
                          for query in queries},
        "relevant_ids": {query["id"]: query["expected_chunk_ids"] for query in queries
                         if query.get("expected_chunk_ids")},
        "indexed_ids": [record["id"] for record in records],
    }
