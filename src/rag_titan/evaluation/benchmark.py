"""
Run retrievers on the same test questions and compare them.

    questions = load_questions("questions.json")
    results = compare_retrievers({"dense": retriever}, questions, top_k=5)
    print(format_report(results))
"""

import time

import numpy as np

from rag_titan.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag_titan.evaluation.questions import check_questions

# the columns format_report() prints: (title, key in the result, number format)
COLUMNS = [
    ("Recall@k", "recall", "{:.3f}"),
    ("MRR", "mrr", "{:.3f}"),
    ("nDCG@k", "ndcg", "{:.3f}"),
    ("p50 ms", "p50_ms", "{:.1f}"),
    ("p95 ms", "p95_ms", "{:.1f}"),
]


def evaluate_retriever(retriever, questions, top_k=5, name=None):
    """
    Ask every question and score what the retriever finds.

    Args:
        retriever: any BaseRetriever with indexed documents.
        questions: list of question dicts (see load_questions).
        top_k: how many chunks to retrieve and score per question.
        name: name shown in the report, defaults to retriever.name.

    Returns:
        A dict with the means ("recall", "mrr", "ndcg"), the latency
        ("p50_ms", "p95_ms") and one row per question in "rows".
    """
    check_questions(questions)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a whole number of at least 1, got {top_k!r}")
    if not retriever.has_documents():
        raise ValueError("The retriever has no documents: index your files first")

    rows = [_score_question(retriever, question, top_k) for question in questions]

    result = {"name": name or retriever.name, "top_k": top_k}
    result.update(summarize_rows(rows))
    result["rows"] = rows
    return result


def compare_retrievers(retrievers, questions, top_k=5):
    """
    Evaluate several retrievers on identical questions and top_k.

    Args:
        retrievers: dict of {name: retriever}, e.g. {"dense": r1, "bm25": r2}.

    Returns:
        A list with one evaluate_retriever() result per retriever, in order.
    """
    if not isinstance(retrievers, dict) or not retrievers:
        raise ValueError("retrievers must be a non-empty dict of {name: retriever}")

    return [evaluate_retriever(retriever, questions, top_k, name=name)
            for name, retriever in retrievers.items()]


def format_report(results):
    """Return the results as a text table, one line per retriever."""
    if not results:
        raise ValueError("There are no results to report")

    top_k = results[0]["top_k"]
    columns = [(title.replace("@k", f"@{top_k}"), key, fmt) for title, key, fmt in COLUMNS]
    title = f"Retrieval report: {len(results[0]['rows'])} questions, top_k={top_k}"
    return format_table(title, "retriever", results, columns)


def format_table(title, first_column, results, columns):
    """
    Return a text table: a title, a header, then one line per result.

    Args:
        title: the first line, e.g. "Retrieval report: 6 questions".
        first_column: header of the name column, e.g. "retriever".
        results: dicts with a "name" and one value per column key.
        columns: list of (title, key, number format) tuples.
    """
    width = max(len(first_column), *(len(result["name"]) for result in results))
    header = f"{first_column:<{width}}" + "".join(f"{name:>10}" for name, _, _ in columns)

    lines = [title, "", header, "-" * len(header)]
    for result in results:
        cells = "".join(f"{fmt.format(result[key]):>10}" for _, key, fmt in columns)
        lines.append(f"{result['name']:<{width}}" + cells)

    return "\n".join(lines)


def _score_question(retriever, question, top_k):
    """Retrieve for one question, time it and score it."""
    start = time.perf_counter()
    found = retriever.retrieve(question["question"], top_k=top_k)
    ms = (time.perf_counter() - start) * 1000

    required = question["required_evidence_ids"]
    return {
        "id": question["id"],
        "task_type": question.get("task_type"),
        "recall": recall_at_k(found, required, top_k),
        "reciprocal_rank": reciprocal_rank(found, required, top_k),
        "ndcg": ndcg_at_k(found, required, top_k),
        "ms": ms,
        "found": [chunk["id"] for chunk in found],
        "sources": [chunk["source"] for chunk in found],
    }


def summarize_rows(rows):
    """
    Mean recall, MRR and nDCG of the rows, and the median (p50) and p95 of "ms".
    Used by both the retrieval and the end-to-end evaluation.
    """
    times = [row["ms"] for row in rows]
    return {
        "recall": float(np.mean([row["recall"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
        "ndcg": float(np.mean([row["ndcg"] for row in rows])),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }
