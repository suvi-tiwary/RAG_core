"""
Measure how well retrieval finds the right evidence, and compare retrievers.

It indexes the sample policies in examples/sample_docs/, asks the questions in
examples/questions.json and prints Recall@k, MRR, nDCG@k and latency for:
    dense          the default retriever (near-copies removed)
    dense_no_dedupe the same retriever keeping near-copies

Before running:
    ollama pull nomic-embed-text

Run:
    python examples/evaluate_retrieval.py
    python examples/evaluate_retrieval.py my_questions.json file1.pdf file2.pdf
"""

import sys
from pathlib import Path

from rag_titan import build_rag_parts, get_settings
from rag_titan.evaluation import compare_retrievers, format_report, load_questions
from rag_titan.retrieval import DenseRetriever

HERE = Path(__file__).parent


def main():
    """Index the files once, then score every retriever on the same questions."""
    questions_path = sys.argv[1] if len(sys.argv) > 1 else HERE / "questions.json"
    files = sys.argv[2:] or sorted(str(path) for path in (HERE / "sample_docs").glob("*.md"))

    questions = load_questions(questions_path)

    # small chunks, so every file has several chunks to choose from
    parts = build_rag_parts(get_settings({"chunk_size": 40, "chunk_overlap": 20}))
    parts["pipeline"].run(files)

    dense = parts["retriever"]
    retrievers = {
        "dense": dense,
        "dense_no_dedupe": DenseRetriever(dense.embedder, dense.store, dedupe_overlap=None),
    }

    results = compare_retrievers(retrievers, questions, top_k=3)
    print(format_report(results))

    # the questions the default retriever did worst on
    print("\nWeakest questions for 'dense':")
    weakest = sorted(results[0]["rows"], key=lambda row: (row["recall"], row["ndcg"]))
    for row in weakest[:3]:
        print(f"  {row['id']}  recall {row['recall']:.2f}  nDCG {row['ndcg']:.2f}  "
              f"found {row['sources']}")


if __name__ == "__main__":
    main()
