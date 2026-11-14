"""
Evaluate every stage of a RAG pipeline: chunking, embedding, retrieval,
reranking, context, answers and the end-to-end result.

Two ways, pick one:

    python examples/evaluate_stages.py
        Scores the logged data in examples/rag_evaluation_sample.json.
        Works for any RAG system and needs no model: data in, report out.

    python examples/evaluate_stages.py --live
        Runs rag-titan on examples/sample_docs/ with the golden questions in
        examples/golden_questions.json, then scores what really happened.
        Before running:  ollama pull nomic-embed-text && ollama pull gemma3:4b

Both write rag_report.json and rag_report.md in the current folder.
"""

import json
import sys
from pathlib import Path

from rag_titan import RAGTitan
from rag_titan.evaluation import RAGEvaluator, ReportManager

HERE = Path(__file__).parent


def evaluate_logged_data():
    """Score a data file: the outputs your own RAG system logged, plus gold labels."""
    dataset = json.loads((HERE / "rag_evaluation_sample.json").read_text(encoding="utf-8"))
    return RAGEvaluator(k=3).evaluate(**dataset)


def evaluate_live_pipeline():
    """Let rag-titan answer the golden questions and score every stage of the run."""
    files = sorted(str(path) for path in (HERE / "sample_docs").glob("*.md"))
    rag = RAGTitan(rag_type="simple")
    return rag.evaluate_stages(files, HERE / "golden_questions.json", include_embeddings=True,
                               show=False)


def main():
    """Evaluate, print the scorecard and save the reports."""
    report = evaluate_live_pipeline() if "--live" in sys.argv else evaluate_logged_data()

    overall = report["scorecard"]["overall"]
    print(f"Overall: {overall['score']}/100 ({overall['rating']}), status: {overall['status']}")
    for stage in report["scorecard"]["stages"].values():
        print(f"  {stage['name']:<22} {stage['score']}")
    print("First failing stage:", report["root_causes"]["first_failing_stage"] or "none")

    manager = ReportManager()
    manager.save_json(report, "rag_report.json")
    manager.save_markdown(report, "rag_report.md")
    print("Saved rag_report.json and rag_report.md")


if __name__ == "__main__":
    main()
