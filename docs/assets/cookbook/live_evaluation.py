"""Run a local Ollama RAG, save its logged outputs, and evaluate them.

Run from the extracted cookbook folder: python live_evaluation.py
Requires Ollama with gemma3:4b and nomic-embed-text already downloaded.
"""

import json
from pathlib import Path

from rag_titan import RAGTitan
from rag_titan.evaluation import RAGEvaluator, ReportManager, check_golden, collect_dataset
from rag_titan.evaluation.reports import summary_text


def main():
    """Validate labels, run the real models, and save data plus reports."""
    here = Path(__file__).resolve().parent
    handbook = here / "handbook.md"
    golden = here / "golden.json"
    check = check_golden(golden, {handbook.name: handbook.read_text(encoding="utf-8")})
    if not check["ready"]:
        raise ValueError(f"Review your golden questions first: {check['problems']}")

    rag = RAGTitan(rag_type="simple", settings={
        "ollama_host": "http://localhost:11434",
        "llm_provider": "ollama", "llm_model": "gemma3:4b",
        "embed_provider": "ollama", "embed_model": "nomic-embed-text",
        "chunker": "paragraph", "chunk_size": 60, "chunk_overlap": 0,
        "vector_store": "memory", "reranker": "keyword", "generator": "grounded",
    })
    dataset = collect_dataset(rag, [str(handbook)], golden, include_embeddings=True)
    Path("live-data.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    report = RAGEvaluator(k=3).evaluate(**dataset)
    manager = ReportManager()
    manager.save_json(report, "live-report.json")
    manager.save_markdown(report, "live-report.md")
    print(summary_text(report))
    print("Saved live-data.json, live-report.json and live-report.md")


if __name__ == "__main__":
    main()
