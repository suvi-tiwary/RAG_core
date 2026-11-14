"""Turn the logs of your RAG + your golden questions into one evaluation file."""

import json
from pathlib import Path

from rag_titan.evaluation import build_dataset

logs = json.loads(Path("my_rag_logs.json").read_text(encoding="utf-8"))

# one dict per question: what YOUR RAG did, under rag-titan's names
runs = [{"question": run["question"],
         "retrieved_ids": run["hits"],              # search results, best first
         "context_ids": run["prompt_chunks"],       # chunks put into the prompt
         "answer": run["answer"],                   # what your RAG answered
         "abstained": run["answer"].startswith("I don't know"),
         "retrieval_latency_ms": run["search_ms"],
         "generation_latency_ms": run["llm_ms"],
         "latency_ms": run["search_ms"] + run["llm_ms"]} for run in logs["runs"]]

data = build_dataset(
    chunks=logs["chunks"],
    runs=runs,
    golden="golden.json",
    documents={"travel_policy.md":
               Path("documents/travel_policy.md").read_text(encoding="utf-8")},
)
Path("my-data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print(f"Wrote my-data.json: {len(data['queries'])} queries, {len(data['chunks'])} chunks")
