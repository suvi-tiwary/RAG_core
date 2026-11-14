rag-titan evaluation cookbook

Teaching data: fictional company policies, hand-written answers and toy vectors.
Scores are computed by the real package, not by an LLM judge.

Install: python -m pip install "rag-titan==0.8.5"
Run: rag-titan evaluate --input sample.json --output baseline.json --k 3
Inspect: python inspect_report.py baseline.json
Compare: rag-titan evaluate --input missed-evidence.json --output candidate.json --baseline baseline.json --k 3 --fail-on-regression
The comparison intentionally exits with code 1.

Optional live recipe: pull gemma3:4b and nomic-embed-text in Ollama, then run
python live_evaluation.py. It uses real models; scores and timings vary.

your-rag/: the complete guide, step by step, on a tiny stand-in RAG (no model
needed). Its expected/ folder holds the real output of every command.
Guide: https://aion7891.github.io/rag-titan/cookbook/
