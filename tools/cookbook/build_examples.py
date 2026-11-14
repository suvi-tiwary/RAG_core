"""Build reproducible cookbook downloads and browser snapshots with the real evaluator.

Run: python tools/cookbook/build_examples.py [--check]
The two-dimensional vectors and answers are teaching fixtures, not live model outputs.
"""

import copy
import io
import json
import sys
import zipfile
from pathlib import Path

from rag_titan import __version__
from rag_titan.evaluation import RAGEvaluator, check_golden
from rag_titan.evaluation.reports import summary_text

sys.path.insert(0, str(Path(__file__).resolve().parent))    # tools/ is not a package
from your_rag_example import your_rag_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "docs/assets/cookbook"


def scenarios():
    """Derive explicit failure cases from the existing tested sample."""
    sample = json.loads((ROOT / "examples/rag_evaluation_sample.json").read_text())
    missed, wrong, missing = [copy.deepcopy(sample) for _ in range(3)]
    query = missed["queries"][0]
    query.update(retrieved_ids=["c2", "c3"], reranked_ids=["c2", "c3"],
                 context_ids=["c2"], answer="I don't know.", citations=[],
                 abstained=True, task_success=False, verified_claims=[])
    for query, answer in zip(wrong["queries"], ["Annual leave is unlimited.",
                                               "Remote work is forbidden."]):
        query.update(answer=answer, citations=["c3"], task_success=False,
                     verified_claims=[{"text": answer, "supported": False}])
    for query in missing["queries"]:
        for key in ("expected_answer", "required_facts", "expected_citations",
                    "verified_claims", "task_success", "answerable"):
            query.pop(key, None)
    return {"sample": sample, "missed-evidence": missed,
            "wrong-answer": wrong, "missing-labels": missing}


def encoded(value):
    """Make stable, readable JSON bytes."""
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def golden_files():
    """Provide a complete tiny handbook and two reviewed questions for live runs."""
    handbook = ("# Example company handbook\n\nEmployees get 25 days of annual leave.\n\n"
                "Sick pay is full salary for 10 days.\n\nTravel needs manager approval.\n\n"
                "Remote work is allowed 3 days a week.\n")
    questions = []
    for number, (question, answer, fact) in enumerate([
        ("How many days of annual leave do employees get?",
         "Employees get 25 days of annual leave.", "25"),
        ("How many days a week can employees work remotely?",
         "Remote work is allowed 3 days a week.", "3"),
    ], 1):
        questions.append({"id": f"q{number}", "question": question,
                          "expected_answer": answer, "expected_evidence": [answer],
                          "required_facts": [fact], "answerable": True, "validated": True})
    assert check_golden(questions, {"handbook.md": handbook})["ready"]
    return {"handbook.md": handbook.encode(), "golden.json": encoded(questions)}


def generated_files():
    """Execute every case for K=1..3; the UI only displays these package results."""
    cases = scenarios()
    files = {f"{name}.json": encoded(data) for name, data in cases.items()}
    files.update(golden_files())
    snapshots = {"version": __version__, "cases": {}}
    for name, data in cases.items():
        snapshots["cases"][name] = {"query": data["queries"][0], "runs": {}}
        for k in (1, 2, 3):
            report = RAGEvaluator(k=k).evaluate(**data)
            snapshots["cases"][name]["runs"][str(k)] = {
                "overall": report["scorecard"]["overall"],
                "root_causes": report["root_causes"],
                "summaries": {stage: result["summary"] for stage, result in report.items()
                              if "summary" in result},
            }
            if k == 3:
                files[f"expected/{name}-report.json"] = encoded(report)
                files[f"expected/{name}-summary.txt"] = (summary_text(report) + "\n").encode()
    files["snapshots.json"] = encoded(snapshots)
    minimal = {"chunks": [{"id": "c1", "text": "Employees get 25 days of annual leave."}],
               "queries": [{"id": "q1", "expected_chunk_ids": ["c1"],
                            "retrieved_ids": ["c1"], "context_ids": ["c1"],
                            "answer": "Employees get 25 days of annual leave.",
                            "expected_answer": "Employees get 25 days of annual leave."}]}
    files["minimal.json"] = encoded(minimal)
    files.update(your_rag_files(files["minimal.json"]))    # the complete guide's example
    files["README.txt"] = (
        "rag-titan evaluation cookbook\n\n"
        "Teaching data: fictional company policies, hand-written answers and toy vectors.\n"
        "Scores are computed by the real package, not by an LLM judge.\n\n"
        f'Install: python -m pip install "rag-titan=={__version__}"\n'
        "Run: rag-titan evaluate --input sample.json --output baseline.json --k 3\n"
        "Inspect: python inspect_report.py baseline.json\n"
        "Compare: rag-titan evaluate --input missed-evidence.json --output candidate.json "
        "--baseline baseline.json --k 3 --fail-on-regression\n"
        "The comparison intentionally exits with code 1.\n\n"
        "Optional live recipe: pull gemma3:4b and nomic-embed-text in Ollama, then run\n"
        "python live_evaluation.py. It uses real models; scores and timings vary.\n\n"
        "your-rag/: the complete guide, step by step, on a tiny stand-in RAG (no model\n"
        "needed). Its expected/ folder holds the real output of every command.\n"
        "Guide: https://aion7891.github.io/rag-titan/cookbook/\n"
    ).encode()
    return files


def bundle(files):
    """Make a deterministic archive with no timestamps from the host system."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(f"rag-titan-cookbook/{name}", date_time=(2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return output.getvalue()


def main():
    """Write generated assets, or fail if the committed assets are out of date."""
    files = generated_files()
    scripts = {path.name: path.read_bytes() for path in DEST.glob("*.py")}
    files["rag-titan-cookbook.zip"] = bundle({**files, **scripts})
    for name, content in files.items():
        path = DEST / name
        if "--check" in sys.argv:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"Out of date: {path}; run tools/cookbook/build_examples.py")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(f"Cookbook assets {'verified' if '--check' in sys.argv else 'built'}: {len(files)}")


if __name__ == "__main__":
    main()
