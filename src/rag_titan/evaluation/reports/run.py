"""
Evaluate a data file and write the reports: what `rag-titan evaluate --input` runs.

    report = evaluate_file("data.json", "reports/today.json", k=5,
                           baseline="reports/yesterday.json")
    print(summary_text(report))

The data file is a JSON object with the inputs of RAGEvaluator.evaluate():
"chunks", "queries", "document_text" or "documents", "embeddings", ...
"""

import json
from pathlib import Path

from rag_titan.evaluation.rag_evaluator import RAGEvaluator
from rag_titan.evaluation.reports.hints import not_measured, queries_to_look_at
from rag_titan.evaluation.reports.manager import ReportManager


def load_dataset(path):
    """Read a data file and return its dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON (line {error.lineno}): {error.msg}")
    if not isinstance(dataset, dict):
        raise ValueError(f"{path} must hold a JSON object with 'chunks', 'queries', ...")
    return dataset


def evaluate_file(input_path, output_path="rag_report.json", k=5, baseline=None):
    """
    Evaluate a data file and save the report as JSON and Markdown.

    Args:
        input_path: the JSON data file.
        output_path: where to write the JSON report; the Markdown report gets
            the same name ending in .md.
        k: how many results to score in the ranking metrics.
        baseline: path of an earlier JSON report; the metrics that got worse
            are added to the report as "regressions".

    Returns:
        The report dict.
    """
    manager = ReportManager()
    report = RAGEvaluator(k=k).evaluate(**load_dataset(input_path))
    if baseline:
        report["regressions"] = manager.regressions(report, manager.load_json(baseline))

    manager.save_json(report, output_path)
    manager.save_markdown(report, Path(output_path).with_suffix(".md"))
    return report


def summary_text(report):
    """
    Return the lines printed after an evaluation: score, status, first failure,
    the queries to look at, regressions and what was not measured.
    """
    overall = report["scorecard"]["overall"]
    score = "N/A" if overall["score"] is None else f"{overall['score']:.1f}/100"
    lines = [
        f"Overall score: {score} ({overall['rating']})",
        f"Status: {overall['status']} | Measurement coverage: "
        f"{overall['evaluation_coverage']:.0%}",
        f"First failing stage: {report['root_causes']['first_failing_stage'] or 'none'}",
    ]
    # a wrong answer must be visible here, not only at the end of the .md report
    lines += _section("Queries to look at:", queries_to_look_at(report))
    if "regressions" in report:
        names = ", ".join(item["metric"] for item in report["regressions"]) or "none"
        lines.append(f"Regressions against the baseline: {names}")
    lines += _section("Not measured (null in the report) - to measure it:", not_measured(report))
    return "\n".join(lines)


def _section(title, items):
    """A title and its items indented below it, or nothing when there are no items."""
    return [title] + [f"  - {item}" for item in items] if items else []
