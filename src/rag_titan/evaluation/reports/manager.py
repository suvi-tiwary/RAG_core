"""
ReportManager: save, load, print and compare evaluation reports.

    manager = ReportManager()
    manager.save_json(report, "reports/today.json")
    manager.save_markdown(report, "reports/today.md")

    baseline = manager.load_json("reports/last_week.json")
    for item in manager.regressions(report, baseline):
        print(item["metric"], item["baseline"], "->", item["current"])
"""

import json
from pathlib import Path

from rag_titan.evaluation.reports.compare import (
    LATENCY_MIN_MS,
    LATENCY_SHARE,
    find_regressions,
    flatten,
    trends,
)
from rag_titan.evaluation.reports.failures import failed_queries
from rag_titan.evaluation.reports.markdown import to_markdown


class ReportManager:
    """One place for everything you do with a finished report."""

    def save_json(self, report, path):
        """Write the report as JSON (folders are created) and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_json(self, path):
        """Read a report saved with save_json()."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Report file not found: {path}")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not valid JSON (line {error.lineno}): {error.msg}")
        if not isinstance(report, dict):
            raise ValueError(f"{path} does not hold a report (a JSON object)")
        return report

    def markdown(self, report, title="RAG Pipeline Evaluation", generated=None):
        """Return the report as Markdown text (see to_markdown)."""
        return to_markdown(report, title, generated)

    def save_markdown(self, report, path, title="RAG Pipeline Evaluation"):
        """Write the report as Markdown (folders are created) and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.markdown(report, title), encoding="utf-8")
        return path

    def flatten(self, report):
        """Return every numeric stage metric as {"stage.metric": value}."""
        return flatten(report)

    def regressions(self, report, baseline, tolerance=0.02, latency_share=LATENCY_SHARE,
                    latency_min_ms=LATENCY_MIN_MS):
        """List the metrics that got worse than in the baseline (see find_regressions)."""
        return find_regressions(report, baseline, tolerance, latency_share, latency_min_ms)

    def trends(self, reports):
        """Return the first-to-last change of every metric over reports, oldest first."""
        return trends(reports)

    def failed_queries(self, report):
        """Return the queries that failed a check, earliest stage first."""
        return failed_queries(report)
