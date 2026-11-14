"""Reports: save evaluation reports as JSON or Markdown and compare them over time."""

from rag_titan.evaluation.reports.compare import find_regressions, flatten, trends
from rag_titan.evaluation.reports.failures import failed_queries
from rag_titan.evaluation.reports.manager import ReportManager
from rag_titan.evaluation.reports.markdown import to_markdown
from rag_titan.evaluation.reports.run import evaluate_file, summary_text

__all__ = [
    "ReportManager",
    "evaluate_file",
    "failed_queries",
    "find_regressions",
    "flatten",
    "summary_text",
    "to_markdown",
    "trends",
]
