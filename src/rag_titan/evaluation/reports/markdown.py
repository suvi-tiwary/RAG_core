"""
Write a report as Markdown: scorecard first, then the numbers behind it.

    text = to_markdown(report)
    Path("report.md").write_text(text, encoding="utf-8")
"""

from datetime import datetime, timezone

from rag_titan.evaluation.reports.failures import failed_queries
from rag_titan.evaluation.scoring.policy import STAGE_NAMES, STAGE_WEIGHTS
from rag_titan.evaluation.stages.text import is_number

DISCLAIMER = ("*Ratings apply only to the measured criteria on this test set. Measurement "
              "coverage is not statistical confidence. A high score is not a production "
              "certification.*")

# how many failed queries the report lists
MAX_FAILURES = 20


def to_markdown(report, title="RAG Pipeline Evaluation", generated=None):
    """
    Return the report as Markdown text.

    Args:
        report: a report from RAGEvaluator. A "regressions" entry is shown when present.
        title: the heading of the report.
        generated: the timestamp text; default is the current UTC time.
    """
    if not isinstance(report, dict):
        raise ValueError("A report must be a dict")
    generated = generated or datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [f"# {title}", "", f"Generated: {generated}", ""]
    if report.get("scorecard"):
        lines += _scorecard_lines(report["scorecard"])
    if "regressions" in report:
        lines += _regression_lines(report["regressions"])
    for stage in STAGE_WEIGHTS:
        if isinstance(report.get(stage), dict):
            lines += _stage_lines(stage, report[stage])
    lines += _failure_lines(failed_queries(report))
    lines += _hint_lines(report.get("root_causes") or {})
    return "\n".join(lines) + "\n"


def _scorecard_lines(card):
    """The executive scorecard: overall score, one row per stage, what to improve."""
    overall = card["overall"]
    score = "N/A" if overall["score"] is None else f"{overall['score']:.1f}/100"
    lines = ["## Executive scorecard", "",
             f"**Overall: {score} - {overall['rating']}**", "",
             f"**Assessment: {overall['status']}** · "
             f"**Measurement coverage: {overall['evaluation_coverage']:.0%}**", "",
             overall["message"], "",
             "| Pipeline stage | Score / 100 | Rating | Measurement coverage |",
             "|---|---:|---|---:|"]
    for stage in card["stages"].values():
        stage_score = "N/A" if stage["score"] is None else f"{stage['score']:.1f}"
        lines.append(f"| {stage['name']} | {stage_score} | {stage['rating']} | "
                     f"{stage['evaluation_coverage']:.0%} |")

    lines += ["", "### Improvement priorities", ""]
    for area in overall["areas_to_review"]:
        lines.append(f"- {area['label']}: {area['score']:.1f}/100")
        lines += [f"  - {item['reason']}" for item in area["improvement_areas"][:2]]
    for gate in overall["quality_gate_failures"]:
        lines.append(f"- **Quality check:** {gate['message']}")
    return lines + ["", DISCLAIMER, ""]


def _stage_lines(stage, result):
    """One table with every measured number of a stage."""
    lines = [f"## {stage.replace('_', ' ').capitalize()} ({STAGE_NAMES[stage][0]})", "",
             "| Metric | Value |", "|---|---:|"]
    for name, value in result.get("summary", result).items():
        if is_number(value):
            shown = str(value) if isinstance(value, int) else f"{value:.4f}"
            lines.append(f"| {name} | {shown} |")
    for warning in result.get("warnings") or []:
        lines.append(f"\n> Warning: {warning}")
    return lines + [""]


def _regression_lines(regressions):
    """Metrics that got worse than the baseline report."""
    lines = ["## Regressions against the baseline", ""]
    if not regressions:
        return lines + ["No metric got worse than the baseline.", ""]

    lines += ["| Metric | Baseline | Current | Change |", "|---|---:|---:|---:|"]
    for item in regressions:
        lines.append(f"| {item['metric']} | {item['baseline']:.4f} | {item['current']:.4f} | "
                     f"{item['delta']:+.4f} |")
    return lines + [""]


def _failure_lines(failures):
    """Queries that failed a check, earliest stage first."""
    if not failures:
        return []
    lines = ["## Queries to look at", ""]
    for item in failures[:MAX_FAILURES]:
        lines.append(f"- `{item['query_id']}` ({item['stage']}): {item['reason']}")
    if len(failures) > MAX_FAILURES:
        lines.append(f"- ... and {len(failures) - MAX_FAILURES} more (see the JSON report)")
    return lines + [""]


def _hint_lines(causes):
    """Root-cause hints: the thresholds that failed, in pipeline order."""
    lines = ["## Diagnostic hints", ""]
    failures = causes.get("failures") or []
    if not failures:
        return lines + ["No threshold failure was detected."]

    lines.append(f"First failing stage: **{causes['first_failing_stage']}**")
    lines.append("")
    for item in failures:
        lines.append(f"- {item['stage']} / {item['metric']} = {item['observed']:.2f} "
                     f"(minimum {item['minimum']}): {item['suggestion']}")
    return lines
