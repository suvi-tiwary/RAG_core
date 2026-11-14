"""
Compare reports: find regressions against a baseline and trends over many runs.

    regressions = find_regressions(report, baseline)     # what got worse since last time
    changes = trends([monday, tuesday, wednesday])       # first-to-last change per metric

A metric is named "stage.metric", e.g. "retrieval.recall_at_k". Whether higher
or lower is better comes from the scoring policy, so both use the same rules.
"""

from rag_titan.evaluation.scoring.policy import METRICS, STAGE_WEIGHTS
from rag_titan.evaluation.stages.text import is_number

LOWER_DIRECTIONS = {"low", "ratio", "fragmentation", "cos_low"}

# telemetry is compared by relative change: +2% of 900 ms, not +0.02 ms
TELEMETRY = {"p50_latency_ms", "p95_latency_ms", "mean_cost_usd", "mean_total_tokens"}

# timings jitter from run to run: a latency only counts as worse when it is
# more than 10% AND more than 5 ms slower (7.2 -> 7.55 ms is noise, not a regression)
LATENCY_SHARE = 0.10
LATENCY_MIN_MS = 5.0


def flatten(report):
    """Return every numeric stage metric of a report as {"stage.metric": value}."""
    if not isinstance(report, dict):
        raise ValueError("A report must be a dict")

    flat = {}
    for stage in STAGE_WEIGHTS:
        result = report.get(stage)
        if not isinstance(result, dict):
            continue
        for name, value in result.get("summary", result).items():
            if is_number(value):
                flat[f"{stage}.{name}"] = value
    return flat


def better_direction(name):
    """
    Return "higher" or "lower" for a "stage.metric" name, or None when neither
    is better (counts and sizes such as chunk_count or mean_chunk_tokens).
    """
    stage, _, metric = name.partition(".")
    if metric in TELEMETRY:
        return "lower"
    if metric not in METRICS.get(stage, {}):
        return None
    direction = METRICS[stage][metric][0]
    return "lower" if direction in LOWER_DIRECTIONS else "higher"


def find_regressions(report, baseline, tolerance=0.02, latency_share=LATENCY_SHARE,
                     latency_min_ms=LATENCY_MIN_MS):
    """
    List the metrics that got worse than in the baseline report.

    Args:
        report: the new report.
        baseline: an earlier report to compare with.
        tolerance: how much worse still counts as noise. For scores it is an
            absolute change (0.02 = 2 points of 100); for latency, cost and
            tokens a relative one (0.02 = 2% slower or more expensive).
        latency_share: a latency must also be this share slower (0.10 = 10%).
        latency_min_ms: ... and this many milliseconds slower. Set both to 0
            to count every slower timing.

    Returns:
        A list of {"metric", "baseline", "current", "delta"}, in pipeline order.
    """
    for label, value in (("tolerance", tolerance), ("latency_share", latency_share),
                         ("latency_min_ms", latency_min_ms)):
        if not is_number(value) or value < 0:
            raise ValueError(f"{label} must be a number of 0 or more, got {value!r}")

    current, previous = flatten(report), flatten(baseline)
    found = []
    for name, value in current.items():
        direction = better_direction(name)
        if name not in previous or direction is None:
            continue

        old = previous[name]
        worse_by = value - old if direction == "lower" else old - value
        metric = name.partition(".")[2]
        allowed = tolerance * abs(old) if metric in TELEMETRY else tolerance
        if metric.endswith("latency_ms"):
            allowed = max(allowed, latency_share * abs(old), latency_min_ms)
        if worse_by > allowed:
            found.append({"metric": name, "baseline": old, "current": value,
                          "delta": value - old})
    return found


def trends(reports):
    """
    Return the change of every metric from the first to the last report.

    Args:
        reports: report dicts, oldest first.

    Returns:
        {"stage.metric": {"first", "last", "change", "improved", "runs"}} for the
        metrics present in every report. "improved" is None when neither
        direction is better. Empty with fewer than two reports.
    """
    history = [flatten(report) for report in reports]
    if len(history) < 2:
        return {}

    shared = set(history[0]).intersection(*history[1:])
    result = {}
    for name in sorted(shared):
        change = history[-1][name] - history[0][name]
        direction = better_direction(name)
        improved = None
        if direction and change:
            improved = (change < 0) == (direction == "lower")
        result[name] = {"first": history[0][name], "last": history[-1][name],
                        "change": change, "improved": improved, "runs": len(history)}
    return result
