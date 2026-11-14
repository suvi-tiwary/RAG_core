"""
Score one stage: combine its metrics into a single 0 to 100 number.

    Stage score = Σ weight_m × score_m / Σ weight_m      over the measured metrics m

A metric that was not measured is left out, never counted as 0. How much of
the stage was really measured is shown separately as "evaluation_coverage":

    Coverage = Σ weight of measured metrics / Σ weight of all the stage's metrics
"""

from rag_titan.evaluation.scoring.explanations import weakest_metrics
from rag_titan.evaluation.scoring.normalize import normalize
from rag_titan.evaluation.scoring.policy import METRICS, STAGE_NAMES, TARGET_METRICS, rating


def stage_rules(stage, metric_weights, targets):
    """
    Return {metric: (direction, weight)} for a stage.
    Target metrics (sizes, latency, cost) are included only when a target was given.
    """
    rules = dict(METRICS[stage])
    for name, weight in TARGET_METRICS.get(stage, {}).items():
        if name in targets.get(stage, {}):
            rules[name] = ("target", weight)

    custom = metric_weights.get(stage, {})
    final = {}
    for name, (direction, weight) in rules.items():
        weight = custom.get(name, weight)
        if weight > 0:     # weight 0 switches a metric off
            final[name] = (direction, weight)
    return final


def score_metrics(values, rules, targets):
    """
    Normalise every measured metric of a stage.

    Returns:
        (details, missing, warnings): the breakdown per scored metric, the
        names that were not measured, and values that could not be scored.
    """
    details, missing, warnings = {}, [], []
    for name, (direction, weight) in rules.items():
        value = values.get(name)
        if value is None:
            missing.append(name)
            continue
        try:
            score = normalize(value, direction, targets.get(name))
        except ValueError as error:
            warnings.append(f"{name}: {error}")
            missing.append(name)
            continue
        details[name] = {"value": value, "normalized_score": round(score, 4),
                         "weight": weight, "direction": direction,
                         "contribution": weight * score}
    return details, missing, warnings


def score_stage(stage, result, rules, targets, min_coverage):
    """
    Build the scorecard entry of one stage.

    Args:
        stage: the stage name, e.g. "retrieval".
        result: the stage's result from its evaluator ({} when it did not run).
        rules: the dict from stage_rules().
        targets: {metric: target} of this stage.
        min_coverage: below this coverage the message says data is incomplete.

    Returns:
        (card, score, coverage): the entry, and the unrounded 0 to 1 score
        (None when nothing was measured) and coverage for the overall score.
    """
    values = result.get("summary", result) if isinstance(result, dict) else {}
    details, missing, warnings = score_metrics(values, rules, targets)

    total = sum(weight for _, weight in rules.values())
    used = sum(info["weight"] for info in details.values())
    score = sum(info["contribution"] for info in details.values()) / used if used else None
    coverage = used / total if total else 0.0

    label, message = rating(None if score is None else score * 100)
    improvements = weakest_metrics(details)
    if improvements:
        names = ", ".join(item["metric"].replace("_", " ") for item in improvements[:2])
        message += f" Main improvement areas: {names}."
    if score is not None and coverage < min_coverage:
        message += " Only some configured metrics were available."

    card = {
        "name": STAGE_NAMES[stage][0], "description": STAGE_NAMES[stage][1],
        "score": None if score is None else round(score * 100, 2),
        "score_0_1": None if score is None else round(score, 4),
        "rating": label, "message": message,
        "evaluation_coverage": round(coverage, 4),
        "metrics_scored": len(details), "metrics_expected": len(rules),
        "missing_metrics": missing, "metric_breakdown": details,
        "improvement_areas": improvements, "warnings": warnings,
    }
    return card, score, coverage
