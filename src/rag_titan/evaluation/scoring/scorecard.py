"""
ScoreAggregator: turn a RAGEvaluator report into 0 to 100 scores you can show anyone.

    card = ScoreAggregator().evaluate(report)
    print(card["overall"]["score"], card["overall"]["rating"])     # 87.4 Very Good
    print(card["stages"]["retrieval"]["score"])

    Overall = Σ stage weight × coverage × stage score / Σ stage weight × coverage

A stage counts for less when only part of it was measured. Ratings describe
the test set you supplied; they are not a certificate that a system is ready.
"""

from rag_titan.evaluation.scoring.policy import (
    METRICS,
    QUALITY_GATES,
    STAGE_NAMES,
    STAGE_WEIGHTS,
    TARGET_METRICS,
    rating,
)
from rag_titan.evaluation.scoring.stage_score import score_stage, stage_rules
from rag_titan.evaluation.stages.text import is_number

METHOD = ("Weighted normalized metrics; metrics that were not measured are left out of the "
          "score and shown in the coverage. The overall score weights each stage by its "
          "weight x measured coverage. Ratings describe the supplied test set, not "
          "production readiness.")


class ScoreAggregator:
    """Combines the stage metrics into stage scores, an overall score and a status."""

    def __init__(self, stage_weights=None, metric_weights=None, targets=None,
                 quality_gates=None, stages=None, min_coverage=0.75):
        """
        Args:
            stage_weights: {stage: weight} to change how much a stage counts.
            metric_weights: {stage: {metric: weight}}; weight 0 switches a metric off.
            targets: {stage: {metric: target}} for sizes, latency and cost, e.g.
                {"end_to_end": {"p95_latency_ms": 2000}}.
            quality_gates: {stage: {metric: minimum}} replacing the default gates.
            stages: the stages to score, e.g. ["retrieval"]; default is all.
            min_coverage: below this measured share the score is "Provisional".
        """
        self.stages = list(STAGE_WEIGHTS) if stages is None else list(stages)
        unknown = set(self.stages) - set(STAGE_WEIGHTS)
        if unknown or not self.stages or len(self.stages) != len(set(self.stages)):
            raise ValueError(f"stages must be unique names from {list(STAGE_WEIGHTS)}, "
                             f"got {self.stages}")
        if not is_number(min_coverage) or not 0 <= min_coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1")

        self.stage_weights = {stage: STAGE_WEIGHTS[stage] for stage in self.stages}
        self.stage_weights.update(stage_weights or {})
        self.metric_weights = metric_weights or {}
        self.targets = targets or {}
        self.quality_gates = QUALITY_GATES if quality_gates is None else quality_gates
        self.min_coverage = min_coverage
        self._check_weights()

    def _check_weights(self):
        """Raise a clear error for a weight that is unknown or not a valid number."""
        for stage, weight in self.stage_weights.items():
            if stage not in self.stages or not is_number(weight) or weight <= 0:
                raise ValueError(f"Stage weight of {stage!r} must be a number above 0 "
                                 "for a scored stage")

        for stage, weights in self.metric_weights.items():
            known = set(METRICS.get(stage, {})) | set(TARGET_METRICS.get(stage, {}))
            for name, weight in weights.items():
                if name not in known or not is_number(weight) or weight < 0:
                    raise ValueError(f"Invalid metric weight for {stage}.{name}")

    def evaluate(self, report):
        """
        Score a report from RAGEvaluator (or a dict of {stage: stage result}).

        Returns:
            {"overall": score, rating, status, coverage, gate failures and the
            stages to review, "stages": one entry per stage, "stage_weights", "method"}.
        """
        cards, weighted, measured = {}, 0.0, 0.0
        for stage in self.stages:
            rules = stage_rules(stage, self.metric_weights, self.targets)
            card, score, coverage = score_stage(stage, report.get(stage) or {}, rules,
                                                self.targets.get(stage, {}), self.min_coverage)
            cards[stage] = card
            measured += self.stage_weights[stage] * coverage
            weighted += self.stage_weights[stage] * coverage * (score or 0.0)

        score = weighted / measured if measured else None
        coverage = measured / sum(self.stage_weights.values())
        return {"overall": self._overall(score, coverage, cards, self._gates(report)),
                "stages": cards, "stage_weights": dict(self.stage_weights), "method": METHOD}

    def _gates(self, report):
        """Return the quality gates that failed: measured metrics below their minimum."""
        failures = []
        for stage, gates in self.quality_gates.items():
            if stage not in self.stages:
                continue
            result = report.get(stage) or {}
            values = result.get("summary", result)
            for metric, minimum in gates.items():
                value = values.get(metric)
                if is_number(value) and value < minimum:
                    failures.append({
                        "stage": stage, "metric": metric, "observed": value, "minimum": minimum,
                        "message": f"{STAGE_NAMES[stage][0]}: {metric.replace('_', ' ')} "
                                   "is below the recommended minimum."})
        return failures

    def _overall(self, score, coverage, cards, gates):
        """Build the "overall" entry: score, rating, status and what to review first."""
        label, message = rating(None if score is None else score * 100)
        provisional = coverage < self.min_coverage

        if score is None:
            status = "Not Evaluated"
        elif gates:
            status = "Action Required"
            message += " One or more critical quality checks failed."
        elif provisional:
            status = "Provisional"
        else:
            status = "Evaluated"
            message += " Review individual stages before deciding whether to deploy."
        if provisional and score is not None:
            message += " This is a provisional score because evaluation data is incomplete."

        scored = [stage for stage in self.stages if cards[stage]["score"] is not None]
        weakest = sorted(scored, key=lambda stage: cards[stage]["score"])[:3]
        return {
            "score": None if score is None else round(score * 100, 2),
            "score_0_1": None if score is None else round(score, 4),
            "rating": label, "status": status, "message": message,
            "evaluation_coverage": round(coverage, 4), "is_provisional": provisional,
            "evaluated_stages": len(scored), "total_stages": len(self.stages),
            "quality_gate_failures": gates,
            "areas_to_review": [{"stage": stage, "label": cards[stage]["name"],
                                 "score": cards[stage]["score"],
                                 "improvement_areas": cards[stage]["improvement_areas"]}
                                for stage in weakest],
        }
